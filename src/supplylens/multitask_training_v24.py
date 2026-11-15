"""V2.4 joint multi-currency attention CNN-LSTM ensemble training."""
import json
import hashlib
import platform
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from .config import (
    PROJECT_ROOT, DATA_RAW_DIR, DISPLAY_HORIZONS,
    WALK_FORWARD_SPACING, MIN_SELECTION_ORIGINS,
    CALIBRATION_ORIGIN_STRIDE, CALIBRATION_MIN_SAMPLES,
    SPLIT_FRACTIONS, V24_LOOKBACK, FORECAST_HORIZON,
    PRIORITY_CURRENCIES, V24_ENSEMBLE_SEEDS,
    V24_AUXILIARY_LOSS_WEIGHT, V24_AUXILIARY_BLEND,
)
from .data import load_exchange_rates
from .multivariate import make_multivariate_supervised_samples_v24
from .model_v24 import build_joint_cnn_lstm_v24
from .losses_v24 import HorizonBalancedHuber
from .v24_utils import decode_v24_currency_prediction
from .ensemble import PredictionEnsemble
from .multitask_training import (
    _resolve_training_device,
    _matched_selection_metrics,
    _atomic_csv,
    _atomic_json,
)
from .evaluation import (
    evaluate_walk_forward,
    select_benchmark_methods,
    calibrate_method_forecast_intervals,
)


V24_MODEL_VERSION = "v2_4"


def _scale_X(X, scaler):
    shape = X.shape
    transformed = scaler.transform(X.reshape(-1, shape[-1]))
    return transformed.reshape(shape).astype(np.float32)


def _fit_and_scale_targets(partitions, currencies, horizon, block_size):
    """Fit each target scaler on daily-return training targets only."""
    scalers_y = {}
    for index, currency in enumerate(currencies):
        start = index * block_size
        daily_train = partitions["train"]["y"][
            :, start:start + int(horizon)
        ].reshape(-1, 1)
        scalers_y[currency] = StandardScaler().fit(daily_train)

    scaled = {}
    for name, part in partitions.items():
        raw = np.asarray(part["y"], dtype=np.float32)
        result = np.empty_like(raw, dtype=np.float32)
        for index, currency in enumerate(currencies):
            start = index * block_size
            daily_stop = start + int(horizon)
            aux_stop = start + block_size
            scaler = scalers_y[currency]
            result[:, start:daily_stop] = scaler.transform(
                raw[:, start:daily_stop].reshape(-1, 1)
            ).reshape(raw[:, start:daily_stop].shape)
            if aux_stop > daily_stop:
                aux = scaler.transform(
                    raw[:, daily_stop:aux_stop].reshape(-1, 1)
                ).reshape(raw[:, daily_stop:aux_stop].shape)
                result[:, daily_stop:aux_stop] = (
                    aux * float(V24_AUXILIARY_LOSS_WEIGHT)
                )
        scaled[name] = result
    return scalers_y, scaled


class V24TargetOutputAdapter:
    """Expose a currency's adjusted 63-step path to the shared evaluator."""

    def __init__(
        self, model, target_index, scaler_y, horizon, auxiliary_horizons,
        auxiliary_loss_weight=V24_AUXILIARY_LOSS_WEIGHT,
        auxiliary_blend=V24_AUXILIARY_BLEND,
    ):
        self.model = model
        self.target_index = int(target_index)
        self.scaler_y = scaler_y
        self.horizon = int(horizon)
        self.auxiliary_horizons = list(auxiliary_horizons)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.auxiliary_blend = float(auxiliary_blend)

    def predict(self, X, verbose=0):
        raw = self.model.predict(X, verbose=verbose)
        decoded = decode_v24_currency_prediction(
            raw,
            target_index=self.target_index,
            scaler_y=self.scaler_y,
            horizon=self.horizon,
            auxiliary_horizons=self.auxiliary_horizons,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
            auxiliary_blend=self.auxiliary_blend,
        )
        return self.scaler_y.transform(
            decoded.reshape(-1, 1)
        ).reshape(decoded.shape).astype(np.float32)




def _ensemble_dispersion_table(
    models, target_index, scaler_y, X_scaled, currency,
    horizon, auxiliary_horizons,
):
    """Measure member disagreement at the dashboard horizons on held-out origins."""
    member_cumulative = []
    for model in models:
        adapter = V24TargetOutputAdapter(
            model,
            target_index=target_index,
            scaler_y=scaler_y,
            horizon=horizon,
            auxiliary_horizons=auxiliary_horizons,
        )
        scaled_path = adapter.predict(X_scaled, verbose=0)
        raw_path = scaler_y.inverse_transform(
            scaled_path.reshape(-1, 1)
        ).reshape(-1, int(horizon))
        cumulative = np.cumsum(raw_path, axis=1)
        member_cumulative.append(
            cumulative[:, np.asarray(DISPLAY_HORIZONS, dtype=int) - 1]
        )

    values = np.stack(member_cumulative, axis=0)
    rows = []
    for column, horizon_value in enumerate(DISPLAY_HORIZONS):
        dispersion = np.std(values[:, :, column], axis=0, ddof=1)
        rows.append({
            "currency": currency,
            "horizon": int(horizon_value),
            "n_test_origins": int(values.shape[1]),
            "mean_member_std_cumulative_log_return": float(np.mean(dispersion)),
            "median_member_std_cumulative_log_return": float(np.median(dispersion)),
            "p95_member_std_cumulative_log_return": float(np.quantile(dispersion, 0.95)),
            "interpretation": "member disagreement; not a calibrated prediction interval",
        })
    return pd.DataFrame(rows)


def _atomic_joblib(value, path):
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    joblib.dump(value, temporary)
    temporary.replace(path)


def _save_model_atomic(model, path):
    temporary = path.with_name(path.stem + ".part.keras")
    model.save(temporary)
    temporary.replace(path)


def train_joint_multicurrency_model_v24(
    rates=None,
    currencies=None,
    epochs=300,
    batch_size=64,
    lookback=V24_LOOKBACK,
    horizon=FORECAST_HORIZON,
    force_download=False,
    include_arima=True,
    calibration_stride=CALIBRATION_ORIGIN_STRIDE,
    patience=40,
    seeds=None,
    verbose=1,
    device="auto",
    require_gpu=False,
    output_dir=None,
    evaluate_test=False,
    architecture="residual",
    objective="horizon_balanced",
):
    """Train a three-member V2.4 CNN-LSTM ensemble and evaluate its mean forecast.

    V2.4 keeps the V2.2 chronological split discipline and artifact family intact.
    It adds expanded features, temporal attention, currency-specific decoders,
    direct multi-horizon auxiliary targets, and seed ensembling.
    """
    currencies = [str(c).upper() for c in (currencies or PRIORITY_CURRENCIES)]
    if len(currencies) < 2 or len(set(currencies)) != len(currencies):
        raise ValueError("Provide at least two unique currencies in a fixed order.")
    seeds = [int(x) for x in (seeds or V24_ENSEMBLE_SEEDS)]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("V2.4 requires at least two unique ensemble seeds.")
    if int(epochs) < 1 or int(batch_size) < 1 or int(patience) < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")

    if int(horizon) != FORECAST_HORIZON:
        raise ValueError("V2.4 evaluation requires horizon=63.")
    if int(lookback) < 1 or int(calibration_stride) < 1:
        raise ValueError("lookback and calibration_stride must be positive.")
    if architecture not in {"residual", "original"} or objective not in {"horizon_balanced", "original"}:
        raise ValueError("Unknown architecture or objective.")
    if force_download:
        raise ValueError("Training uses a frozen dataset. Supply a separately downloaded BNM rates panel.")
    if rates is None:
        rates = load_exchange_rates(raw_path=DATA_RAW_DIR / "daily_1200.csv")
    run_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "experiments" / (
        datetime.now(timezone.utc).strftime("v24_%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8])
    run_dir = run_dir.resolve()
    protected = [PROJECT_ROOT / "artifacts", PROJECT_ROOT / "data"]
    if any(run_dir == folder.resolve() or folder.resolve() in run_dir.parents for folder in protected):
        raise ValueError("Experiment output must be outside data/ and artifacts/.")
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir, scaler_dir, metadata_dir = [run_dir / part for part in ("models", "scalers", "metadata")]
    for folder in (model_dir, scaler_dir, metadata_dir):
        folder.mkdir()
    rates.to_csv(run_dir / "input_rates.csv")
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (PROJECT_ROOT / "src").rglob("*.py")}
    for relative in source_hashes:
        snapshot = run_dir / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, snapshot)
    run_info = {"status": "started", "architecture": architecture, "objective": objective,
                "evaluate_test": bool(evaluate_test), "currencies": currencies, "seeds": seeds,
                "epochs": int(epochs), "batch_size": int(batch_size), "patience": int(patience),
                "lookback": int(lookback), "horizon": int(horizon), "include_arima": bool(include_arima),
                "calibration_stride": int(calibration_stride), "source_sha256": source_hashes,
                "python": platform.python_version(), "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "data_sha256": hashlib.sha256((run_dir / "input_rates.csv").read_bytes()).hexdigest()}
    _atomic_json(run_info, run_dir / "run.json")
    missing = [currency for currency in currencies if currency not in rates.columns]
    if missing:
        raise ValueError(f"Dataset is missing requested currencies: {missing}")

    data = make_multivariate_supervised_samples_v24(
        rates, currencies, lookback=lookback, horizon=horizon
    )
    partition_names = ["train", "validation", "selection", "calibration", "test"]
    partitions = {name: data[name] for name in partition_names}
    for name, part in partitions.items():
        if len(part["X"]) == 0:
            raise ValueError(f"{name} partition is empty.")

    n_currencies = len(currencies)
    n_features = len(data["feature_columns"])
    block_size = int(data["output_block_size"])
    auxiliary_horizons = list(data["auxiliary_horizons"])

    scaler_X = StandardScaler().fit(
        partitions["train"]["X"].reshape(-1, n_features)
    )
    X_scaled = {
        name: _scale_X(part["X"], scaler_X)
        for name, part in partitions.items()
    }
    scalers_y, y_scaled = _fit_and_scale_targets(
        partitions, currencies, int(horizon), block_size
    )

    training_device, gpu_name = _resolve_training_device(
        device=device, require_gpu=bool(require_gpu)
    )
    keras.backend.clear_session()
    models = []
    history_rows = []

    for member_index, seed in enumerate(seeds, start=1):
        keras.utils.set_random_seed(seed)
        with tf.device(training_device):
            model = build_joint_cnn_lstm_v24(
                lookback=int(lookback),
                n_input_features=n_features,
                n_currencies=n_currencies,
                forecast_horizon=int(horizon),
                auxiliary_horizons=auxiliary_horizons,
                architecture=architecture,
            )
            model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=0.0005,
                    weight_decay=1e-5,
                    clipnorm=1.0,
                ),
                loss=(HorizonBalancedHuber(horizon=int(horizon), auxiliary_horizons=auxiliary_horizons)
                      if objective == "horizon_balanced" else keras.losses.Huber(delta=1.0)),
                metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
            )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(patience),
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(8, int(patience // 3)),
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        print(
            f"\nTraining V2.4 ensemble member {member_index}/{len(seeds)} "
            f"(seed={seed}) on {training_device}"
        )
        with tf.device(training_device):
            history = model.fit(
                X_scaled["train"],
                y_scaled["train"],
                validation_data=(X_scaled["validation"], y_scaled["validation"]),
                epochs=int(epochs),
                batch_size=int(batch_size),
                callbacks=callbacks,
                shuffle=True,
                verbose=verbose,
            )

        if not history.history.get("val_loss") or not all(
            np.isfinite(values).all() for values in history.history.values()
        ):
            raise RuntimeError("Non-finite training history; candidate will not be published.")
        checkpoint = model_dir / f"cnn_lstm_{V24_MODEL_VERSION}_member_{member_index:02d}.keras"
        _save_model_atomic(model, checkpoint)
        from .lite_inference import LiteModel
        probe = X_scaled["validation"][:2]
        np.testing.assert_allclose(LiteModel(checkpoint).predict(probe),
                                   model(probe, training=False).numpy(), rtol=1e-4, atol=1e-5)
        member_history = pd.DataFrame(history.history)
        member_history.insert(0, "epoch", np.arange(1, len(member_history) + 1))
        member_history.insert(0, "seed", seed)
        member_history.insert(0, "member", member_index)
        history_rows.append(member_history)
        models.append(model)

    ensemble = PredictionEnsemble(models)
    reports = {}

    for target_index, currency in enumerate(currencies):
        adapter = V24TargetOutputAdapter(
            ensemble,
            target_index=target_index,
            scaler_y=scalers_y[currency],
            horizon=int(horizon),
            auxiliary_horizons=auxiliary_horizons,
        )
        aligned_target_rates = data["aligned_rates"][:, target_index]

        _, selection_details = evaluate_walk_forward(
            adapter,
            X_scaled["selection"],
            scalers_y[currency],
            partitions["selection"]["origins"],
            aligned_target_rates,
            data["aligned_dates"],
            currency=currency,
            horizons=DISPLAY_HORIZONS,
            spacing=WALK_FORWARD_SPACING,
            spacing_by_horizon={h: h for h in DISPLAY_HORIZONS},
            include_arima=bool(include_arima),
        )
        selection_metrics = _matched_selection_metrics(
            selection_details,
            currency,
            DISPLAY_HORIZONS,
            MIN_SELECTION_ORIGINS,
            bool(include_arima),
        )
        selection_policy = select_benchmark_methods(
            selection_metrics, min_origins=MIN_SELECTION_ORIGINS
        )

        calibration_table = calibrate_method_forecast_intervals(
            adapter,
            X_scaled["calibration"],
            scalers_y[currency],
            partitions["calibration"]["origins"],
            aligned_target_rates,
            horizon=int(horizon),
            include_arima=bool(include_arima),
            origin_stride=int(calibration_stride),
            min_samples=CALIBRATION_MIN_SAMPLES,
        )

        if evaluate_test:
            metrics, details = evaluate_walk_forward(
                adapter,
                X_scaled["test"],
                scalers_y[currency],
                partitions["test"]["origins"],
                aligned_target_rates,
                data["aligned_dates"],
                currency=currency,
                horizons=DISPLAY_HORIZONS,
                spacing=WALK_FORWARD_SPACING,
                include_arima=bool(include_arima),
            )
    
            ensemble_dispersion = _ensemble_dispersion_table(
                models,
                target_index=target_index,
                scaler_y=scalers_y[currency],
                X_scaled=X_scaled["test"],
                currency=currency,
                horizon=int(horizon),
                auxiliary_horizons=auxiliary_horizons,
            )
        else:
            metrics = pd.DataFrame(columns=["model", "mae_skill_vs_random_walk_pct"])
            details = pd.DataFrame(columns=["origin_index"])
            ensemble_dispersion = pd.DataFrame(columns=["currency", "horizon"])

        reports[currency] = {
            "metrics": metrics,
            "ensemble_dispersion": ensemble_dispersion,
            "details": details,
            "selection_metrics": selection_metrics,
            "selection_details": selection_details,
            "selection_policy": selection_policy,
            "calibration": calibration_table,
            "target_index": target_index,
        }

    for folder in [model_dir, scaler_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    ensemble_model_files = []
    for member_index, (seed, model) in enumerate(zip(seeds, models), start=1):
        filename = f"cnn_lstm_{V24_MODEL_VERSION}_member_{member_index:02d}.keras"
        _save_model_atomic(model, model_dir / filename)
        ensemble_model_files.append(filename)

    shared_scaler_path = scaler_dir / f"scaler_X_{V24_MODEL_VERSION}_shared.joblib"
    _atomic_joblib(scaler_X, shared_scaler_path)

    for currency in currencies:
        target_index = reports[currency]["target_index"]
        report = reports[currency]
        paths = {
            "scaler_y": scaler_dir / f"scaler_y_{V24_MODEL_VERSION}_{currency}.joblib",
            "metadata": metadata_dir / f"metadata_{V24_MODEL_VERSION}_{currency}.json",
            "metrics": metadata_dir / f"metrics_{V24_MODEL_VERSION}_{currency}.csv",
            "details": metadata_dir / f"walk_forward_details_{V24_MODEL_VERSION}_{currency}.csv",
            "calibration": metadata_dir / f"calibration_{V24_MODEL_VERSION}_{currency}.csv",
            "selection_metrics": metadata_dir / f"selection_metrics_{V24_MODEL_VERSION}_{currency}.csv",
            "selection_details": metadata_dir / f"selection_details_{V24_MODEL_VERSION}_{currency}.csv",
            "selection_policy": metadata_dir / f"selection_policy_{V24_MODEL_VERSION}_{currency}.csv",
            "ensemble_dispersion": metadata_dir / f"ensemble_dispersion_{V24_MODEL_VERSION}_{currency}.csv",
        }

        _atomic_joblib(scalers_y[currency], paths["scaler_y"])
        _atomic_csv(report["metrics"], paths["metrics"])
        _atomic_csv(report["details"], paths["details"])
        _atomic_csv(report["calibration"], paths["calibration"])
        _atomic_csv(report["selection_metrics"], paths["selection_metrics"])
        _atomic_csv(report["selection_details"], paths["selection_details"])
        _atomic_csv(report["selection_policy"], paths["selection_policy"])
        _atomic_csv(report["ensemble_dispersion"], paths["ensemble_dispersion"])

        cnn_rows = report["metrics"][report["metrics"]["model"] == "CNN-LSTM"]
        beats = int((cnn_rows["mae_skill_vs_random_walk_pct"] > 0).sum())
        best_validation_losses = [
            float(np.min(member_history["val_loss"].to_numpy()))
            for member_history in history_rows
        ]
        latest_common_date = pd.Timestamp(data["aligned_dates"][-1]).date().isoformat()

        metadata = {
            "model_version": V24_MODEL_VERSION,
            "candidate_revision": "v2.4.1",
            "final_test_evaluated": bool(evaluate_test),
            "architecture_preset": architecture,
            "training_objective": objective,
            "model_type": "joint_multicurrency",
            "currency": currency,
            "target_index": int(target_index),
            "target_currencies": currencies,
            "input_currencies": currencies,
            "rate_convention": f"MYR per 1 {currency}",
            "source": "Bank Negara Malaysia via data.gov.my",
            "source_url": "https://data.gov.my/data-catalogue/exchangerates_daily_1200",
            "rate_type": "middle",
            "lookback": int(lookback),
            "forecast_horizon": int(horizon),
            "feature_version": "v2_4",
            "feature_columns": data["feature_columns"],
            "input_feature_columns": data["feature_columns"],
            "n_input_features": int(n_features),
            "n_features_per_currency": int(data["n_features_per_currency"]),
            "auxiliary_horizons": auxiliary_horizons,
            "output_block_size": int(block_size),
            "auxiliary_loss_weight": float(V24_AUXILIARY_LOSS_WEIGHT),
            "auxiliary_blend": float(V24_AUXILIARY_BLEND),
            "ensemble_model_files": ensemble_model_files,
            "ensemble_seeds": seeds,
            "ensemble_members": len(models),
            "ensemble_dispersion_file": paths["ensemble_dispersion"].name,
            "ensemble_dispersion_interpretation": (
                "Standard deviation across independently trained member cumulative-return "
                "predictions at each dashboard horizon. Diagnostic disagreement only; "
                "not a calibrated prediction interval."
            ),
            "output_layout": (
                "currency-major; each currency block contains 63 scaled daily "
                "log-return outputs followed by sqrt(h)-normalised cumulative "
                "return auxiliaries at the configured horizons"
            ),
            "architecture": model.to_json(),
            "training_strategy": (
                "Three independently initialized joint multi-currency CNN-LSTM "
                "members. Predictions are averaged. Each currency has an independent "
                "decoder head and direct auxiliary cumulative-return outputs."
            ),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_device": training_device,
            "accelerator": "GPU" if training_device.startswith("/GPU") else "CPU",
            "gpu_name": gpu_name,
            "latest_observation_date": latest_common_date,
            "latest_rate": float(data["aligned_rates"][-1, target_index]),
            "common_observations": int(len(data["aligned_dates"])),
            "train_samples": int(len(partitions["train"]["X"])),
            "validation_samples": int(len(partitions["validation"]["X"])),
            "selection_samples": int(len(partitions["selection"]["X"])),
            "calibration_samples": int(len(partitions["calibration"]["X"])),
            "test_candidate_samples": int(len(partitions["test"]["X"])),
            "walk_forward_origins": int(report["details"]["origin_index"].nunique()),
            "walk_forward_spacing": int(WALK_FORWARD_SPACING),
            "split_fractions": dict(SPLIT_FRACTIONS),
            "selection_method": (
                "Per-currency, per-horizon selection on a separate chronological "
                "selection partition using common matched forecast origins. Lowest "
                "selection-partition MAE wins; RMSE and absolute-error standard "
                "deviation break ties. Final test data is not used for selection."
            ),
            "selection_min_origins": int(MIN_SELECTION_ORIGINS),
            "selection_policy": report["selection_policy"][
                ["horizon", "selected_model", "n_selection_origins"]
            ].to_dict("records"),
            "benchmarks": [
                "Random walk", "Historical drift (252)", "Recent drift (20)",
                "ARIMA(1,1,1)" if include_arima else "ARIMA disabled",
            ],
            "cnn_lstm_horizons_beating_random_walk": beats,
            "cnn_lstm_horizons_evaluated": int(len(cnn_rows)),
            "member_best_validation_losses": best_validation_losses,
            "epochs_completed_by_member": [
                int(len(frame)) for frame in history_rows
            ],
            "max_epochs_requested": int(epochs),
            "calibration_method": (
                "Method-specific empirical absolute log-error quantiles from a "
                "separate chronological calibration partition. Calibration bands "
                "are descriptive scenario ranges, not formal coverage guarantees."
            ),
            "calibration_origin_stride": int(calibration_stride),
            "calibration_min_samples": int(CALIBRATION_MIN_SAMPLES),
            "arima_included": bool(include_arima),
            "metric_definitions": {
                "mae_skill_vs_random_walk_pct": "100 * (1 - model_MAE / random_walk_MAE)",
                "rmse_skill_vs_random_walk_pct": "100 * (1 - model_RMSE / random_walk_RMSE)",
                "directional_accuracy_pct": "Not applicable to random walk because it predicts no directional change.",
            },
        }
        _atomic_json(metadata, paths["metadata"])

    history_frame = pd.concat(history_rows, ignore_index=True)
    _atomic_csv(
        history_frame,
        metadata_dir / f"training_history_{V24_MODEL_VERSION}.csv",
    )

    run_info.update({"status": "complete", "split_boundaries": data["split_boundaries"],
                     "final_test_evaluated": bool(evaluate_test),
                     "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    run_info["artifact_sha256"] = {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in run_dir.rglob("*") if path.is_file() and path.name != "run.json"}
    _atomic_json(run_info, run_dir / "run.json")
    return {
        "run_dir": str(run_dir),
        "model_path": str(model_dir / ensemble_model_files[0]),
        "model_paths": [str(model_dir / name) for name in ensemble_model_files],
        "currencies": currencies,
        "metadata": {
            "model_version": V24_MODEL_VERSION,
            "candidate_revision": "v2.4.1",
            "final_test_evaluated": bool(evaluate_test),
            "architecture_preset": architecture,
            "training_objective": objective,
            "training_device": training_device,
            "accelerator": "GPU" if training_device.startswith("/GPU") else "CPU",
            "gpu_name": gpu_name,
            "ensemble_members": len(models),
            "ensemble_seeds": seeds,
            "common_observations": int(len(data["aligned_dates"])),
            "train_samples": int(len(partitions["train"]["X"])),
        },
        "reports": reports,
        "history": history_frame,
        "data": data,
        "models": models,
        "model": ensemble,
    }
