import json
import os

import joblib
import numpy as np
import pandas as pd

from .config import (
    LOOKBACK, FORECAST_HORIZON, FEATURE_COLUMNS, MODEL_VERSION,
    LEGACY_MODEL_VERSIONS, MODEL_DIR, SCALER_DIR, METADATA_DIR, FORECAST_DIR,
    DISPLAY_HORIZONS,
)
from .features import make_feature_frame
from .multivariate import make_multivariate_feature_frame
from .data import get_currency_series
from .evaluation import benchmark_forecast_paths
from .ensemble import PredictionEnsemble
from .v24_utils import decode_v24_currency_prediction


def _bundle_paths(currency, version):
    if version == "v2_4":
        model_path = MODEL_DIR / "cnn_lstm_v2_4_member_01.keras"
        scaler_x_path = SCALER_DIR / "scaler_X_v2_4_shared.joblib"
    else:
        model_path = (
            MODEL_DIR / f"cnn_lstm_{version}_{currency}.keras"
            if version != "v1" else MODEL_DIR / f"cnn_lstm_{currency}.keras"
        )
        scaler_x_path = (
            SCALER_DIR / f"scaler_X_{version}_{currency}.joblib"
            if version != "v1" else SCALER_DIR / f"scaler_X_{currency}.joblib"
        )
    return {
        "version": version,
        "model": model_path,
        "scaler_X": scaler_x_path,
        "scaler_y": SCALER_DIR / f"scaler_y_{version}_{currency}.joblib" if version != "v1" else SCALER_DIR / f"scaler_y_{currency}.joblib",
        "metadata": METADATA_DIR / f"metadata_{version}_{currency}.json" if version != "v1" else METADATA_DIR / f"metadata_{currency}.json",
        "calibration": METADATA_DIR / f"calibration_{version}_{currency}.csv" if version != "v1" else None,
        "selection": METADATA_DIR / f"selection_policy_{version}_{currency}.csv" if version in {MODEL_VERSION, "v2_2", "v2_1"} else None,
    }


def _load_network(path):
    """Load a saved network with the deployment-safe NumPy inference engine.

    TensorFlow is deliberately not installed in the Streamlit runtime. A Keras
    fallback is available only when a developer explicitly sets
    ``SUPPLYLENS_USE_KERAS=1`` in a training/development environment.
    """
    if os.environ.get("SUPPLYLENS_USE_KERAS", "") == "1":
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise RuntimeError(
                "SUPPLYLENS_USE_KERAS=1 was set, but TensorFlow is not installed. "
                "Use the training environment or unset this variable."
            ) from exc
        # Register V2.4 custom layers for explicitly requested Keras loading.
        from . import model_v24  # noqa: F401
        return keras.models.load_model(path, compile=False)

    from .lite_inference import LiteModel
    try:
        return LiteModel(path)
    except Exception as exc:
        raise RuntimeError(
            f"The model bundle '{path.name}' is not loadable by the NumPy inference engine. "
            "This Streamlit release intentionally does not install TensorFlow. "
            "Verify the model file and export a LiteModel-compatible bundle before deployment."
        ) from exc


def load_model_bundle(currency):
    currency = currency.upper()
    versions = [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]
    for version in versions:
        item = _bundle_paths(currency, version)
        required = [item["model"], item["scaler_X"], item["scaler_y"], item["metadata"]]
        if not all(path.exists() for path in required):
            continue

        metadata = json.loads(item["metadata"].read_text(encoding="utf-8"))
        ensemble_files = metadata.get("ensemble_model_files", [])
        if ensemble_files:
            ensemble_paths = [MODEL_DIR / filename for filename in ensemble_files]
            if not all(path.exists() for path in ensemble_paths):
                continue
            model = PredictionEnsemble([
                _load_network(path) for path in ensemble_paths
            ])
        else:
            model = _load_network(item["model"])
        scaler_X = joblib.load(item["scaler_X"])
        scaler_y = joblib.load(item["scaler_y"])
        calibration = None
        if item["calibration"] is not None and item["calibration"].exists():
            calibration = pd.read_csv(item["calibration"])
        selection = None
        if item["selection"] is not None and item["selection"].exists():
            selection = pd.read_csv(item["selection"])
        return model, scaler_X, scaler_y, metadata, calibration, selection

    raise FileNotFoundError(
        f"No complete trained model bundle was found for {currency}. "
        "Check that the model, scaler and metadata files are included in the repository."
    )


def _calibration_quantile(calibration, model_name, horizon, level):
    if calibration is None or calibration.empty:
        return np.nan
    table = calibration
    if "model" in table.columns:
        table = table[table["model"] == model_name]
    else:
        # Legacy V2 calibration tables contain CNN-LSTM only.
        if model_name != "CNN-LSTM":
            return np.nan
    table = table[table["horizon"] == int(horizon)]
    if table.empty:
        return np.nan
    column = f"q{level}_abs_log_error"
    value = table.iloc[0].get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def bundle_signature(currency):
    """Identify the bundle that would be loaded, without loading it.

    Returns (version, ((filename, mtime_ns), ...)). The UI uses this as a cache
    key, so a retrained model is picked up without restarting the app. Raises
    FileNotFoundError when no complete bundle exists.
    """
    currency = currency.upper()
    versions = [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]
    for version in versions:
        item = _bundle_paths(currency, version)
        required = [item["model"], item["scaler_X"], item["scaler_y"], item["metadata"]]
        if not all(path.exists() for path in required):
            continue
        files = required + [
            path for path in (item["calibration"], item["selection"])
            if path is not None and path.exists()
        ]
        try:
            metadata = json.loads(item["metadata"].read_text(encoding="utf-8"))
            ensemble_paths = [
                MODEL_DIR / filename
                for filename in metadata.get("ensemble_model_files", [])
            ]
            if ensemble_paths and not all(path.exists() for path in ensemble_paths):
                continue
            files.extend(path for path in ensemble_paths if path.exists())
        except Exception:
            pass
        return version, tuple((path.name, path.stat().st_mtime_ns) for path in files)
    raise FileNotFoundError(
        f"No complete trained model bundle was found for {currency}. "
        "Check that the model, scaler and metadata files are included in the repository."
    )


def forecast_currency(rate_data, currency, bundle=None):
    """Forecast one currency from a legacy, V2.2 or V2.4 model bundle.

    Joint models require the full aligned currency panel because each target
    depends on historical features from the complete training currency universe.
    """
    currency = currency.upper()
    if bundle is None:
        bundle = load_model_bundle(currency)
    model, scaler_X, scaler_y, metadata, calibration, selection = bundle

    model_type = metadata.get("model_type", "univariate")
    lookback = int(metadata.get("lookback", LOOKBACK))
    horizon = int(metadata.get("forecast_horizon", FORECAST_HORIZON))

    if model_type == "joint_multicurrency":
        if not isinstance(rate_data, pd.DataFrame):
            raise TypeError(
                "This joint model requires the complete rates DataFrame, not a single currency Series."
            )
        input_currencies = [str(c).upper() for c in metadata.get("input_currencies", [])]
        if currency not in input_currencies:
            raise ValueError(f"{currency} is not in this model's input currency universe.")
        feature_columns = metadata.get("input_feature_columns", metadata.get("feature_columns", []))
        features = make_multivariate_feature_frame(
            rate_data,
            input_currencies,
            feature_version=metadata.get("feature_version", "v2_2"),
        ).dropna()
        missing_features = [c for c in feature_columns if c not in features.columns]
        if missing_features:
            raise ValueError(f"Missing multivariate model features: {missing_features}")
        if len(features) < lookback:
            raise ValueError(f"Need at least {lookback} usable multivariate feature observations.")
        window = features.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)
        target_index = int(metadata["target_index"])
        start = target_index * horizon
        stop = start + horizon
    else:
        if isinstance(rate_data, pd.DataFrame):
            if currency not in rate_data.columns:
                raise ValueError(f"{currency} is missing from the rates DataFrame.")
            rate_series = get_currency_series(rate_data, currency)
        else:
            rate_series = rate_data
        features = make_feature_frame(rate_series).dropna()
        feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
        if len(features) < lookback:
            raise ValueError(f"Need at least {lookback} usable feature observations.")
        missing_features = [c for c in feature_columns if c not in features.columns]
        if missing_features:
            raise ValueError(f"Missing model features: {missing_features}")
        window = features.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)
        start, stop = 0, horizon

    X = scaler_X.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, lookback, window.shape[-1]
    ).astype(np.float32)

    raw_prediction = np.asarray(model.predict(X, verbose=0), dtype=np.float64)
    if raw_prediction.ndim == 1:
        raw_prediction = raw_prediction.reshape(1, -1)

    if metadata.get("model_version") == "v2_4":
        pred_returns = decode_v24_currency_prediction(
            raw_prediction,
            target_index=int(metadata["target_index"]),
            scaler_y=scaler_y,
            horizon=horizon,
            auxiliary_horizons=metadata.get("auxiliary_horizons", DISPLAY_HORIZONS),
            auxiliary_loss_weight=float(metadata.get("auxiliary_loss_weight", 2.5)),
            auxiliary_blend=float(metadata.get("auxiliary_blend", 0.35)),
        ).reshape(-1)
    else:
        if raw_prediction.ndim != 2 or raw_prediction.shape[1] < stop:
            raise ValueError(
                f"Model returned shape {raw_prediction.shape}; expected at least {stop} output values."
            )
        pred_scaled = raw_prediction[:, start:stop]
        if not np.isfinite(pred_scaled).all():
            raise ValueError("CNN-LSTM produced non-finite scaled predictions.")
        pred_returns = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).reshape(-1)

    if len(pred_returns) != horizon or not np.isfinite(pred_returns).all():
        raise ValueError("CNN-LSTM produced invalid future log returns.")

    if isinstance(rate_data, pd.DataFrame):
        latest = get_currency_series(rate_data, currency)
    else:
        latest = rate_data.dropna()
    latest = latest[latest > 0].sort_index()
    if latest.empty:
        raise ValueError(f"No valid latest exchange rate is available for {currency}.")

    latest_rate = float(latest.iloc[-1])
    latest_date = pd.Timestamp(latest.index[-1])
    log_history = np.log(latest.to_numpy(dtype=np.float64)[-(756 + 1):])

    # Construct a candidate without clipping. The benchmark guard decides whether
    # it is plausible enough to publish. A rejected CNN path is never reinserted.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        candidate_cnn = latest_rate * np.exp(np.cumsum(pred_returns))
    if not np.isfinite(candidate_cnn).all() or np.any(candidate_cnn <= 0):
        candidate_cnn = None

    paths = benchmark_forecast_paths(
        log_history,
        latest_rate,
        horizon,
        cnn_path=candidate_cnn,
        include_arima=bool(metadata.get("arima_included", True)),
    )
    cnn_valid = "CNN-LSTM" in paths
    cnn_rates = (
        np.asarray(paths["CNN-LSTM"], dtype=np.float64)
        if cnn_valid else np.full(horizon, np.nan, dtype=np.float64)
    )

    forecast_dates = pd.bdate_range(
        start=latest_date + pd.Timedelta(days=1), periods=horizon
    )
    forecast = pd.DataFrame({
        "date": forecast_dates,
        "forecast_rate": cnn_rates,
        "predicted_log_return": pred_returns,
        "cumulative_log_return": np.cumsum(pred_returns),
        "currency": currency,
    })

    path_columns = {
        "CNN-LSTM": "forecast_cnn_lstm",
        "Random walk": "forecast_random_walk",
        "Historical drift (252)": "forecast_historical_drift_252",
        "Recent drift (20)": "forecast_recent_drift_20",
        "ARIMA(1,1,1)": "forecast_arima_111",
    }
    for method, column in path_columns.items():
        path = paths.get(method)
        forecast[column] = (
            np.asarray(path, dtype=np.float64)
            if path is not None else np.full(horizon, np.nan, dtype=np.float64)
        )

    def _interval(center, q):
        if not np.isfinite(center) or center <= 0 or not np.isfinite(q) or q < 0:
            return np.nan, np.nan
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            low = float(center * np.exp(-q))
            high = float(center * np.exp(q))
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= 0:
            return np.nan, np.nan
        return low, high

    # CNN intervals remain visible only when the CNN path passes the safety guard.
    for level in ("80", "95"):
        q_values = [
            _calibration_quantile(calibration, "CNN-LSTM", step, level)
            for step in range(1, horizon + 1)
        ]
        pairs = [_interval(float(cnn_rates[i]), q_values[i]) for i in range(horizon)]
        forecast[f"lower_{level}"] = [p[0] for p in pairs]
        forecast[f"upper_{level}"] = [p[1] for p in pairs]

    forecast["selected_model"] = None
    forecast["selected_forecast_rate"] = np.nan
    forecast["selection_reason"] = None
    for level in ("80", "95"):
        forecast[f"selected_lower_{level}"] = np.nan
        forecast[f"selected_upper_{level}"] = np.nan

    policy_lookup = {}
    if selection is not None and not selection.empty:
        policy_lookup = {
            int(row["horizon"]): row.to_dict()
            for _, row in selection.iterrows()
        }

    for h in DISPLAY_HORIZONS:
        if h > horizon:
            continue
        policy = policy_lookup.get(int(h), {})
        selected_method = str(policy.get("selected_model", "CNN-LSTM"))
        reason = str(policy.get("selection_reason", "No selection policy: CNN-LSTM default."))

        if selected_method not in paths:
            selected_method = "Random walk"
            reason = (
                "The selected path was unavailable or rejected by numerical safety checks; "
                "Random walk used as the conservative fallback."
            )

        selected_rate = float(paths[selected_method][h - 1])
        if not np.isfinite(selected_rate) or selected_rate <= 0:
            raise ValueError(
                f"The selected {selected_method} forecast is invalid at horizon {h}."
            )

        forecast.loc[h - 1, "selected_model"] = selected_method
        forecast.loc[h - 1, "selected_forecast_rate"] = selected_rate
        forecast.loc[h - 1, "selection_reason"] = reason

        for level in ("80", "95"):
            q = _calibration_quantile(calibration, selected_method, h, level)
            low, high = _interval(selected_rate, q)
            forecast.loc[h - 1, f"selected_lower_{level}"] = low
            forecast.loc[h - 1, f"selected_upper_{level}"] = high

    metadata = dict(metadata)
    metadata["selection_mode"] = (
        "selection-partition per-currency/per-horizon method selection"
        if policy_lookup else "No selection policy artifact; conservative inference fallback"
    )

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    version = str(metadata.get("model_version", "legacy"))
    forecast.to_csv(FORECAST_DIR / f"{currency}_forecast_{version}.csv", index=False)
    forecast.to_csv(FORECAST_DIR / f"{currency}_forecast.csv", index=False)
    return forecast, metadata

