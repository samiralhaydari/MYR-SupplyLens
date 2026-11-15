"""Leakage-safe multivariate features and samples for joint currency modelling."""
import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLUMNS, LOOKBACK, FORECAST_HORIZON, SPLIT_FRACTIONS,
    V24_CURRENCY_FEATURE_COLUMNS, V24_GLOBAL_FEATURE_COLUMNS,
    DISPLAY_HORIZONS,
)
from .features import make_feature_frame


def _make_multivariate_feature_frame_v22(rates, currencies):
    """Return one aligned feature frame with nine channels per currency.

    Column order is currency-major and follows ``currencies`` exactly.
    Missing observations are retained as NaN here and removed during alignment.
    """
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("Multivariate inference requires a rates DataFrame.")
    currencies = [str(c).upper() for c in currencies]
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing cross-currency input columns: {missing}")
    if not currencies:
        raise ValueError("At least one currency is required.")

    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    frames = []
    for currency in currencies:
        frame = make_feature_frame(panel[currency])
        frame.columns = [f"{currency}__{name}" for name in FEATURE_COLUMNS]
        frames.append(frame)

    result = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    return result


def _make_multivariate_supervised_samples_v22(
    rates,
    currencies,
    lookback=LOOKBACK,
    horizon=FORECAST_HORIZON,
):
    """Build joint X windows and currency-major multi-output targets.

    X shape: (samples, lookback, n_currencies * n_features)
    y shape: (samples, n_currencies * horizon)

    Targets are laid out as [currency_0's horizon, currency_1's horizon, ...].
    Five chronological partitions mirror the V2.1 protocol. No target window
    crosses a partition boundary.
    """
    currencies = [str(c).upper() for c in currencies]
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("rates must be a pandas DataFrame.")
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing currencies from rate panel: {missing}")

    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    feature_frame = make_multivariate_feature_frame(panel, currencies)
    log_rates = np.log(panel)
    target_frame = log_rates.diff()
    target_frame.columns = [f"{c}__target_return" for c in currencies]

    aligned = pd.concat([feature_frame, target_frame], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    feature_columns = list(feature_frame.columns)
    target_columns = list(target_frame.columns)
    feature_values = aligned[feature_columns].to_numpy(dtype=np.float32)
    target_values = aligned[target_columns].to_numpy(dtype=np.float32)
    aligned_rates = panel.reindex(aligned.index).to_numpy(dtype=np.float64)
    dates = aligned.index
    n = len(aligned)
    n_currencies = len(currencies)

    minimum = int(lookback) + int(horizon) + 300
    if n < minimum:
        raise ValueError(
            f"Only {n} common usable observations remain for {currencies}. "
            f"At least {minimum} are recommended for the five-way chronological split."
        )

    train_end = int(n * SPLIT_FRACTIONS["train"]) - 1
    validation_end = int(
        n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"])
    ) - 1
    selection_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
        )
    ) - 1
    calibration_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
            + SPLIT_FRACTIONS["calibration"]
        )
    ) - 1

    X, y, origins, labels = [], [], [], []
    for origin in range(int(lookback) - 1, n - int(horizon)):
        target_start = origin + 1
        target_end = origin + int(horizon)

        if target_end <= train_end:
            label = "train"
        elif target_start > train_end and target_end <= validation_end:
            label = "validation"
        elif target_start > validation_end and target_end <= selection_end:
            label = "selection"
        elif target_start > selection_end and target_end <= calibration_end:
            label = "calibration"
        elif target_start > calibration_end and target_end < n:
            label = "test"
        else:
            continue

        X.append(feature_values[origin - int(lookback) + 1:origin + 1])
        # (horizon, currencies) -> (currencies, horizon) -> currency-major flat vector
        future = target_values[target_start:target_end + 1, :].T.reshape(-1)
        y.append(future)
        origins.append(origin)
        labels.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    labels = np.asarray(labels)

    result = {}
    for label in ["train", "validation", "selection", "calibration", "test"]:
        mask = labels == label
        result[label] = {
            "X": X[mask],
            "y": y[mask],
            "origins": origins[mask],
        }

    result["aligned_rates"] = aligned_rates
    result["aligned_dates"] = dates
    result["lookback"] = int(lookback)
    result["horizon"] = int(horizon)
    result["currencies"] = currencies
    result["feature_columns"] = feature_columns
    result["target_columns"] = target_columns
    result["n_features_per_currency"] = len(FEATURE_COLUMNS)
    result["split_boundaries"] = {
        "train_end": int(train_end),
        "validation_end": int(validation_end),
        "selection_end": int(selection_end),
        "calibration_end": int(calibration_end),
        "n_aligned_observations": int(n),
        "fractions": dict(SPLIT_FRACTIONS),
    }
    return result


def _make_v24_currency_feature_frame(series):
    """Expanded, strictly backward-looking FX features for V2.4."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if (values.dropna() <= 0).any():
        raise ValueError("Exchange rates must be strictly positive.")

    log_rate = np.log(values)
    returns = log_rate.diff()

    frame = pd.DataFrame(index=values.index)
    frame["log_rate"] = log_rate
    frame["return_1d"] = returns
    frame["return_2d"] = log_rate.diff(2)
    for window in (5, 10, 20, 60, 120):
        frame[f"momentum_{window}"] = log_rate.diff(window)

    for window in (5, 10, 20, 60, 120):
        frame[f"volatility_{window}"] = returns.rolling(
            window, min_periods=window
        ).std()

    eps = 1e-12
    frame["volatility_ratio_5_20"] = (
        frame["volatility_5"] / (frame["volatility_20"] + eps)
    )
    frame["volatility_ratio_20_60"] = (
        frame["volatility_20"] / (frame["volatility_60"] + eps)
    )

    for window in (20, 60):
        rolling_mean = log_rate.rolling(window, min_periods=window).mean()
        rolling_std = returns.rolling(window, min_periods=window).std()
        frame[f"zscore_{window}"] = (
            (log_rate - rolling_mean) / (rolling_std * np.sqrt(window) + eps)
        )
        rolling_high = log_rate.rolling(window, min_periods=window).max()
        rolling_low = log_rate.rolling(window, min_periods=window).min()
        frame[f"range_{window}"] = rolling_high - rolling_low
        frame[f"drawdown_{window}"] = log_rate - rolling_high

    ema_10 = log_rate.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_30 = log_rate.ewm(span=30, adjust=False, min_periods=30).mean()
    ema_20 = log_rate.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_60 = log_rate.ewm(span=60, adjust=False, min_periods=60).mean()
    frame["ema_gap_10_30"] = ema_10 - ema_30
    frame["ema_gap_20_60"] = ema_20 - ema_60

    return frame[V24_CURRENCY_FEATURE_COLUMNS]


def _make_multivariate_feature_frame_v24(rates, currencies):
    """Expanded currency features plus common cross-currency market factors."""
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("Multivariate inference requires a rates DataFrame.")
    currencies = [str(c).upper() for c in currencies]
    if not currencies:
        raise ValueError("At least one currency is required.")
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing cross-currency input columns: {missing}")

    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    frames = []
    for currency in currencies:
        frame = _make_v24_currency_feature_frame(panel[currency])
        frame.columns = [f"{currency}__{name}" for name in V24_CURRENCY_FEATURE_COLUMNS]
        frames.append(frame)

    # Cross-currency factors are calculated only from information available
    # at or before each date. They provide a compact shared-MYR market context.
    log_panel = np.log(panel)
    panel_returns = log_panel.diff()
    median_return = panel_returns.median(axis=1)
    global_frame = pd.DataFrame(index=panel.index)
    global_frame["panel_median_return_1d"] = median_return
    global_frame["panel_median_return_5d"] = median_return.rolling(
        5, min_periods=5
    ).sum()
    global_frame["panel_median_volatility_20"] = median_return.rolling(
        20, min_periods=20
    ).std()
    global_frame["panel_return_dispersion_20"] = panel_returns.std(
        axis=1, ddof=1
    ).rolling(20, min_periods=20).mean()
    global_frame = global_frame[V24_GLOBAL_FEATURE_COLUMNS]

    result = pd.concat(frames + [global_frame], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def make_multivariate_feature_frame(rates, currencies, feature_version="v2_2"):
    """Versioned feature builder; defaults to the unchanged V2.2 feature schema."""
    version = str(feature_version).lower()
    if version in {"v2_4", "2.4", "v24"}:
        return _make_multivariate_feature_frame_v24(rates, currencies)
    if version in {"v2_2", "v2_1", "v2", "v1"}:
        return _make_multivariate_feature_frame_v22(rates, currencies)
    raise ValueError(f"Unsupported multivariate feature version: {feature_version}")


def make_multivariate_supervised_samples(
    rates,
    currencies,
    lookback=LOOKBACK,
    horizon=FORECAST_HORIZON,
    feature_version="v2_2",
):
    """Build the versioned joint training set.

    V2.2 retains its original 63-target-per-currency layout.
    V2.4 appends four direct cumulative-return targets (5/21/42/63 sessions)
    to each currency block. Auxiliary targets are divided by sqrt(horizon)
    to keep their scale comparable with daily returns.
    """
    if str(feature_version).lower() not in {"v2_4", "2.4", "v24"}:
        return _make_multivariate_supervised_samples_v22(
            rates, currencies, lookback=lookback, horizon=horizon
        )

    currencies = [str(c).upper() for c in currencies]
    if int(lookback) < 1 or int(horizon) < 1:
        raise ValueError("lookback and horizon must be positive")
    if len(currencies) < 2 or len(set(currencies)) != len(currencies):
        raise ValueError("At least two unique currencies required")
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("rates must be a pandas DataFrame.")
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"Missing currencies from rate panel: {missing}")

    if not rates.index.is_unique:
        raise ValueError("Rate dates must be unique")
    panel = rates.loc[:, currencies].copy().sort_index()
    for currency in currencies:
        panel[currency] = pd.to_numeric(panel[currency], errors="coerce")
        valid = panel[currency].dropna()
        if (valid <= 0).any():
            raise ValueError(f"{currency} contains non-positive exchange rates.")

    feature_frame = _make_multivariate_feature_frame_v24(panel, currencies)
    log_rates = np.log(panel)
    target_frame = log_rates.diff()
    target_frame.columns = [f"{c}__target_return" for c in currencies]

    aligned = pd.concat([feature_frame, target_frame], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    feature_columns = list(feature_frame.columns)
    target_columns = list(target_frame.columns)
    feature_values = aligned[feature_columns].to_numpy(dtype=np.float32)
    target_values = aligned[target_columns].to_numpy(dtype=np.float32)
    aligned_rates = panel.reindex(aligned.index).to_numpy(dtype=np.float64)
    # Horizons count common retained observations, so recompute after alignment.
    # The first row is never a future target because every origin is >= 0.
    target_values = np.vstack([np.zeros((1, len(currencies))),
                              np.diff(np.log(aligned_rates), axis=0)]).astype(np.float32)
    dates = aligned.index
    n = len(aligned)
    n_currencies = len(currencies)
    auxiliary_horizons = [int(h) for h in DISPLAY_HORIZONS if int(h) <= int(horizon)]
    output_block_size = int(horizon) + len(auxiliary_horizons)

    minimum = int(lookback) + int(horizon) + 300
    if n < minimum:
        raise ValueError(
            f"Only {n} common usable observations remain for {currencies}. "
            f"At least {minimum} are recommended for the five-way chronological split."
        )

    train_end = int(n * SPLIT_FRACTIONS["train"]) - 1
    validation_end = int(
        n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"])
    ) - 1
    selection_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
        )
    ) - 1
    calibration_end = int(
        n * (
            SPLIT_FRACTIONS["train"]
            + SPLIT_FRACTIONS["validation"]
            + SPLIT_FRACTIONS["selection"]
            + SPLIT_FRACTIONS["calibration"]
        )
    ) - 1

    X, y, origins, labels = [], [], [], []
    for origin in range(int(lookback) - 1, n - int(horizon)):
        target_start = origin + 1
        target_end = origin + int(horizon)

        if target_end <= train_end:
            label = "train"
        elif target_start > train_end and target_end <= validation_end:
            label = "validation"
        elif target_start > validation_end and target_end <= selection_end:
            label = "selection"
        elif target_start > selection_end and target_end <= calibration_end:
            label = "calibration"
        elif target_start > calibration_end and target_end < n:
            label = "test"
        else:
            continue

        X.append(feature_values[origin - int(lookback) + 1:origin + 1])
        future = target_values[target_start:target_end + 1, :]
        currency_blocks = []
        for currency_index in range(n_currencies):
            daily_path = future[:, currency_index]
            auxiliary = [
                float(np.sum(daily_path[:h]) / np.sqrt(h))
                for h in auxiliary_horizons
            ]
            currency_blocks.append(
                np.concatenate([
                    daily_path.astype(np.float32),
                    np.asarray(auxiliary, dtype=np.float32),
                ])
            )
        y.append(np.concatenate(currency_blocks))
        origins.append(origin)
        labels.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    origins = np.asarray(origins, dtype=np.int64)
    labels = np.asarray(labels)

    result = {}
    for label in ["train", "validation", "selection", "calibration", "test"]:
        mask = labels == label
        result[label] = {
            "X": X[mask],
            "y": y[mask],
            "origins": origins[mask],
        }

    result["aligned_rates"] = aligned_rates
    result["aligned_dates"] = dates
    result["lookback"] = int(lookback)
    result["horizon"] = int(horizon)
    result["currencies"] = currencies
    result["feature_columns"] = feature_columns
    result["target_columns"] = target_columns
    result["n_features_per_currency"] = len(V24_CURRENCY_FEATURE_COLUMNS)
    result["auxiliary_horizons"] = auxiliary_horizons
    result["output_block_size"] = output_block_size
    result["feature_version"] = "v2_4"
    result["split_boundaries"] = {
        "train_end": int(train_end),
        "validation_end": int(validation_end),
        "selection_end": int(selection_end),
        "calibration_end": int(calibration_end),
        "n_aligned_observations": int(n),
        "fractions": dict(SPLIT_FRACTIONS),
    }
    return result


def make_multivariate_supervised_samples_v24(
    rates, currencies, lookback=180, horizon=FORECAST_HORIZON
):
    return make_multivariate_supervised_samples(
        rates, currencies, lookback=lookback, horizon=horizon,
        feature_version="v2_4",
    )
