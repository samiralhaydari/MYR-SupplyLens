"""NumPy-only V2.4 decoding helpers shared by training and lightweight inference."""
import numpy as np
from .config import DISPLAY_HORIZONS, V24_AUXILIARY_LOSS_WEIGHT, V24_AUXILIARY_BLEND


def adjust_path_with_auxiliary_horizons(
    daily_returns,
    auxiliary_cumulative_returns,
    horizons,
    blend=V24_AUXILIARY_BLEND,
):
    """Blend direct horizon forecasts into the daily path at horizon anchors.

    The correction is distributed evenly inside each interval. Therefore the
    adjusted cumulative path reaches each blended 5/21/42/63-session anchor
    exactly, while retaining the model's day-to-day path shape.
    """
    path = np.asarray(daily_returns, dtype=np.float64).copy()
    aux = np.asarray(auxiliary_cumulative_returns, dtype=np.float64).reshape(-1)
    horizons = [int(h) for h in horizons]
    if path.ndim != 1:
        raise ValueError("daily_returns must be a one-dimensional path.")
    if len(aux) != len(horizons):
        raise ValueError("Auxiliary forecasts and horizons must have equal lengths.")
    if not np.isfinite(path).all() or not np.isfinite(aux).all():
        raise ValueError("Multi-horizon predictions must be finite.")

    blend = float(np.clip(blend, 0.0, 1.0))
    original_path = path.copy()
    previous_horizon = 0
    previous_desired_cumulative = 0.0
    for horizon, direct_cumulative in zip(horizons, aux):
        if horizon < 1 or horizon > len(path) or horizon <= previous_horizon:
            raise ValueError("Auxiliary horizons must be increasing and within the path.")
        model_cumulative = float(np.sum(original_path[:horizon]))
        desired_cumulative = (
            (1.0 - blend) * model_cumulative + blend * float(direct_cumulative)
        )
        current_segment = float(np.sum(path[previous_horizon:horizon]))
        desired_segment = desired_cumulative - previous_desired_cumulative
        correction = (desired_segment - current_segment) / (
            horizon - previous_horizon
        )
        path[previous_horizon:horizon] += correction
        previous_horizon = horizon
        previous_desired_cumulative = desired_cumulative
    return path


def decode_v24_currency_prediction(
    raw_prediction,
    target_index,
    scaler_y,
    horizon=63,
    auxiliary_horizons=None,
    auxiliary_loss_weight=V24_AUXILIARY_LOSS_WEIGHT,
    auxiliary_blend=V24_AUXILIARY_BLEND,
):
    """Decode one currency's V2.4 output block into adjusted daily log returns."""
    auxiliary_horizons = [
        int(h) for h in (auxiliary_horizons or DISPLAY_HORIZONS)
        if int(h) <= int(horizon)
    ]
    block_size = int(horizon) + len(auxiliary_horizons)
    prediction = np.asarray(raw_prediction, dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)
    start = int(target_index) * block_size
    stop = start + block_size
    if prediction.ndim != 2 or prediction.shape[1] < stop:
        raise ValueError(
            f"V2.4 model returned {prediction.shape}; expected at least {stop} outputs."
        )

    block = prediction[:, start:stop]
    daily_scaled = block[:, :int(horizon)]
    aux_scaled = block[:, int(horizon):]
    if not np.isfinite(block).all():
        raise ValueError("V2.4 model produced non-finite outputs.")

    daily_raw = scaler_y.inverse_transform(
        daily_scaled.reshape(-1, 1)
    ).reshape(-1, int(horizon))
    aux_scaled = aux_scaled / float(auxiliary_loss_weight)
    aux_normalized = scaler_y.inverse_transform(
        aux_scaled.reshape(-1, 1)
    ).reshape(-1, len(auxiliary_horizons))
    aux_cumulative = aux_normalized * np.sqrt(
        np.asarray(auxiliary_horizons, dtype=np.float64)
    )[None, :]

    adjusted = np.vstack([
        adjust_path_with_auxiliary_horizons(
            daily_raw[row], aux_cumulative[row], auxiliary_horizons,
            blend=auxiliary_blend,
        )
        for row in range(len(daily_raw))
    ])
    return adjusted
