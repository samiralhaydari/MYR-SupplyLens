#!/usr/bin/env python3
"""Verify the model/data bundle expected by the hosted MYR SupplyLens app."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.supplylens.config import (
    DATA_RAW_DIR, MODEL_VERSION, PRIORITY_CURRENCIES, MODEL_DIR, SCALER_DIR,
    METADATA_DIR, FORECAST_HORIZON,
)
from src.supplylens.data import load_exchange_rates
from src.supplylens.forecasting import load_model_bundle
from src.supplylens.lite_inference import LiteModel
from src.supplylens.multivariate import make_multivariate_feature_frame


def main() -> int:
    print("MYR SupplyLens deployment verification")
    print(f"Active model version: {MODEL_VERSION}")
    if MODEL_VERSION != "v2_2":
        raise SystemExit("This prepared deployment ZIP is expected to use V2.2 as its active model.")

    snapshot = DATA_RAW_DIR / "daily_1200.csv"
    if not snapshot.is_file():
        raise SystemExit(f"Missing BNM data snapshot: {snapshot}")
    rates = load_exchange_rates(raw_path=snapshot)
    print(f"BNM snapshot: {len(rates):,} daily rows; latest {rates.index.max():%Y-%m-%d}")

    for currency in PRIORITY_CURRENCIES:
        bundle = load_model_bundle(currency)
        model, scaler_x, scaler_y, metadata, calibration, selection = bundle
        if metadata.get("model_version") != "v2_2":
            raise SystemExit(f"{currency}: unexpected model version {metadata.get('model_version')!r}")
        if not isinstance(model, LiteModel):
            raise SystemExit(f"{currency}: inference did not use LiteModel")
        if selection is None or set(selection["horizon"].astype(int)) != {5, 21, 42, 63}:
            raise SystemExit(f"{currency}: missing V2.2 selection-policy horizons")
        if calibration is None or calibration.empty:
            raise SystemExit(f"{currency}: calibration table is missing")

        input_currencies = [str(c).upper() for c in metadata["input_currencies"]]
        feature_columns = metadata.get("input_feature_columns", metadata.get("feature_columns", []))
        features = make_multivariate_feature_frame(
            rates, input_currencies, feature_version=metadata.get("feature_version", "v2_2")
        ).dropna()
        lookback = int(metadata["lookback"])
        window = features.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)
        X = scaler_x.transform(window).reshape(1, lookback, len(feature_columns)).astype(np.float32)
        prediction = model.predict(X, verbose=0)
        expected = len(input_currencies) * int(metadata["forecast_horizon"])
        if prediction.shape != (1, expected) or not np.isfinite(prediction).all():
            raise SystemExit(f"{currency}: invalid model output shape/values: {prediction.shape}")
        if int(metadata["forecast_horizon"]) != FORECAST_HORIZON:
            raise SystemExit(f"{currency}: unexpected forecast horizon")
        if not hasattr(scaler_y, "inverse_transform"):
            raise SystemExit(f"{currency}: target scaler could not be loaded")
        print(f"  OK {currency}: {model.__class__.__name__}, input={X.shape}, output={prediction.shape}")

    print("\nPASS: all seven active V2.2 bundles load with NumPy inference; no TensorFlow required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
