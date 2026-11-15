# MYR SupplyLens V2.4 changelog

## Strategic pivot
- Fully remove Dukascopy from the V2.4 data/training path.
- Use the existing BNM historical rate panel.
- Make CNN-LSTM the primary model-development objective.

## Architecture
- Expand each currency from 9 to 23 leakage-safe historical features.
- Add four shared cross-currency market factors.
- Increase lookback from 90 to 180 aligned observations.
- Replace the shallow V2.2 stack with three causal dilated Conv1D layers.
- Add sequence-returning LSTM and four-head temporal-attention pooling.
- Add independent currency-specific output heads.
- Add direct cumulative-return auxiliary heads at 5, 21, 42 and 63 sessions.
- Train three independently seeded models and average their predictions.

## Training and evaluation
- AdamW with weight decay and gradient clipping.
- Huber loss, early stopping, learning-rate reduction and NaN termination.
- Keep chronological train/validation/selection/calibration/final-test partitions.
- Keep model selection and final testing separate.
- Evaluate the ensemble, not an arbitrarily selected member.
- Preserve Random Walk, drift and ARIMA as independent benchmarks.

## Compatibility and safety
- V1, V2, V2.1 and V2.2 artifacts remain untouched.
- V2.4 has a separate artifact namespace.
- App inference supports V2.4 through the NumPy LiteModel path.
- The app falls back to V2.2 if V2.4 artifacts are not yet present.
