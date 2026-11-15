> **Development continuation:** See [README_V2_4_1.md](README_V2_4_1.md). The instructions below document the original V2.4 release; use the isolated V2.4.1 workflow for new development.

# MYR SupplyLens V2.4 — Attention CNN-LSTM Ensemble

**Predict. Plan. Procure. Profit.**

V2.4 makes the joint CNN-LSTM the primary forecasting engine. It replaces the shallow V2.2 sequence stack with a multi-scale dilated CNN, four-head temporal LSTM attention, currency-specific decoder heads, direct multi-horizon auxiliary targets and a three-member seed ensemble.

Dukascopy is not used. V2.4 trains from the existing Bank Negara Malaysia daily middle-rate panel and the existing project data pipeline.

## V2.4 model design

- Seven currencies are learned jointly: `USD, CNY, IDR, SGD, THB, JPY, KRW`.
- The default historical lookback is 180 aligned observations.
- Each currency contributes 23 backward-looking features.
- Four shared cross-currency factors capture the common MYR market movement and cross-sectional dispersion.
- The supplied release yields 4,966 common aligned observations; the 180-session V2.4 split contains 2,737 training windows. The input contains 165 channels: 7 × 23 currency features + 4 shared market factors.
- Three causal Conv1D layers use kernel sizes 3, 5 and 9 with dilation rates 1, 2 and 3.
- An LSTM returns the full temporal sequence. A four-head temporal-attention pooling layer learns which hidden states matter most.
- Each currency has its own decoder head.
- Each currency head predicts 63 daily log returns plus direct cumulative-return auxiliaries at 5, 21, 42 and 63 sessions.
- Three independently initialized members are trained. Their raw outputs are averaged before the multi-horizon decoder constructs the final path.

## Why direct multi-horizon outputs?

A daily-return path can accumulate error over 5, 21, 42 and 63 sessions. V2.4 therefore trains both:
1. a 63-step daily log-return path; and
2. direct cumulative log-return targets at the four dashboard horizons.

The auxiliary targets are normalized by the square root of their horizon, scaled using the training-only currency scaler, and given a higher training loss weight. During inference, the direct horizon estimates are blended with the cumulative daily path. Small interval corrections preserve the day-to-day shape while aligning the cumulative path to each blended horizon anchor.

## Training protocol

- Chronological partitions remain: 60% train, 10% early-stop validation, 10% method selection, 5% calibration and 15% untouched final test.
- Feature and target scalers are fitted only on the training partition.
- The three ensemble members use seeds `42, 73, 2026`.
- AdamW uses learning rate `0.0005`, weight decay `1e-5` and gradient clipping.
- Huber loss, early stopping, learning-rate reduction and NaN termination are enabled.
- Training windows are shuffled within the training partition; the chronological validation, selection, calibration and final-test boundaries are not shuffled across partitions.
- Model selection remains currency-specific and horizon-specific, using only the selection partition and common matched origins.
- Final test origins retain the existing non-overlapping 63-session spacing.
- Random walk, historical drift, recent drift and ARIMA remain independent benchmarks. They do not replace or train the CNN-LSTM.

## Train in Google Colab

From the project root, run:

```bash
python scripts/train_model.py \
  --epochs 300 \
  --batch-size 64 \
  --patience 40 \
  --lookback 180 \
  --require-gpu
```

To run on CPU, omit `--require-gpu` and add `--device CPU`.

Optional parameters:
- `--seeds 42 73 2026`
- `--currencies USD CNY IDR SGD THB JPY KRW`
- `--skip-arima`
- `--force-download`

The script trains all seven priority currencies jointly by default. It does not call Dukascopy.

## V2.4 artifacts

V2.4 writes only its own versioned artifacts:
- `artifacts/models/cnn_lstm_v2_4_member_01.keras`
- `artifacts/models/cnn_lstm_v2_4_member_02.keras`
- `artifacts/models/cnn_lstm_v2_4_member_03.keras`
- `artifacts/scalers/scaler_X_v2_4_shared.joblib`
- `artifacts/scalers/scaler_y_v2_4_<CURRENCY>.joblib`
- `artifacts/metadata/metadata_v2_4_<CURRENCY>.json`
- `artifacts/metadata/metrics_v2_4_<CURRENCY>.csv`
- `artifacts/metadata/walk_forward_details_v2_4_<CURRENCY>.csv`
- `artifacts/metadata/calibration_v2_4_<CURRENCY>.csv`
- `artifacts/metadata/selection_policy_v2_4_<CURRENCY>.csv`
- `artifacts/metadata/training_history_v2_4.csv`

The existing V1, V2, V2.1 and V2.2 model files are not overwritten. The app prefers a complete V2.4 bundle and falls back to V2.2 and earlier bundles if V2.4 has not yet been trained.

## Deployment

The app's NumPy inference engine supports the V2.4 Conv1D dilation, sequence-returning LSTM, temporal attention pooling and currency-specific heads. TensorFlow remains a training dependency, not a hosting dependency. `tests/test_lite_inference.py` checks NumPy predictions against Keras for available saved models.

Deploy source and artifacts together after V2.4 training. Do not deploy the V2.4 source while omitting its three ensemble members or currency scalers.

## Evaluation rules

The goal is to improve CNN-LSTM performance, not to force a positive score. Report:
- MAE and RMSE against random walk;
- directional accuracy;
- paired bootstrap confidence intervals;
- performance by currency and horizon;
- ensemble member dispersion and validation loss;
- interval calibration and coverage diagnostics.

A model is not declared superior based on training loss or validation loss alone. The untouched final test set is used once for the final comparison. If V2.4 does not outperform V2.2 on the final test, report that honestly and use the diagnostics to guide the next experiment.

## Data and limitations

Bank Negara Malaysia daily middle reference rates via data.gov.my, quoted as MYR per one unit of foreign currency. These are reference rates, not necessarily bank transaction rates. Historical cross-currency co-movement is predictive input, not evidence of causation. The model does not observe future policy decisions, market shocks or transaction spreads. Forecasts and empirical error ranges are planning estimates, not guarantees or financial advice.
