# Release status and model activation policy

## Hosted release

- Active model family: **V2.2**.
- Coverage: USD, CNY, IDR, SGD, THB, JPY, KRW.
- Inference runtime: NumPy + h5py; TensorFlow is not installed by the hosted application.
- Data source: BNM / data.gov.my only.

## V2.4.1 candidate

V2.4.1 is retained as a research candidate. The source audit corrected target alignment, improved the residual CNN training objective, added safe experiment directories, strengthened inference parity tests, and withheld final-test evaluation during training. However, model accuracy and superiority were not established by the smoke run or the uploaded reports.

The uploaded V2.4 training-results archive contains final-test reports but not the trained model weights and matching scalers required by the application. Therefore, V2.4.1 is not active.

## Promotion requirements

A future V2.4.1 promotion must be a separate release decision. At minimum:

1. Export all three trained model members, the shared input scaler, all seven target scalers, and matching metadata/calibration/selection artifacts.
2. Verify every source and artifact hash.
3. Confirm Keras/NumPy inference parity for the exact saved model files.
4. Compare V2.4.1 with V2.2 on locked, matched calendar origins and report MAE, RMSE, confidence intervals, calibration, and rejection rates.
5. Keep the final-test access ledger and report its limitations honestly.
6. Build a new deployment ZIP and deliberately update `MODEL_VERSION` only after review.

The candidate exporter does not write to `artifacts/`, change `MODEL_VERSION`, or modify the active application.
