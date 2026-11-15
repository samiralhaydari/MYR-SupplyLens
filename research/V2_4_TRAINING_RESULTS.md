# V2.4 training-results record

**Source:** `MYR_SupplyLens_V24_Training_Results.zip`, uploaded on 15 November 2026.  
**Disposition:** Research evidence only; not an active deployment bundle.

## Reported run outcome

- Final-test reports were completed for all seven currencies: USD, CNY, IDR, SGD, THB, JPY, and KRW.
- The three-member CNN-LSTM ensemble trained for 41 epochs; the best validation epoch was reported as epoch 1.
- Across the 28 currency/horizon MAE comparisons, CNN-LSTM had lower MAE than random walk in 15.
- Across the 28 currency/horizon RMSE comparisons, CNN-LSTM had lower RMSE than random walk in 17.
- The reported MAE confidence intervals did not establish statistically significant improvement.
- No forecasts were rejected in the reported final-test results.

## Deployment decision

The uploaded results archive contains reports, not the trained `.keras` model members and matching scaler bundle needed for inference. The V2.4.1 source archive's two-member integration run is also a smoke test, not the finished three-member model release.

Accordingly:

1. The live Streamlit application uses the preserved V2.2 model files, scalers, metadata, calibration tables, and selection policies.
2. V2.4.1 results remain separate from the active V2.2 report-card artifacts.
3. Do not copy V2.4 smoke models into `artifacts/` or label V2.4.1 as the deployed model.
4. A future candidate export requires a completed three-member V2.4.1 run directory, verified hashes, matching scalers/metadata, and completed final-test reports. Use `scripts/export_v24_candidate.py`; it creates a separate ZIP and never activates the candidate.

These results are promising enough to retain for research, but they do not justify claiming that V2.4.1 is a statistically superior production model.
