# Build verification — 15 November 2026

## Checks performed on the assembled GitHub/Streamlit repository

- Python compilation: `python -m compileall -q app.py src pages scripts tests` — passed.
- Deployment verifier: `python scripts/verify_deployment.py` — passed for all seven V2.2 currencies.
- NumPy inference: all seven saved V2.2 bundles loaded as `LiteModel` and returned finite `(1, 441)` outputs from `(1, 90, 63)` inputs, without TensorFlow.
- End-to-end forecasting: all seven currencies generated 63-session forecast paths and valid 5/21/42/63-session selected checkpoints from the bundled BNM data snapshot.
- Test suite: `pytest -q` — **28 passed, 20 skipped** in the packaging environment.
- Test skips were expected because this packaging environment did not have Streamlit or TensorFlow installed. The Streamlit-dependent helper test and TensorFlow/Keras parity tests are optional here; the V2.4.1 audit record documents the separate Streamlit AppTest run.
- The environment had scikit-learn 1.8.0 while some preserved StandardScaler artifacts were serialized by 1.6.1. The loader emitted the expected `InconsistentVersionWarning`; the bundle loaded and inference completed. `requirements.txt` pins the audited hosting version (1.9.1).

The packaging sandbox used Python 3.13.5 with preinstalled NumPy/pandas/scikit-learn versions, not the pinned Python 3.12 hosting environment.

## Scope of this verification

This verifies repository structure, Python syntax, the BNM snapshot, model/scaler/metadata completeness, and NumPy inference. It does not claim that a live Streamlit Community Cloud deployment has been created or that browser rendering was tested in this packaging container. Streamlit AppTest coverage from the audited development environment is documented in `AUDIT_V2_4_1.md`.
