# MYR SupplyLens — Streamlit Community Cloud deployment

## Before uploading

- Keep `app.py` and `requirements.txt` at the repository root.
- Keep the complete `artifacts/` directory. Model files, scalers, metadata, calibration files, and selection policies must travel together.
- Keep `data/raw/daily_1200.csv`; it is the public BNM first-run fallback snapshot.
- Do not upload `data/supplylens.sqlite`, `.env`, `.streamlit/secrets.toml`, local logs, or virtual environments.
- Do not move V2.4 smoke models from `experiments/` into `artifacts/`. They are not production model weights.

## Deploy

1. Create a GitHub repository.
2. Upload the contents of this folder to the repository root.
3. Commit to `main`.
4. Go to https://share.streamlit.io/ and choose **Create app**.
5. Select the repository and `main` branch.
6. Set **Main file path** to `app.py`.
7. Open **Advanced settings** and choose **Python 3.12**.
8. Deploy.

The hosted app installs `requirements.txt`, which intentionally excludes TensorFlow. The saved models are evaluated with `src/supplylens/lite_inference.py` using NumPy and h5py.

## Verify after deployment

- Overview displays the seven priority currencies.
- The Forecast Centre loads a model for USD, CNY, IDR, SGD, THB, JPY, and KRW.
- The Model Report Card displays V2.2 metrics and saved per-horizon selection policies.
- Procurement, supplier, and margin simulators open without errors.
- If the BNM endpoint is temporarily unavailable, the bundled CSV snapshot remains usable.

## Common deployment problems

### `ModuleNotFoundError`

Confirm that `requirements.txt` is in the repository root and that the app was deployed from the repository root with `app.py` as the entrypoint. Wait for dependency installation to finish, then reboot the app.

### Model bundle missing

Check that the complete `artifacts/models/`, `artifacts/scalers/`, and `artifacts/metadata/` directories were uploaded. A `.keras` model without its matching scaler and metadata is not a complete bundle.

### NumPy inference cannot load a model

This deployment intentionally has no TensorFlow fallback. Restore the matching original model file or use the separate training environment to export a compatible bundle. Do not solve this by silently replacing active model artifacts.

### BNM data unavailable

The app uses `data/raw/daily_1200.csv` as a fallback and refreshes from the official data.gov.my URL when possible. Do not substitute a third-party FX feed; this project is BNM-only.

### Saved suppliers or invoices disappear

The demo uses local SQLite storage. Treat Community Cloud's local filesystem as ephemeral. This release does not claim durable database persistence.

## Python version

Choose Python 3.12 in Streamlit Community Cloud's Advanced settings. Streamlit Community Cloud currently defaults to Python 3.12; the version is selected when creating the app.
