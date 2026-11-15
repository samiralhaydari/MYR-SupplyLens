# MYR SupplyLens

**Predict. Plan. Procure. Profit.**

MYR SupplyLens is a Streamlit decision-support application for Malaysian importers and manufacturing businesses. It combines Bank Negara Malaysia (BNM) daily middle exchange rates, benchmark-aware currency forecasts, procurement-cost simulations, supplier comparisons, and product-margin scenarios.

> **Deployment release:** V2.2 model bundles are active in this repository. V2.4.1 remains a separate research candidate and is not promoted into the live application. The V2.4 results archive contained evaluation reports, not the model-weight/scaler bundle required for inference. Do not replace the active V2.2 files with smoke-test outputs or report-only results.

## Deploy on Streamlit Community Cloud

This repository is already structured for Streamlit Community Cloud. No Colab notebook, model training, TensorFlow installation, or GPU is required to run the hosted application.

1. Download and extract this repository ZIP.
2. Create a GitHub repository, for example `myr-supplylens`.
3. Upload **the contents** of the extracted `MYR_SupplyLens_Streamlit_GitHub` folder to the root of the GitHub repository. `app.py` and `requirements.txt` must be at the repository root.
4. Commit the files to the `main` branch.
5. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
6. Choose your GitHub repository, branch `main`, and main file path `app.py`.
7. In **Advanced settings**, select **Python 3.12**, then deploy.

Streamlit installs the pinned packages from `requirements.txt`. The app loads its saved neural networks through the NumPy inference engine; TensorFlow is deliberately absent from the hosted requirements.

## What is included

- Six Streamlit screens: Overview, FX Outlook, Invoice Cost, Supplier Comparison, Product Margin, and Model Report Card.
- The preserved V2.2 seven-currency joint CNN-LSTM bundles for USD, CNY, IDR, SGD, THB, JPY, and KRW.
- Historical V1, V2, and V2.1 model bundles retained as fallback artifacts.
- Matching scalers, metadata, benchmark metrics, selection policies, and calibration files.
- A public BNM daily-rate CSV snapshot for first launch. The app refreshes it from BNM and falls back to the bundled copy if refresh fails.
- V2.4.1 source, audit notes, training/evaluation scripts, and a separate candidate-export utility. Candidate exports do not activate themselves.

## Current model policy

| Item | Deployment status |
|---|---|
| Active application model | V2.2 |
| Active currency coverage | USD, CNY, IDR, SGD, THB, JPY, KRW |
| V2.4.1 | Research candidate; not promoted |
| Forecasting alternatives | Random walk, historical drift, recent drift, ARIMA |
| Model inference | NumPy + h5py; no TensorFlow/GPU |
| FX data source | BNM / data.gov.my only |
| Dukascopy | Not used |

The V2.4.1 research results are recorded separately in [`research/V2_4_TRAINING_RESULTS.md`](research/V2_4_TRAINING_RESULTS.md). They are not merged into the active V2.2 report-card files.

## Run locally

Use Python 3.12:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Verify the deployment bundle

From the repository root:

```bash
python scripts/verify_deployment.py
```

This checks the BNM data snapshot, all seven active V2.2 model bundles, their matching scalers/metadata, and NumPy model loading without TensorFlow.

## Research and training are separate from hosting

The V2.4.1 training code is retained for reproducibility, but the hosted application does not train. The research environment is recorded in `requirements-training-py312.txt`; it includes TensorFlow and is **not** the Streamlit Cloud requirements file.

A completed V2.4.1 run can be packaged separately:

```bash
python scripts/export_v24_candidate.py experiments/<completed_run_directory>
```

The exporter requires a complete three-member model bundle, matching scalers and metadata, verified hashes, and a completed final-test report directory. It creates a separate candidate ZIP and never copies files into `artifacts/` or changes the active model version. Promotion must be a deliberate, separately reviewed release decision.

## Data, persistence, and limitations

- Exchange rates are BNM daily middle reference rates, quoted as MYR per one unit of foreign currency. They are not bank transaction rates and do not include spreads or fees.
- Forecasts and empirical error ranges are planning estimates, not guarantees or financial advice.
- The supplier directory and saved invoice book use a local SQLite database. Streamlit Community Cloud's local filesystem should be treated as ephemeral; records may be lost when the app container is rebuilt or restarted. This release is a showcase/demo, not a persistent multi-user SaaS system.
- The repository includes the public historical BNM snapshot, but the app attempts to refresh it from the official BNM data.gov.my endpoint.
- No credentials, private keys, or Streamlit secrets are required for the default demo.

## Project documentation

- [`DEPLOY.md`](DEPLOY.md) — deployment checklist and troubleshooting.
- [`docs/AUDIT_V2_4_1.md`](docs/AUDIT_V2_4_1.md) — V2.4.1 audit and validation record.
- [`docs/V2_4_1_DEVELOPMENT_NOTES.md`](docs/V2_4_1_DEVELOPMENT_NOTES.md) — V2.4.1 architecture and research controls.
- [`research/V2_4_TRAINING_RESULTS.md`](research/V2_4_TRAINING_RESULTS.md) — summary of the uploaded V2.4 training results.

---

**Copyright:** No open-source licence is included in this repository. Add a licence only if you intentionally want to grant reuse rights.
