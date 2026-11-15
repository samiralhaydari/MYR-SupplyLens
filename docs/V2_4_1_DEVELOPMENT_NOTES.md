# MYR SupplyLens V2.4.1 — audited development candidate

This is a separate continuation of V2.4, not a promoted production model. V2.2 and all 124 supplied data/artifact files remain byte-identical. The deployed app, Google Drive project and live database were not accessed or changed.

## Start here

Use `MYR_SupplyLens_V2_4_1_Isolated_Colab.ipynb`. It creates a unique sibling copy of your existing Drive project, verifies protected-file hashes, applies this candidate's source only to the copy, and runs tests before training. Upload `MYR_SupplyLens_V2_4_1_Audited_Development.zip` to My Drive first. The complete archive also runs locally after extraction.

For the locally validated Python 3.12 environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-validated-py312.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_project.py --output preflight_audit.json
```

The lock is a Linux/Python 3.12 validation snapshot; it is not a GPU-driver lock. Colab verifies its supplied runtime without downgrading packages. A GPU must be visible for `--require-gpu`.

## Changes that matter

- **Corrected V2.4 target alignment:** daily future log returns are recomputed after common-date alignment. Their cumulative sums now reconstruct the rates used in evaluation, including gaps.
- **Stronger CNN-LSTM candidate:** a 96-channel causal stem, three normalized residual dilated blocks, sequence LSTM, four learned temporal attention heads, currency-specific decoders and a seed ensemble. The CNN receptive field grows from 35 to 99 retained input observations; the LSTM still consumes the full 180-step window. This is architectural capacity, not a measured accuracy claim.
- **Horizon-balanced training:** explicit daily, direct-horizon and blended cumulative-path Huber losses. Existing encoded auxiliary targets are decoded before weighting. Daily loss has weight 1, mean auxiliary loss 2.5, blended cumulative-path loss 1. All currencies and auxiliary horizons receive equal within-group weight. Original target encoding remains compatible with existing decoding.
- **Safe experiments:** fresh output directories, frozen dataset copies, source snapshots/hashes, finite-history checks, per-member checkpoints and Keras/NumPy parity before a run completes. The final test is withheld by default.
- **Fallback/report correctness:** V2.2 selection policies load again; the report card reflects saved policies across versions; legacy descriptions no longer falsely claim that V2.2 has no selection step. Historical rate revisions invalidate forecast caches.

Original architecture and original Huber objective remain available as explicit ablation controls. They use the corrected V2.4 targets; all V2.2 code/artifacts are preserved separately in the original release.

## Train without replacing the working app

```sh
python scripts/train_model.py --epochs 300 --batch-size 64 --patience 40 --seeds 42 73 2026 --require-gpu
```

BNM data is read from `data/raw/daily_1200.csv` without refreshing it. The old `--force-download` option now rejects training-time refresh. To use a new BNM snapshot, retain the old dataset and supply a separately prepared rates panel through the Python API in a new experiment. No Dukascopy connection or dependency is used.

Every run writes to a new `experiments/v24_<timestamp>_<id>/` directory. Explicit `--output-dir` must not exist and must lie outside the project's data/artifacts directories. Outputs are not automatically loaded by the app. Repeated model training must never be implemented by copying outputs directly into working artifact slots.

Controlled ablation, with identical dates/seeds/budget:

```sh
python scripts/train_model.py --architecture original --objective original --require-gpu
python scripts/train_model.py --architecture original --objective horizon_balanced --require-gpu
python scripts/train_model.py --architecture residual --objective horizon_balanced --require-gpu
```

Compare early-stop validation losses only when objectives are identical. Otherwise compare validation/selection forecast errors on matched dates; keep the number of experiments bounded. Selection error is not independent final-test evidence. The seven available 63-session selection origins are below the existing eight-origin policy threshold, so the conservative random-walk fallback is retained.

## Final evaluation and promotion

Once the candidate is frozen:

```sh
python scripts/evaluate_v24_run.py experiments/<chosen_run>
```

This verifies source and candidate hashes, creates an exclusive first-access ledger, and evaluates without retraining. It refuses repeated access and does not activate the candidate. An interrupted final evaluation retains its access ledger; inspect it rather than deleting it to rerun silently. Prefer this command over the trainer's explicit `--evaluate-test` one-shot option.

Before promotion, require complete parity checks, no data leakage, frozen matched-date comparisons against V2.2 and all benchmarks, rejection rates, calibration coverage and uncertainty intervals. V2.2 and V2.4 have different aligned observation calendars: their old summary CSV scores are not a fair direct comparison. Prior research has already exposed historical test results; reserve a genuinely new prospective period for independent confirmation. No promotion or replacement is included here.

## Verification and remaining limits

Read `AUDIT_V2_4_1.md` and `evidence/`. The included two-member, one-epoch run is explicitly an integration smoke run, not the finished three-seed trained release. ARIMA was disabled for that smoke run only; the production training default retains it. Full 300-epoch/GPU research, ablations, independent V2.2 comparisons and a prospective test have not been completed. No forecast-superiority claim is made.
