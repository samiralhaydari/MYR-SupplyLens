# MYR SupplyLens V2.4.1 audit and validation

Date: 15 November 2026. Status: audited development candidate; not promoted.

## Scope and recovered decisions

Audited the supplied V2.4 source release (182 archive entries), its source patch and latest preflight notebook, alongside the V2.2 release (171 entries), earlier evaluation findings and GitHub audit. Inventoried all source, pages, requirements, tests, datasets, scalers, metadata and model files. Reviewed the data/alignment, training, decoding, inference, evaluation, selection, UI reporting and release workflows, then exercised the full supplied test suite and app pages. This is an audit of the supplied snapshots, not a claim that the live Drive/GitHub deployment is identical. No live SQLite database was included; no live project or deployed app was modified.

Preserved decisions: BNM daily reference-rate data only; no Dukascopy; V2.3 remains archived; seven-currency joint CNN-LSTM; 180-step input, 165 channels; three-seed default; independent benchmark validation; chronological train/validation/selection/calibration/test boundaries; no test-based tuning or misleading superiority claims.

## Findings and disposition

| Priority | Finding | Evidence and action |
|---|---|---|
| Critical | V2.4 target returns did not always match the aligned future rate path | Returns were computed before common-row filtering. 24/2,737 training windows exceeded 1e-6 reconstruction error; maximum cumulative log-path error was 0.0770474. Recomputed V2.4 targets after alignment. All partitions now reconstruct to maximum 1.2654e-8 error. V2.2 artifacts and its historical target builder remain untouched. |
| High | V2.2 selection-policy fallback regression | `_bundle_paths` excluded v2_2 after MODEL_VERSION changed to v2_4; the report card also hid non-current policies. Fixed both, with a test loading the retained V2.2 bundle. |
| High | Training overwrote existing V2.4 members/scalers/metadata | Fixed filenames were atomically replaced individually, which did not provide whole-bundle safety. New runs use exclusively created experiment directories and never activate automatically. |
| High | Notebook overlaid original source and refreshed the dataset | Replaced workflow with a new sibling copy and protected-file hash comparisons. Training reads a frozen BNM CSV and refuses `force_download=True`. The original app's refresh behavior is unchanged. |
| High | Auxiliary target multiplication was presented as explicit loss weighting | Multiplying targets under a shared Huber loss changes residual scale and the Huber transition; it is not the documented group weight. Added an explicit horizon-balanced loss, undoing encoded multiplication before weighting and supervising blended cumulative forecast errors. |
| Medium | No real V2.4 Keras/Lite serialization test | Existing V2.4 test used a handcrafted synthetic archive; saved-model tests covered the existing legacy files. Added real model optimization, save/reload and Lite parity for original and residual architectures. |
| Medium | Final test was evaluated automatically on each training run | Withheld by default; separate frozen-run evaluator verifies hashes and records exclusive first access. No final-test forecasts were computed in the integration runs. |
| Medium | UI contained unsupported/hard-coded claims | Removed the assertion that every interval crosses zero or errors are always within a percent or two, corrected V2.2 selection wording, and explicitly labelled the old architecture diagram. |
| Medium | Rate cache ignored historical revisions | Cache key previously tracked only row count/latest date/latest value. It now hashes panel contents including dates and columns. |
| Medium | Environment was not reproducibly locked | Built a fresh isolated environment, checked dependencies, recorded exact tested package versions and retained existing broad requirements separately. |
| Medium | Experiment metadata could leak into source exports | Added `experiments` to the source exporter exclusion set. |

## CNN-LSTM development changes

The original builder is retained through `architecture='original'`. The residual candidate adds two-convolution residual blocks at (kernel,dilation) = (5,2), (9,3), (3,8), with per-timestep feature normalization and GELU. Together with the initial kernel-3 stem, the convolutional receptive field is 99 steps versus the original 35. A 96-unit sequence LSTM, four temporal attention heads, 192/128 dense layers and independent currency decoders remain. This does not implement bidirectional LSTM or a parallel multiscale branch architecture; those were earlier proposals, not the recovered implementation.

The loss combines daily-return Huber error, equally weighted direct-horizon errors, and the cumulative errors of the actual 0.35-blended output at 5/21/42/63 steps. Each group's error is normalized consistently with the retained target encoding. Group weights are 1/2.5/1. Model capacity and objective alignment have improved; forecast accuracy has not yet been established.

Official implementation references: [Keras regression losses](https://keras.io/api/losses/regression_losses/) and [TensorFlow serialization](https://www.tensorflow.org/guide/keras/serialization_and_saving). The actual numerical behavior was checked locally rather than inferred from documentation.

## Validation performed

- Incoming V2.4 baseline: **38 tests passed**.
- Final candidate: **46 tests passed**, including real original/residual CNN-LSTM gradient, serialization, Lite parity, target alignment, cache revision, frozen data loading and no-overwrite checks.
- Dependency resolution: `pip check` passed in a fresh Python 3.12.14 environment. Exact versions are in `requirements-validated-py312.txt`; TensorFlow 2.21.0 and NumPy 2.4.6 were used.
- Streamlit AppTest: main app plus all six pages completed with zero exceptions. Database and forecast output paths were redirected to temporary directories, and data loading used the supplied snapshot. This checks execution and default flows, not visual browser rendering or every interaction.
- Existing saved-model Keras/Lite parity tests passed. Legacy scaler version warnings remain visible; they were not dismissed as universally harmless. The retained bundles were exercised in real page forecasts.
- Full-shape real-data integration: **all seven currencies**, **180 × 165 input**, **2,737 training windows**, seeds **42 and 73**, **one epoch each**, CPU. Both models trained, checkpointed, passed Keras/Lite parity, and produced per-currency selection and calibration reports. **ARIMA was disabled only for this bounded smoke run**. Full training defaults to ARIMA enabled and three seeds.
- Original protected files: **124/124 byte-identical** between V2.2 and the incoming V2.4 release, and unchanged in the final development copy. Original archives were retained without modification.
- New notebook code cells compile. The notebook was not executed in a connected Colab session; the local equivalent training path was executed.
- Full-data audit reconstruction checks inspect target consistency, not forecast performance. Final-test evaluation remains withheld. No superiority or final-test metric is claimed.

The finalized smoke run under `experiments/verified_integration/` includes the exact source snapshot, configuration, input snapshot, hashes, trained smoke models, scalers and reports. Its models are not production artifacts. `run.json` records completion and `final_test_evaluated=false`.

## Data and split inventory

Frozen input: 7,359 dates × 27 currencies, 2 January 1997 through 13 November 2026. Seven-currency V2.4 alignment retains 4,966 observations. Partition window counts: train 2,737; early-stop validation 435; selection 434; calibration 187; test 683. Targets stay inside their partition; earlier input history may cross a boundary because it is available at the forecast origin.

## Open issues and next research gates

1. **Accuracy remains unproven.** Run bounded, same-date/same-seed ablations: corrected original architecture/objective, original architecture/new objective, and residual/new objective. The shipped one-epoch integration is not a finished model. Do not promote based on parameter count, training loss or smoke-run policy winners.
2. **Long-horizon selection sample is too small.** At 63-session spacing, the 434-window selection partition supplies seven non-overlapping origins, below the established minimum of eight. The random-walk fallback is retained. Do not relax the threshold merely to select CNN-LSTM; any split redesign must be declared before final evaluation.
3. **V2.2 comparison needs common calendar origins.** Different lookbacks/features produce different aligned calendars and split dates. Comparing old aggregate CSVs directly would be misleading. Regenerate paired forecasts on locked common dates without retraining the preserved baseline.
4. **Historical holdout has prior exposure.** Earlier versions already reported results on related historical periods. A nominally untouched V2.4 partition is not a genuinely new research holdout; reserve future observations for independent confirmation.
5. **V2.2 target alignment remains a historical limitation.** Do not change or retrain preserved artifacts silently. A separately named corrected baseline would be a new experiment.
6. **Live download validation is shallow.** The legacy app refresh routine checks for a date header before replacing its cached CSV and does not archive old downloads. The new trainer avoids that path entirely; a future app-data change should add schema/content validation and versioned snapshots.
7. **Corrupt present bundles do not consistently fall back.** Discovery checks presence; loading corrupt metadata/weights can still produce an error instead of trying a prior family. Isolated candidate outputs prevent this change from destabilizing the deployed bundle, but production promotion should add full-bundle validation.
8. **Joint latest-date alignment needs hardening.** The existing inference code can pair an older complete joint feature window with a newer target-currency spot if another currency is missing. Verify same-date completeness before promotion. Supplied snapshot default page smoke tests passed.
9. **Rejected forecasts and calibration uncertainty matter.** Existing metrics can score valid forecasts only. Any promotion comparison must report rejection/availability rates and matched origins, not only average error; empirical calibration bands do not guarantee future coverage.
10. **GPU/full training and live deployment were not validated here.** CPU execution, saved-model compatibility and app test execution passed. No external project was overwritten or deployed.

Next concrete step: run the isolated Colab notebook on GPU, keep the final evaluation switch off during ablation, then evaluate one frozen candidate and prepare a verified promotion bundle only if the evidence supports it.
