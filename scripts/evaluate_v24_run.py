"""Evaluate a frozen candidate once, without retraining or activating it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / 'run.json').read_text())
    if manifest['status'] != 'complete' or manifest.get('final_test_evaluated'):
        raise RuntimeError('Require a completed candidate with final test still withheld.')
    for relative, digest in manifest['artifact_sha256'].items():
        if hashlib.sha256((run / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Candidate changed: {relative}')
    for relative, digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed since training: {relative}. Restore the frozen source first.')
    from src.supplylens.multivariate import make_multivariate_supervised_samples_v24
    from src.supplylens.multitask_training_v24 import V24TargetOutputAdapter
    from src.supplylens.forecasting import _load_network
    from src.supplylens.ensemble import PredictionEnsemble
    from src.supplylens.evaluation import evaluate_walk_forward
    rates = pd.read_csv(run / 'input_rates.csv', index_col=0, parse_dates=True)
    data = make_multivariate_supervised_samples_v24(rates, manifest['currencies'],
             lookback=manifest['lookback'], horizon=manifest['horizon'])
    if data['split_boundaries'] != manifest['split_boundaries']:
        raise RuntimeError('Split boundaries changed.')
    scaler_X = joblib.load(run / 'scalers/scaler_X_v2_4_shared.joblib')
    test = data['test']
    X = scaler_X.transform(test['X'].reshape(-1, test['X'].shape[-1])).reshape(test['X'].shape).astype('float32')
    first = json.loads((run / f"metadata/metadata_v2_4_{manifest['currencies'][0]}.json").read_text())
    ensemble = PredictionEnsemble([_load_network(run / 'models' / name) for name in first['ensemble_model_files']])
    # Exclusive directory is the holdout-access ledger, including interrupted runs.
    output = run / 'final_test'
    output.mkdir(exist_ok=False)
    (output / 'ACCESS_STARTED.json').write_text(json.dumps({'candidate_sha256': hashlib.sha256((run / 'run.json').read_bytes()).hexdigest()}))
    for index, currency in enumerate(manifest['currencies']):
        sy = joblib.load(run / f'scalers/scaler_y_v2_4_{currency}.joblib')
        adapter = V24TargetOutputAdapter(ensemble, index, sy, 63, data['auxiliary_horizons'])
        metrics, details = evaluate_walk_forward(adapter, X, sy, test['origins'],
            data['aligned_rates'][:, index], data['aligned_dates'], currency,
            spacing=63, include_arima=manifest['include_arima'])
        metrics.to_csv(output / f'metrics_{currency}.csv', index=False)
        details.to_csv(output / f'details_{currency}.csv', index=False)
    (output / 'COMPLETE.json').write_text(json.dumps({'currencies': manifest['currencies']}))
    print('Frozen final evaluation:', output)


if __name__ == '__main__':
    main()
