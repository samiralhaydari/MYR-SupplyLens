"""Read-only inventory and target alignment audit. Does not refresh data."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.supplylens.config import PRIORITY_CURRENCIES
    from src.supplylens.data import load_exchange_rates
    from src.supplylens.multivariate import make_multivariate_supervised_samples_v24
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protected = {}
    for folder in ['data', 'artifacts']:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rates = load_exchange_rates(raw_path=ROOT / 'data/raw/daily_1200.csv')
    data = make_multivariate_supervised_samples_v24(rates, PRIORITY_CURRENCIES)
    max_error = 0.
    for part in ['train', 'validation', 'selection', 'calibration', 'test']:
        for origin, targets in zip(data[part]['origins'], data[part]['y']):
            daily = targets.reshape(len(PRIORITY_CURRENCIES), 67)[:, :63]
            actual = np.log(data['aligned_rates'][origin+1:origin+64] / data['aligned_rates'][origin]).T
            max_error = max(max_error, float(np.max(np.abs(np.cumsum(daily, axis=1, dtype='float64') - actual))))
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'protected_sha256': protected, 'rates_shape': list(rates.shape),
              'first_date': str(rates.index.min()), 'last_date': str(rates.index.max()),
              'aligned_observations': len(data['aligned_dates']),
              'split_boundaries': data['split_boundaries'],
              'partitions': {p: len(data[p]['X']) for p in ['train', 'validation', 'selection', 'calibration', 'test']},
              'max_target_reconstruction_log_error': max_error,
              'target_reconstruction_passed': max_error < 1e-7}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    if not report['target_reconstruction_passed']:
        raise RuntimeError('Target reconstruction failed; do not train.')
    print('Read-only audit passed:', args.output)


if __name__ == '__main__':
    main()
