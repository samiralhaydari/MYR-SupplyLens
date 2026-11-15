from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
SCALER_DIR = PROJECT_ROOT / "artifacts" / "scalers"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
FORECAST_DIR = PROJECT_ROOT / "artifacts" / "forecasts"
DB_PATH = PROJECT_ROOT / "data" / "supplylens.sqlite"

DATA_URL = "https://storage.data.gov.my/finsector/exr/daily_1200.csv"
DATASET_PAGE = "https://data.gov.my/data-catalogue/exchangerates_daily_1200"

PRIORITY_CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]
CURRENCY_NAMES = {
    "USD": "US Dollar", "CNY": "Chinese Yuan", "IDR": "Indonesian Rupiah",
    "SGD": "Singapore Dollar", "THB": "Thai Baht", "JPY": "Japanese Yen",
    "KRW": "South Korean Won", "PHP": "Philippine Peso", "VND": "Vietnamese Dong",
    "BND": "Brunei Dollar", "EUR": "Euro", "GBP": "British Pound",
    "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "CHF": "Swiss Franc",
    "HKD": "Hong Kong Dollar", "TWD": "Taiwan Dollar", "INR": "Indian Rupee",
    "NZD": "New Zealand Dollar"
}

# This hosted release deliberately serves the preserved, complete V2.2 bundles.
# V2.4.1 remains a research candidate until its full weights/scalers are separately
# exported and verified; never point the live app at smoke-test or report-only outputs.
MODEL_VERSION = "v2_2"
LEGACY_MODEL_VERSIONS = ["v2_1", "v2", "v1"]
LOOKBACK = 90  # Legacy model default; V2.4 records its own lookback in metadata.
V24_LOOKBACK = 180
FORECAST_HORIZON = 63
V24_ENSEMBLE_SEEDS = [42, 73, 2026]
V24_AUXILIARY_LOSS_WEIGHT = 2.5
V24_AUXILIARY_BLEND = 0.35
DISPLAY_HORIZONS = [5, 21, 42, 63]
# Test origins use common 63-observation spacing to avoid overlapping test targets.
WALK_FORWARD_SPACING = FORECAST_HORIZON
# V2.1 chronological partitions: 60% train, 10% early-stop validation,
# 10% method selection, 5% calibration, 15% untouched final test.
SPLIT_FRACTIONS = {
    "train": 0.60,
    "validation": 0.10,
    "selection": 0.10,
    "calibration": 0.05,
    "test": 0.15,
}
MIN_SELECTION_ORIGINS = 8
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_ORIGIN_STRIDE = 5
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
ARIMA_MAX_HISTORY = 756
FEATURE_COLUMNS = [
    "log_rate",
    "return_1d",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "range_20",
]

BENCHMARK_METHODS = [
    "CNN-LSTM",
    "Random walk",
    "Historical drift (252)",
    "Recent drift (20)",
    "ARIMA(1,1,1)",
]


# V2.4 expands the historical feature set while preserving FEATURE_COLUMNS for
# V1/V2/V2.1/V2.2 inference compatibility.
V24_CURRENCY_FEATURE_COLUMNS = [
    "log_rate",
    "return_1d",
    "return_2d",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "volatility_120",
    "volatility_ratio_5_20",
    "volatility_ratio_20_60",
    "zscore_20",
    "zscore_60",
    "range_20",
    "range_60",
    "drawdown_20",
    "drawdown_60",
    "ema_gap_10_30",
    "ema_gap_20_60",
]
V24_GLOBAL_FEATURE_COLUMNS = [
    "panel_median_return_1d",
    "panel_median_return_5d",
    "panel_median_volatility_20",
    "panel_return_dispersion_20",
]
V24_FEATURE_VERSION = "v2_4"
