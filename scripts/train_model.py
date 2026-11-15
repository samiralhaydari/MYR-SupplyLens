import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplylens.config import PRIORITY_CURRENCIES, V24_ENSEMBLE_SEEDS
from src.supplylens.multitask_training_v24 import train_joint_multicurrency_model_v24


def main():
    parser = argparse.ArgumentParser(
        description="Train the MYR SupplyLens V2.4 attention CNN-LSTM ensemble."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-arima", action="store_true")
    parser.add_argument("--device", choices=["auto", "GPU", "CPU"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=V24_ENSEMBLE_SEEDS)
    parser.add_argument(
        "--currencies", nargs="+", default=PRIORITY_CURRENCIES,
        help="Currencies trained jointly; default is all seven app currencies."
    )
    parser.add_argument("--output-dir", help="New experiment directory; must not already exist")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate holdout only after freezing the candidate")
    parser.add_argument("--architecture", choices=["original", "residual"], default="residual")
    parser.add_argument("--objective", choices=["original", "horizon_balanced"], default="horizon_balanced")
    args = parser.parse_args()

    result = train_joint_multicurrency_model_v24(
        currencies=args.currencies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lookback=args.lookback,
        force_download=args.force_download,
        include_arima=not args.skip_arima,
        calibration_stride=args.calibration_stride,
        seeds=args.seeds,
        device=args.device,
        require_gpu=args.require_gpu,
        output_dir=args.output_dir,
        evaluate_test=args.evaluate_test,
        architecture=args.architecture,
        objective=args.objective,
    )
    print("\\nV2.4 joint ensemble training complete.")
    print("Experiment:", result["run_dir"])
    print("Final test evaluated:", args.evaluate_test)
    print("Model members:")
    for path in result["model_paths"]:
        print("  ", path)
    print("Currencies:", ", ".join(result["currencies"]))
    print("Training metadata:", result["metadata"])
    for currency, report in result["reports"].items():
        print(f"\\n{currency} — validation-selected methods")
        print(report["selection_policy"].to_string(index=False))
        print(f"\\n{currency} — final-test metrics (empty when withheld)")
        print(report["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
