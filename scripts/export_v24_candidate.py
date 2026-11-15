#!/usr/bin/env python3
"""Export a verified V2.4.1 experiment as a separate, non-activating candidate ZIP.

The exporter never copies files into artifacts/, never changes MODEL_VERSION,
and refuses incomplete, modified, or report-only runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

CURRENCIES = ["USD", "CNY", "IDR", "SGD", "THB", "JPY", "KRW"]
MEMBERS = [f"cnn_lstm_v2_4_member_{i:02d}.keras" for i in (1, 2, 3)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe relative path in run manifest: {relative}")
    path = root.joinpath(*rel.parts).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Manifest path escapes candidate directory: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed V2.4.1 experiment directory")
    parser.add_argument("--output", type=Path, help="Candidate ZIP path (default: next to run directory)")
    args = parser.parse_args()

    run = args.run_dir.expanduser().resolve()
    manifest_path = run / "run.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("Candidate run status must be 'complete'.")
    if set(manifest.get("currencies", [])) != set(CURRENCIES):
        raise SystemExit("Candidate must contain the complete seven-currency universe.")
    if len(manifest.get("seeds", [])) != 3:
        raise SystemExit("Candidate must contain the three-member ensemble.")

    # Verify every file recorded by the training run's own artifact manifest.
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        path = safe_file(run, relative)
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Artifact hash mismatch or missing file: {relative}")

    # Verify that the archived source snapshot corresponds to the source used for training.
    for relative, expected in manifest.get("source_sha256", {}).items():
        source = safe_file(run, f"source_snapshot/{relative}")
        if not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"Source snapshot hash mismatch or missing file: {relative}")

    for name in MEMBERS:
        if not (run / "models" / name).is_file():
            raise SystemExit(f"Missing trained ensemble member: models/{name}")
    if not (run / "scalers/scaler_X_v2_4_shared.joblib").is_file():
        raise SystemExit("Missing shared V2.4 input scaler.")

    for currency in CURRENCIES:
        required = [
            run / f"scalers/scaler_y_v2_4_{currency}.joblib",
            run / f"metadata/metadata_v2_4_{currency}.json",
            run / f"metadata/selection_policy_v2_4_{currency}.csv",
            run / f"metadata/calibration_v2_4_{currency}.csv",
            run / f"metadata/metrics_v2_4_{currency}.csv",
        ]
        missing = [str(path.relative_to(run)) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"{currency}: missing required bundle files: {missing}")
        metadata = json.loads(required[1].read_text(encoding="utf-8"))
        if metadata.get("model_version") != "v2_4" or metadata.get("candidate_revision") != "v2.4.1":
            raise SystemExit(f"{currency}: metadata does not identify a V2.4.1 candidate.")
        if metadata.get("ensemble_model_files") != MEMBERS:
            raise SystemExit(f"{currency}: ensemble member list does not match the three expected members.")

    final_test = run / "final_test"
    if not (final_test / "ACCESS_STARTED.json").is_file() or not (final_test / "COMPLETE.json").is_file():
        raise SystemExit("A completed final_test/ directory with ACCESS_STARTED.json and COMPLETE.json is required.")
    for currency in CURRENCIES:
        for kind in ("metrics", "details"):
            path = final_test / f"{kind}_{currency}.csv"
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing/empty final-test report: final_test/{path.name}")

    output = args.output.expanduser().resolve() if args.output else run.parent / f"{run.name}_V2_4_1_Candidate.zip"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    if run == output or run in output.parents:
        raise SystemExit("Output ZIP must be outside the experiment directory.")

    include = [manifest_path, run / "input_rates.csv"]
    for folder in ("models", "scalers", "metadata", "final_test", "source_snapshot"):
        include.extend(path for path in (run / folder).rglob("*") if path.is_file())
    include = sorted(set(path for path in include if path.is_file()))
    file_hashes = {str(path.relative_to(run)): sha256(path) for path in include}
    export_readme = """# MYR SupplyLens V2.4.1 candidate package\n\nThis is a separately exported research candidate. It is not the active Streamlit model bundle.\n\nThe archive contains the completed three-member model ensemble, matching scalers and metadata,\nsource snapshot, training manifest, and final-test reports. Verify the SHA256 manifest before\nany review. This export does not activate the candidate or modify the deployed artifacts.\n\nThe live application remains on V2.2 until a separate promotion decision is made.\n"""
    export_manifest = {
        "candidate_revision": "v2.4.1",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": run.name,
        "files": file_hashes,
        "activation": "not_activated",
    }

    root_name = "MYR_SupplyLens_V2_4_1_Candidate"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root_name}/CANDIDATE_README.md", export_readme)
        archive.writestr(f"{root_name}/EXPORT_SHA256.json", json.dumps(export_manifest, indent=2))
        for path in include:
            archive.write(path, f"{root_name}/{path.relative_to(run).as_posix()}")

    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            output.unlink(missing_ok=True)
            raise SystemExit(f"ZIP verification failed at {bad}")

    print(f"Candidate export created: {output}")
    print(f"Files included: {len(include)}")
    print("Activation: NOT ACTIVATED; active artifacts/ and MODEL_VERSION were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
