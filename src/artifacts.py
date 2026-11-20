"""Persist the complete fitted pipeline and its training recipe for inference."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib


LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "optuna", "PyYAML", "xgboost", "joblib")


def library_versions():
    return {name: version(name) for name in LIBRARIES}


def save_model(pipeline, config, run, development_rows, reserved_rows):
    """Save after refit, without the temporary training cache or reserved data."""
    path = Path(config.get("output", {}).get("model_path", "models/model.joblib"))
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.set_params(memory=None)
    joblib.dump(pipeline, path)
    metadata = {
        "run": run, "config": config, "versions": library_versions(),
        "python": platform.python_version(), "development_rows": development_rows,
        "reserved_rows": reserved_rows, "reserved_test_evaluated": False,
        "threshold": 0.5, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return str(path)


def load_model(path):
    """Load a trusted local artifact with its saved config and matching libraries."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    differences = [f"{name}: saved {saved}, installed {version(name)}"
                   for name, saved in metadata["versions"].items() if saved != version(name)]
    if differences:
        raise ValueError("Model library versions differ: " + "; ".join(differences))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Model checksum does not match its metadata")
    return joblib.load(path), metadata["config"]
