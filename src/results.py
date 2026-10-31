"""Saving each run: its scores to the runs table, and its full report to disk."""
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, config: dict, report_text: str, run: str | None = None) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<run>.txt` (the folder is
    created if needed) and returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)
    run = run or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    tuning_line = (f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
                   f"random_state={tuning['random_state']}" if tuning.get("enabled") else "off")
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {cv['metrics']}  selection metric: {cv['selection_metric']}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)
    Path(path).with_name(f"run_{run}_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8",
    )
    return path



def comparison_key(config: dict) -> str:
    """Identify paired experiments by data, cleaning, split, folds and software."""
    settings = {key: config[key] for key in ("data", "diagnostics", "test_set")}
    settings["cv"] = {key: config["cv"].get(key) for key in ("n_splits", "shuffle", "random_state")}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(Path(config["data"]["path"]).read_bytes())
    for name in ("preprocessing.py", "evaluate.py", "tuning.py", "model.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "optuna"):
        digest.update(f"{name}={version(name)}".encode())
    return digest.hexdigest()
