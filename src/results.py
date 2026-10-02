"""
Saving each run's results to disk.

This module writes the holdout and cross-validation report, classification
results, and fairness table to a timestamped file in `results/` so runs can be
compared after changing `config.yaml`.
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params', {})}\n"
        f"Preprocessing: {config['preprocessing']['numeric_imputer']} imputer, "
        f"{config['preprocessing']['encoder']} encoder, {config['preprocessing']['scaler']} scaler\n"
        f"Reserved test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"CV: {config['cv']['n_splits']} folds, scoring={config['cv']['scoring']}, "
        f"random_state={config['cv']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
