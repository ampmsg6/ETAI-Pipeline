"""Compare tuned XGBoost with/without fold-fitted selection on paired outer folds."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

from joblib import parallel_config
import pandas as pd
from threadpoolctl import threadpool_limits

from main import load_config, run_pipeline
from scripts.compare_week7 import statistical_settings
from src.evaluate import corrected_paired_test
from src.model import default_model_config
from src.results import comparison_key


COURSE_REFERENCE = "sofiacper/ETAI-Pipeline@7dc7713640fe3b97f16be6df6be67ce417f9d950"


def matching_run(config):
    """Resume only complete runs of this exact statistical recipe and software."""
    board_path = Path(config["output"]["scoreboard"])
    if not board_path.exists():
        return None
    board = pd.read_csv(board_path, dtype={"run": str})
    matches = board[(board["comparison_key"] == comparison_key(config))
                    & (board["model"] == "xgboost")
                    & (board["metric"] == config["cv"]["selection_metric"])
                    & (board["trials"] == config["tuning"]["n_trials"])]
    for _, row in matches.iloc[::-1].iterrows():
        prefix = Path(row["report_path"]).with_suffix("")
        paths = [Path(f"{prefix}{suffix}") for suffix in
                 (".txt", "_config.json", "_default.csv", "_nested.csv", "_trials.csv", "_features.csv")]
        if all(path.exists() for path in paths):
            saved = json.loads(Path(f"{prefix}_config.json").read_text(encoding="utf-8"))
            if statistical_settings(saved) == statistical_settings(config):
                return row.to_dict()
    return None


def export_tables(rows, configs, publish_dir):
    """Publish both recipes, their paired folds, selected features and corrected test."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(publish_dir / "scoreboard.csv", index=False)
    metrics, folds, features, trials = [], [], [], []
    for row in rows:
        recipe = row["recipe"]
        prefix = Path(row["report_path"]).with_suffix("")
        for source in ("default", "nested"):
            table = pd.read_csv(f"{prefix}_{source}.csv")
            for name in configs[0]["cv"]["metrics"]:
                metrics.append({"recipe": recipe, "source": source, "metric": name,
                                "mean": table[name].mean(), "std": table[name].std(ddof=1)})
            table.insert(0, "recipe", recipe)
            table.insert(1, "source", source)
            folds.append(table)
        for suffix, destination in (("_features.csv", features), ("_trials.csv", trials)):
            table = pd.read_csv(f"{prefix}{suffix}")
            table.insert(0, "recipe", recipe)
            destination.append(table)
        shutil.copyfile(row["report_path"], publish_dir / f"{recipe}.txt")
        shutil.copyfile(f"{prefix}_config.json", publish_dir / f"{recipe}_config.json")
    pd.DataFrame(metrics).to_csv(publish_dir / "metrics.csv", index=False)
    for name, tables in (("folds", folds), ("features", features), ("trials", trials)):
        pd.concat(tables, ignore_index=True).to_csv(publish_dir / f"{name}.csv", index=False)
    config = configs[0]
    a, b = rows
    score_cols = [f"fold_{i}" for i in range(1, config["cv"]["n_splits"] + 1)]
    comparison = corrected_paired_test([a[col] for col in score_cols], [b[col] for col in score_cols],
                                       1 / (config["cv"]["n_splits"] - 1))
    pd.DataFrame([{"recipe_a": a["recipe"], "recipe_b": b["recipe"],
                   **comparison}]).to_csv(publish_dir / "selection_comparison.csv", index=False)
    metadata = {
        "course_reference": COURSE_REFERENCE,
        "comparison_key": comparison_key(config),
        "selection_metric": config["cv"]["selection_metric"],
        "n_trials": config["tuning"]["n_trials"],
        "outer_folds": config["cv"]["n_splits"], "inner_folds": config["tuning"]["n_splits"],
        "cv_seed": config["cv"]["random_state"], "tuning_seed": config["tuning"]["random_state"],
        "reserved_test_evaluated": False,
        "data_sha256": hashlib.sha256(Path(config["data"]["path"]).read_bytes()).hexdigest(),
        "recipes": [row["recipe"] for row in rows],
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "scipy", "optuna", "PyYAML", "xgboost")},
    }
    (publish_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nSelection on minus off:", comparison, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scoreboard", default="results/runs.csv")
    parser.add_argument("--output-dir", default="results/week8")
    parser.add_argument("--publish-dir", default="reports/week8")
    args = parser.parse_args()
    if args.trials < 1 or args.parallel_folds < 1:
        parser.error("trials and parallel-folds must be positive")
    base = load_config()
    base["tuning"].update(enabled=True, n_trials=args.trials)
    base["cv"]["n_jobs"] = args.parallel_folds
    base["model"] = default_model_config("xgboost", base["cv"]["random_state"])
    base["model"]["params"]["n_jobs"] = 1
    base["output"].update(results_dir=args.output_dir, scoreboard=args.scoreboard)
    rows, configs = [], []
    for enabled in (True, False):
        config = deepcopy(base)
        config["feature_selection"]["enabled"] = enabled
        recipe = f"xgboost_selection_{'on' if enabled else 'off'}"
        start = time.perf_counter()
        row = matching_run(config) if args.resume else None
        if row is None:
            print(f"\nStarting {recipe}: {args.trials} trials, {args.parallel_folds} CV workers", flush=True)
            with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
                row = run_pipeline(config)["score_row"]
        else:
            print(f"Reusing completed {recipe}", flush=True)
        rows.append({"recipe": recipe, **row})
        configs.append(config)
        print(f"Completed {recipe} in {time.perf_counter() - start:.1f}s", flush=True)
    export_tables(rows, configs, Path(args.publish_dir))


if __name__ == "__main__":
    main()
