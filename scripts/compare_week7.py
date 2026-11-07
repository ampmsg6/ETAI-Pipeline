"""Fill the Week 7 scoreboard, retaining matching runs when --resume is used."""
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
from src.evaluate import corrected_paired_test
from src.model import default_model_config
from src.results import comparison_key


MODELS = ("dummy", "logistic_regression", "naive_bayes", "decision_tree",
          "random_forest", "oblique_random_forest", "adaboost",
          "gradient_boosting", "hist_gradient_boosting")


def statistical_settings(config):
    """Ignore output locations and CPU workers when checking whether a run can resume."""
    settings = deepcopy(config)
    settings.pop("output", None)
    settings["cv"].pop("n_jobs", None)
    settings["model"].get("params", {}).pop("n_jobs", None)
    return settings


def export_tables(board_path, publish_dir, config):
    """Publish compact tables and the full evidence needed to review each model."""
    board = pd.read_csv(board_path, dtype={"run": str})
    board = board[(board["comparison_key"] == comparison_key(config))
                  & (board["metric"] == config["cv"]["selection_metric"])]
    # A rerun supersedes an earlier run of that model in the published summary.
    board = board.drop_duplicates("model", keep="last").sort_values("mean", ascending=False)
    publish_dir.mkdir(parents=True, exist_ok=True)
    board.to_csv(publish_dir / "scoreboard.csv", index=False)
    metrics, folds = [], []
    for _, row in board.iterrows():
        prefix = Path(row["report_path"]).with_suffix("")
        for source in ("default", "nested"):
            path = Path(f"{prefix}_{source}.csv")
            if path.exists():
                table = pd.read_csv(path)
                for name in config["cv"]["metrics"]:
                    metrics.append({"model": row["model"], "source": source, "metric": name,
                                    "mean": table[name].mean(), "std": table[name].std(ddof=1)})
                table.insert(0, "model", row["model"])
                table.insert(1, "source", source)
                folds.append(table)
        # Reports include confusion matrices, fairness and chosen hyperparameters.
        shutil.copyfile(row["report_path"], publish_dir / f"{row['model']}.txt")
        shutil.copyfile(f"{prefix}_config.json", publish_dir / f"{row['model']}_config.json")
    pd.DataFrame(metrics).to_csv(publish_dir / "metrics.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(publish_dir / "folds.csv", index=False)
    comparisons = []
    best = board.iloc[0]
    score_cols = [f"fold_{i}" for i in range(1, config["cv"]["n_splits"] + 1)]
    for _, other in board.iloc[1:].iterrows():
        comparisons.append({"model_a": best["model"], "model_b": other["model"],
                            **corrected_paired_test(best[score_cols].to_numpy(dtype=float),
                                                   other[score_cols].to_numpy(dtype=float),
                                                   1 / (config["cv"]["n_splits"] - 1))})
    pd.DataFrame(comparisons).to_csv(publish_dir / "model_comparisons.csv", index=False)
    metadata = {
        "course_reference": "sofiacper/ETAI-Pipeline@a170b26b131659ec9b5aff6d818d21354c3930c2",
        "comparison_key": comparison_key(config), "selection_metric": config["cv"]["selection_metric"],
        "metrics": config["cv"]["metrics"], "n_trials": config["tuning"]["n_trials"],
        "outer_folds": config["cv"]["n_splits"], "inner_folds": config["tuning"]["n_splits"],
        "cv_seed": config["cv"]["random_state"], "tuning_seed": config["tuning"]["random_state"],
        "reserved_test_evaluated": False,
        "data_sha256": hashlib.sha256(Path(config["data"]["path"]).read_bytes()).hexdigest(),
        "completed_models": board["model"].tolist(),
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "scipy", "optuna", "PyYAML")},
    }
    (publish_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scoreboard", default="results/runs.csv")
    parser.add_argument("--output-dir", default="results/week7")
    parser.add_argument("--publish-dir", default="reports/week7")
    args = parser.parse_args()
    if args.trials < 1 or args.parallel_folds < 1:
        parser.error("trials and parallel-folds must be positive")
    base = load_config()
    base["feature_selection"]["enabled"] = False  # Week 7 predates learned selection.
    base["tuning"]["n_trials"] = args.trials
    base["cv"]["n_jobs"] = args.parallel_folds
    base["output"].update(results_dir=args.output_dir, scoreboard=args.scoreboard)
    for model in args.models:
        config = deepcopy(base)
        config["model"] = default_model_config(model, config["cv"]["random_state"])
        if "n_jobs" in config["model"]["params"]:
            config["model"]["params"]["n_jobs"] = 1
        config["tuning"]["enabled"] = model != "dummy"
        trials = args.trials if model != "dummy" else 0
        if args.resume and Path(args.scoreboard).exists():
            existing = pd.read_csv(args.scoreboard)
            matching = existing[(existing["comparison_key"] == comparison_key(config))
                                & (existing["metric"] == config["cv"]["selection_metric"])
                                & (existing["model"] == model) & (existing["trials"] == trials)]
            reusable = False
            if not matching.empty:
                report = Path(matching.iloc[-1]["report_path"])
                saved_path = report.with_name(f"{report.stem}_config.json")
                if report.exists() and saved_path.exists():
                    saved = json.loads(saved_path.read_text(encoding="utf-8"))
                    reusable = statistical_settings(saved) == statistical_settings(config)
            if reusable:
                print(f"Reusing completed {model}", flush=True)
                export_tables(args.scoreboard, Path(args.publish_dir), base)
                continue
        print(f"\nStarting {model}: {trials} trials, {args.parallel_folds} CV workers", flush=True)
        start = time.perf_counter()
        with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
            run_pipeline(config)
        export_tables(args.scoreboard, Path(args.publish_dir), base)
        print(f"Completed {model} in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
