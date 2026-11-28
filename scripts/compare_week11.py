"""Week 11: three anomaly-score recipes on the same development folds."""
import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import time

from joblib import parallel_config
import pandas as pd
from threadpoolctl import threadpool_limits

from main import load_config, run_pipeline
from scripts.compare_week7 import statistical_settings
from src.artifacts import library_versions
from src.evaluate import corrected_paired_test
from src.results import comparison_key


METHODS = ("none", "isolation_forest", "lof")
COURSE_COMMIT = "bcd7b5386d3a5b6130432c560862b60085425ede"


def matching_run(config):
    board_path = Path(config["output"]["scoreboard"])
    if not board_path.exists():
        return None
    board = pd.read_csv(board_path, dtype={"run": str})
    for row in board.iloc[::-1].to_dict("records"):
        if row["comparison_key"] != comparison_key(config):
            continue
        prefix = Path(row["report_path"]).with_suffix("")
        if not all(Path(f"{prefix}{suffix}").exists() for suffix in
                   (".txt", "_config.json", "_default.csv", "_features.csv", "_oof.csv")):
            continue
        model_path = Path(row["model_path"])
        if not model_path.exists() or not model_path.with_suffix(".json").exists():
            continue
        saved = json.loads(Path(f"{prefix}_config.json").read_text(encoding="utf-8"))
        if statistical_settings(saved) == statistical_settings(config):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="results/week11")
    parser.add_argument("--publish-dir", default="reports/week11")
    args = parser.parse_args()
    if args.parallel_folds < 1:
        parser.error("parallel-folds must be positive")
    base = load_config()
    base["tuning"]["enabled"] = False
    base["cv"]["n_jobs"] = args.parallel_folds
    base["model"]["params"]["n_jobs"] = 1
    base.pop("prediction", None)
    base["output"].update(results_dir=args.output_dir, scoreboard=f"{args.output_dir}/runs.csv")
    folder = Path(args.publish_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rows, metrics, folds, features = [], [], [], []
    for strategy in METHODS:
        config = deepcopy(base)
        config["anomaly"]["method"] = strategy
        config["output"]["model_path"] = f"models/week11/{strategy}.joblib"
        row = matching_run(config) if args.resume else None
        if row is None:
            start = time.perf_counter()
            print(f"\nStarting {strategy}", flush=True)
            with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
                row = run_pipeline(config)["score_row"]
            row["elapsed_seconds"] = time.perf_counter() - start
        else:
            print(f"Reusing {strategy}: {row['run']}", flush=True)
            published = folder / "scoreboard.csv"
            if published.exists():
                previous = pd.read_csv(published, dtype={"run": str})
                previous = previous[previous["run"] == row["run"]]
                if not previous.empty and "elapsed_seconds" in previous:
                    row["elapsed_seconds"] = float(previous.iloc[-1]["elapsed_seconds"])
        rows.append({"recipe": strategy, **row})
        prefix = Path(row["report_path"]).with_suffix("")
        table = pd.read_csv(f"{prefix}_default.csv")
        for name in config["cv"]["metrics"]:
            metrics.append({"anomaly": strategy, "metric": name,
                            "mean": table[name].mean(), "std": table[name].std(ddof=1)})
        table.insert(0, "anomaly", strategy)
        folds.append(table)
        feature_table = pd.read_csv(f"{prefix}_features.csv")
        feature_table.insert(0, "anomaly", strategy)
        features.append(feature_table)
        shutil.copyfile(row["report_path"], folder / f"{strategy}.txt")
        shutil.copyfile(f"{prefix}_config.json", folder / f"{strategy}_config.json")
        print(f"Completed {strategy}: ROC-AUC {row['mean']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "scoreboard.csv", index=False)
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(folder / "folds.csv", index=False)
    pd.concat(features, ignore_index=True).to_csv(folder / "features.csv", index=False)
    comparisons = []
    k = base["cv"]["n_splits"]
    for a, b in combinations(rows, 2):
        test = corrected_paired_test([a[f"fold_{i}"] for i in range(1, k + 1)],
                                    [b[f"fold_{i}"] for i in range(1, k + 1)], 1 / (k - 1))
        comparisons.append({"anomaly_a": a["recipe"], "anomaly_b": b["recipe"], **test})
    pd.DataFrame(comparisons).to_csv(folder / "anomaly_comparisons.csv", index=False)
    metadata = {
        "course_reference": f"sofiacper/ETAI-Pipeline@{COURSE_COMMIT}",
        "selection_metric": base["cv"]["selection_metric"], "tuning_enabled": False,
        "model": base["model"], "feature_selection": base["feature_selection"],
        "folds": k, "cv_seed": base["cv"]["random_state"],
        "reserved_test_evaluated": False, "test_train_ratio": 1 / (k - 1),
        "comparison_key": comparison_key(base),
        "multiple_comparisons": "three exploratory pairwise tests; unadjusted p-values",
        "versions": library_versions(),
        "data_sha256": hashlib.sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
        "notebook_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path("../../aula 11").glob("*.ipynb"))},
        "runs": {row["recipe"]: row["run"] for row in rows},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    from scripts.profile_week11 import profile_anomalies
    profile_anomalies(base, folder)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
