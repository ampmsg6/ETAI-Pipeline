"""Run the five models and Challenge 1 ablations, with resumable full-run checkpoints."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from joblib import parallel_config
import numpy as np
import pandas as pd

from main import load_config, run_pipeline
from src.evaluate import corrected_paired_test
from src.model import default_model_config
from src.preprocessing import ENGINEERED_FEATURES, configure_engineered_features


MODELS = ("logistic_regression", "decision_tree", "random_forest", "oblique_random_forest", "adaboost")
FEATURE_MODELS = ("logistic_regression", "random_forest", "adaboost")


def statistical_config(config):
    """Worker counts affect execution cost, rather than the seeded experiment."""
    result = deepcopy(config)
    result["cv"].pop("n_jobs", None)
    result["model"].get("params", {}).pop("n_jobs", None)
    return result


def fingerprint(config):
    """Resume only if the recipe, dependencies, code and input CSV still match."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    paths = [Path("main.py"), *sorted(Path("src").glob("*.py")), Path(config["data"]["path"])]
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "optuna", "PyYAML"):
        digest.update(f"{name}={version(name)}".encode())
    return digest.hexdigest()


def summarize(result, model, features, elapsed):
    default, nested, study = result["default_scores"], result["nested_scores"], result["study"]
    return {
        "model": model, "features": features,
        "default_mean": float(default["validation"].mean()),
        "default_std": float(default["validation"].std(ddof=1)),
        "default_gap": float(default["gap"].mean()),
        "nested_mean": float(nested["validation"].mean()),
        "nested_std": float(nested["validation"].std(ddof=1)),
        "nested_gap": float(nested["gap"].mean()),
        "best_trial": float(study.best_value),
        "optimism": float(nested["inner_best"].mean() - nested["validation"].mean()),
        "best_params": study.best_params,
        "outer_scores": nested["validation"].tolist(),
        "default_scores": default["validation"].tolist(),
        "seconds": elapsed, "report_path": result["report_path"],
    }


def export_tables(records, publish_dir, base):
    publish_dir.mkdir(parents=True, exist_ok=True)
    baseline = {r["model"]: r for r in records if not r["features"]}
    summary = pd.DataFrame([{**r, "features": ",".join(r["features"]),
                             "best_params": json.dumps(r["best_params"], sort_keys=True)}
                            for r in records]).drop(columns=["outer_scores", "default_scores"])
    summary.to_csv(publish_dir / "comparison.csv", index=False)
    fold_rows = [
        {"model": r["model"], "features": ",".join(r["features"]), "fold": fold,
         "default": default, "nested": nested}
        for r in records for fold, (default, nested) in enumerate(
            zip(r["default_scores"], r["outer_scores"]), 1
        )
    ]
    pd.DataFrame(fold_rows).to_csv(publish_dir / "folds.csv", index=False)
    comparisons = []
    ratio = 1 / (base["cv"]["n_splits"] - 1)
    for r in records:
        if r["features"] and r["model"] in baseline:
            original = baseline[r["model"]]
            differences = np.asarray(r["outer_scores"]) - original["outer_scores"]
            comparisons.append({
                "model": r["model"], "features": ",".join(r["features"]),
                "difference_std": float(differences.std(ddof=1)),
                **corrected_paired_test(r["outer_scores"], original["outer_scores"], ratio),
            })
    if comparisons:
        pd.DataFrame(comparisons).to_csv(publish_dir / "feature_comparisons.csv", index=False)
    pairs = []
    if baseline:
        best = max(baseline, key=lambda name: baseline[name]["nested_mean"])
        for other in baseline:
            if other != best:
                pairs.append({"model_a": best, "model_b": other, **corrected_paired_test(
                    baseline[best]["outer_scores"], baseline[other]["outer_scores"], ratio
                )})
        pd.DataFrame(pairs).to_csv(publish_dir / "model_comparisons.csv", index=False)
    metadata = {
        "n_trials": base["tuning"]["n_trials"], "outer_folds": base["cv"]["n_splits"],
        "inner_folds": base["tuning"]["n_splits"], "metric": base["cv"]["scoring"],
        "cv_seed": base["cv"]["random_state"], "tuning_seed": base["tuning"]["random_state"],
        "reserved_test_evaluated": False,
        "data_sha256": hashlib.sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "scikit-learn", "scipy", "optuna", "PyYAML"
        )},
        "completed_runs": len(records),
    }
    (publish_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="results/week6")
    parser.add_argument("--publish-dir", default="reports/week6")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--features", nargs="+", choices=ENGINEERED_FEATURES,
                        default=list(ENGINEERED_FEATURES), help="Individual feature ablations to run")
    parser.add_argument("--parallel-folds", type=int, default=5,
                        help="CV workers for ensembles; each forest uses one worker per fit")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.parallel_folds < 1:
        parser.error("--parallel-folds must be at least 1")
    base = load_config()
    # Retain the Week 6 accuracy objective after the Week 7 config migration.
    base["cv"].update(metrics=["accuracy"], selection_metric="accuracy", scoring="accuracy")
    base["output"]["scoreboard"] = str(Path(args.output_dir) / "runs.csv")
    base["tuning"].update(enabled=True, n_trials=args.trials)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(model, []) for model in args.models]
    if not args.baseline_only:
        jobs += [(model, [feature]) for feature in args.features
                 for model in FEATURE_MODELS if model in args.models]
    records = []
    for number, (model, features) in enumerate(jobs, 1):
        config = deepcopy(base)
        config["model"] = default_model_config(model, config["cv"]["random_state"])
        if model in ("random_forest", "oblique_random_forest", "adaboost"):
            config["cv"]["n_jobs"] = args.parallel_folds
            if model != "adaboost":
                config["model"]["params"]["n_jobs"] = 1
        configure_engineered_features(config, features)
        config["output"]["results_dir"] = str(output_dir)
        key = f"{model}__{'_'.join(features) or 'baseline'}"
        checkpoint = output_dir / f"{key}.json"
        signature = fingerprint(config)
        print(f"\n[{number}/{len(jobs)}] {key}", flush=True)
        saved = json.loads(checkpoint.read_text(encoding="utf-8")) if args.resume and checkpoint.exists() else None
        reusable = False
        if saved and Path(saved["report_path"]).exists():
            saved_config_path = Path(saved["report_path"]).with_suffix("").with_name(
                Path(saved["report_path"]).stem + "_config.json"
            )
            if saved_config_path.exists():
                saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
                reusable = (saved.get("fingerprint") == fingerprint(saved_config)
                            and statistical_config(saved_config) == statistical_config(config))
        if reusable:
            row = saved
            print("Reusing matching completed run.", flush=True)
        else:
            start = time.perf_counter()
            # Limit native threads in the workers, avoiding nested oversubscription.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                result = run_pipeline(config)
            row = summarize(result, model, features, time.perf_counter() - start)
            row["fingerprint"] = signature
            checkpoint.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(f"Completed in {row['seconds']:.1f}s", flush=True)
        records.append(row)
        export_tables(records, Path(args.publish_dir), base)
    print(f"\nSaved {len(records)} runs to {args.publish_dir}; reserved test not evaluated.", flush=True)


if __name__ == "__main__":
    main()
