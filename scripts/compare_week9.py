"""Week 9 option A: paired tuned Adult runs, target checks and new-row predictions."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import joblib
from joblib import parallel_config
import pandas as pd
from threadpoolctl import threadpool_limits

from main import load_config, run_pipeline
from scripts.compare_week8 import export_tables, matching_run
from src.data import load_data
from src.predict import predict_new_data
from src.preprocessing import clean_dataset


TARGETS = {"roc_auc": 0.90, "accuracy": 0.855, "f1": 0.66}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_adult.yaml")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish-dir", default="reports/week9")
    args = parser.parse_args()
    if args.trials < 1 or args.parallel_folds < 1:
        parser.error("trials and parallel-folds must be positive")
    base = load_config(args.config)
    prediction = base.pop("prediction")
    base["tuning"].update(enabled=True, n_trials=args.trials)
    base["cv"]["n_jobs"] = args.parallel_folds
    base["model"]["params"]["n_jobs"] = 1
    if base["model"]["type"] != "xgboost":
        parser.error("This comparison's two predeclared recipes use XGBoost")
    rows, configs = [], []
    for enabled in (True, False):
        config = deepcopy(base)
        config["feature_selection"]["enabled"] = enabled
        recipe = f"adult_xgboost_selection_{'on' if enabled else 'off'}"
        start = time.perf_counter()
        row = matching_run(config) if args.resume else None
        model_path = Path(row["report_path"]).with_suffix("").as_posix() + "_model.joblib" if row else None
        if row is None or not Path(model_path).exists():
            print(f"\nStarting {recipe}: {args.trials} trials, {args.parallel_folds} CV workers", flush=True)
            with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
                result = run_pipeline(config)
            row = result["score_row"]
            model_path = Path(row["report_path"]).with_suffix("").as_posix() + "_model.joblib"
            joblib.dump(result["model"], model_path)
        else:
            print(f"Reusing completed {recipe}", flush=True)
        rows.append({"recipe": recipe, **row})
        configs.append(config)
        print(f"Completed {recipe} in {time.perf_counter() - start:.1f}s", flush=True)

    folder = Path(args.publish_dir)
    export_tables(rows, configs, folder)
    target_rows = [{"recipe": row["recipe"], "metric": name, "target": target,
                    "nested_mean": row[f"mean_{name}"], "passed": row[f"mean_{name}"] >= target}
                   for row in rows for name, target in TARGETS.items()]
    pd.DataFrame(target_rows).to_csv(folder / "target_checks.csv", index=False)

    # The config declares the final recipe; do not silently choose by one noisy row.
    selected_index = 0 if base["feature_selection"]["enabled"] else 1
    chosen, chosen_config = rows[selected_index], deepcopy(configs[selected_index])
    if not all(chosen[f"mean_{name}"] >= target for name, target in TARGETS.items()):
        raise RuntimeError("The configured final recipe has not reached all three course targets")
    chosen_config["prediction"] = prediction
    model_path = Path(chosen["report_path"]).with_suffix("").as_posix() + "_model.joblib"
    # This checkpoint was produced locally by the matching completed run above.
    pipeline = joblib.load(model_path)
    predictions_path = predict_new_data(pipeline, chosen_config)
    raw, clean = load_data(base["data"]["path"]), clean_dataset(load_data(base["data"]["path"]), base)
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "assignment": "09_beyond_compas.ipynb, option A",
        "training_rows": len(raw), "development_rows": 4800, "reserved_rows": 1200,
        "target": base["data"]["target"], "sensitive_attribute": base["data"]["sensitive_attr"],
        "excluded_protected_features": ["sex", "race"],
        "cleaned_missing_counts": {name: int(clean[name].isna().sum())
                                   for name in ("workclass", "occupation", "native_country")},
        "preserved_age_17_rows": int(clean["age"].eq(17).sum()),
        "preserved_capital_gain_99999_rows": int(clean["capital_gain"].eq(99999).sum()),
        "selected_recipe": chosen["recipe"], "selected_run": chosen["run"],
        "predictions_path": predictions_path, "prediction_rows": len(pd.read_csv(predictions_path)),
        "prediction_sha256": hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
        "prediction_input_sha256": hashlib.sha256(Path(prediction["input_path"]).read_bytes()).hexdigest(),
        "implementation_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path("main.py"), *sorted(Path("src").glob("*.py"))]},
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("All three targets passed in the configured recipe:", chosen["recipe"], flush=True)
    print({name: chosen[f"mean_{name}"] for name in TARGETS}, flush=True)


if __name__ == "__main__":
    main()
