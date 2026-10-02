"""Compare Week 4 model and preprocessing choices on the same development folds."""

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from main import load_config
from src.data import load_data
from src.model import build_model
from src.preprocessing import (
    build_preprocessor, clean_dataset, drop_duplicate_rows,
    split_dev_test, split_features_target,
)


def main():
    base = load_config()
    raw = load_data(base["data"]["path"])
    clean = drop_duplicate_rows(clean_dataset(raw, base), base["diagnostics"]["id_column"])
    X, y, extras = split_features_target(clean, base)
    X_dev, _, y_dev, _, _, _ = split_dev_test(
        X, y, extras, base["test_set"]["size"], base["test_set"]["random_state"]
    )
    cv = StratifiedKFold(
        n_splits=base["cv"]["n_splits"], shuffle=True,
        random_state=base["cv"]["random_state"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25,
        random_state=base["cv"]["random_state"], stratify=y_dev,
    )

    candidates = [
        ("dummy", "dummy", {}, "median", "target", "robust"),
        ("logistic_baseline", "logistic_regression", {"max_iter": 1000}, "median", "target", "robust"),
        ("logistic_knn", "logistic_regression", {"max_iter": 1000}, "knn", "target", "robust"),
        ("logistic_onehot", "logistic_regression", {"max_iter": 1000}, "median", "onehot", "robust"),
        ("logistic_count", "logistic_regression", {"max_iter": 1000}, "median", "count", "robust"),
        ("logistic_standard", "logistic_regression", {"max_iter": 1000}, "median", "target", "standard"),
        ("decision_tree", "decision_tree", {"random_state": 42}, "median", "target", "robust"),
        ("random_forest", "random_forest", {"n_estimators": 100, "random_state": 42, "n_jobs": 1}, "median", "target", "robust"),
        ("random_forest_leaf5", "random_forest", {"n_estimators": 100, "min_samples_leaf": 5, "random_state": 42, "n_jobs": 1}, "median", "target", "robust"),
    ]
    rows = []
    finalist_pipelines = {}
    for name, model_type, params, imputer, encoder, scaler in candidates:
        config = deepcopy(base)
        config["model"] = {"type": model_type, "params": params}
        config["preprocessing"].update({"numeric_imputer": imputer, "encoder": encoder, "scaler": scaler})
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor(config)),
            ("model", build_model(config["model"])),
        ])
        if name in {"logistic_baseline", "logistic_knn", "random_forest_leaf5"}:
            finalist_pipelines[name] = pipeline
        holdout = clone(pipeline).fit(X_train, y_train)
        holdout_accuracy = accuracy_score(y_val, holdout.predict(X_val))
        scores = cross_validate(
            pipeline, X_dev, y_dev, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
            return_train_score=True, n_jobs=base["cv"].get("n_jobs", 1),
        )
        row = {
            "candidate": name, "model": model_type, "imputer": imputer,
            "encoder": encoder, "scaler": scaler,
            "holdout_accuracy": float(holdout_accuracy),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_gap_mean": float(np.mean(scores["train_accuracy"] - scores["test_accuracy"])),
        }
        rows.append(row)
        print(
            f"{name:<22s} holdout={row['holdout_accuracy']:.4f} "
            f"CV={row['cv_accuracy_mean']:.4f} +/- {row['cv_accuracy_std']:.4f} "
            f"F1={row['cv_f1_mean']:.4f} gap={row['cv_gap_mean']:+.4f}",
            flush=True,
        )
    rows.sort(key=lambda row: row["cv_accuracy_mean"], reverse=True)
    path = Path("results/week4_comparison.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}; reserved test set was not evaluated.")

    # Check whether the small accuracy differences persist across more splits.
    repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    repeated_scores = {}
    for name, pipeline in finalist_pipelines.items():
        repeated_scores[name] = cross_validate(
            pipeline, X_dev, y_dev, cv=repeated,
            scoring={"accuracy": "accuracy", "f1": "f1"},
            n_jobs=base["cv"].get("n_jobs", 1),
        )
    baseline = repeated_scores["logistic_baseline"]
    repeated_rows = []
    for name, scores in repeated_scores.items():
        row = {
            "candidate": name,
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)),
            "accuracy_difference_vs_baseline": float(np.mean(scores["test_accuracy"] - baseline["test_accuracy"])),
            "f1_mean": float(np.mean(scores["test_f1"])),
            "f1_difference_vs_baseline": float(np.mean(scores["test_f1"] - baseline["test_f1"])),
        }
        repeated_rows.append(row)
        print(
            f"Repeated 5x3 {name:<22s} accuracy={row['accuracy_mean']:.4f} "
            f"delta={row['accuracy_difference_vs_baseline']:+.4f} "
            f"F1={row['f1_mean']:.4f} delta={row['f1_difference_vs_baseline']:+.4f}",
            flush=True,
        )
    repeated_path = Path("results/week4_repeated_cv.csv")
    with repeated_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=repeated_rows[0].keys())
        writer.writeheader()
        writer.writerows(repeated_rows)
    print(f"Saved {repeated_path}.")


if __name__ == "__main__":
    main()
