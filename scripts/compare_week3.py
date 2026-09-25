"""Compare week 2 complete cases with week 3 cleaning on shared test IDs."""

import csv
from copy import deepcopy
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from main import load_config
from src.data import load_data
from src.model import build_model
from src.preprocessing import build_preprocessor, clean_dataset, split_features_target, split_train_test


def score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()
    return {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "train_rows": len(X_train), "test_rows": len(X_test),
    }


def main():
    config = load_config()
    raw = load_data(config["data"]["path"])
    id_column = config["diagnostics"]["id_column"]
    clean = clean_dataset(raw, config)
    X, y, extras = split_features_target(clean, config)
    X_train, X_test, y_train, y_test, _, _ = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"], random_state=config["split"]["random_state"],
    )
    train_ids, test_ids = X_train.index, X_test.index
    train_record_ids = set(clean.loc[train_ids, id_column])

    # The original pipeline dropped all incomplete rows and encoded all raw
    # categoricals before the split. Keep that behavior in this comparison.
    complete = raw.dropna()
    old_X = pd.get_dummies(complete.drop(columns=[
        config["data"]["target"], config["data"]["sensitive_attr"],
        id_column, "decile_score", "score_text",
    ]), drop_first=True)
    old_y = complete[config["data"]["target"]]
    # Baseline training keeps its complete-case duplicate copies; evaluation
    # uses one shared set of unique test records for a like-for-like score.
    old_train_ids = complete.index[complete[id_column].isin(train_record_ids)]
    shared_test_ids = test_ids.intersection(old_X.index)
    if not len(shared_test_ids):
        raise RuntimeError("No complete test rows are available for comparison")

    models = {
        "logistic_regression": {"type": "logistic_regression", "params": {"max_iter": 1000}},
        "decision_tree": {"type": "decision_tree", "params": {"random_state": 42}},
    }
    rows = []
    for name, model_config in models.items():
        old = score(build_model(model_config), old_X.loc[old_train_ids], old_y.loc[old_train_ids],
                    old_X.loc[shared_test_ids], old_y.loc[shared_test_ids])
        rows.append({"model": name, "version": "week2_complete_cases", "imputer": "dropna", **old})
        for imputer in ["median", "knn"]:
            candidate = deepcopy(config)
            candidate["preprocessing"]["numeric_imputer"] = imputer
            pipe = Pipeline([
                ("preprocessing", build_preprocessor(candidate)),
                ("model", build_model(model_config)),
            ])
            result = score(pipe, X.loc[train_ids], y.loc[train_ids],
                           X.loc[shared_test_ids], y.loc[shared_test_ids])
            rows.append({"model": name, "version": "week3_clean_shared_test", "imputer": imputer, **result})

    path = Path("results/week3_comparison.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw rows: {len(raw)}; distinct rows after cleaning: {len(clean)} "
          f"({len(raw) - len(clean)} exact copies removed)")
    print(f"Shared test rows: {len(shared_test_ids)}; full week 3 test rows: {len(test_ids)}")
    for row in rows:
        print(f"{row['model']:>19} {row['version']:>23} {row['imputer']:>6} "
              f"train={row['train_accuracy']:.4f} test={row['test_accuracy']:.4f} "
              f"F1={row['test_f1']:.4f} precision={row['test_precision']:.4f} "
              f"recall={row['test_recall']:.4f} FP={row['false_positives']} "
              f"FN={row['false_negatives']} n_train={row['train_rows']}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
