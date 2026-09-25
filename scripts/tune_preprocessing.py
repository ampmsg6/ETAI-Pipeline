"""Rank notebook encoder/scaler choices on training folds only."""

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from main import load_config
from src.data import load_data
from src.preprocessing import build_preprocessor, clean_dataset, split_features_target, split_train_test


def main():
    config = load_config()
    df = clean_dataset(load_data(config["data"]["path"]), config)
    X, y, extras = split_features_target(df, config)
    X_train, _, y_train, _, _, _ = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"], random_state=config["split"]["random_state"],
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for encoder in ["onehot", "ordinal", "count", "target"]:
        for scaler in ["none", "standard", "minmax", "robust"]:
            candidate = deepcopy(config)
            candidate["preprocessing"].update({"encoder": encoder, "scaler": scaler})
            pipe = Pipeline([
                ("preprocessing", build_preprocessor(candidate)),
                ("model", LogisticRegression(max_iter=1000)),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            rows.append({"encoder": encoder, "scaler": scaler,
                         "mean_accuracy": float(np.mean(scores)), "std_accuracy": float(np.std(scores))})
    rows.sort(key=lambda row: row["mean_accuracy"], reverse=True)
    path = Path("results/week3_tuning.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:5]:
        print(f"{row['encoder']:>7} {row['scaler']:>8}: {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}")
    print(f"Saved all 16 combinations to {path}")


if __name__ == "__main__":
    main()
