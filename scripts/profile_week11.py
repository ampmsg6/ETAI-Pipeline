"""Development-only anomaly/rearrest rates, including fold-held-out scores."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data import load_data
from src.preprocessing import (AnomalyScore, add_engineered_features, build_preprocessor,
                               clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target)


def development_data(config):
    raw = load_data(config["data"]["path"])
    clean = drop_duplicate_rows(clean_dataset(raw, config), config["diagnostics"].get("id_column"))
    clean = add_engineered_features(clean, config["preprocessing"].get("engineered_features", []))
    X, y, extras = split_features_target(clean, config)
    X_dev, X_test, y_dev, _, _, _ = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
    return X_dev, y_dev, X_test


def top_fraction(scores, fraction=0.05):
    """Exactly ceil(5% * n) rows; stable input-order tie breaking, without labels."""
    top = np.zeros(len(scores), dtype=bool)
    top[np.argsort(-np.asarray(scores), kind="stable")[:int(np.ceil(fraction * len(scores)))]] = True
    return top


def profile_anomalies(config, publish_dir):
    folder = Path(publish_dir)
    X, y, _ = development_data(config)
    cv = StratifiedKFold(config["cv"]["n_splits"], shuffle=config["cv"].get("shuffle", True),
                         random_state=config["cv"].get("random_state"))
    methods = ("isolation_forest", "lof")
    scores = {method: np.empty(len(X)) for method in methods}
    top = {method: np.zeros(len(X), dtype=bool) for method in methods}
    fold_number = np.zeros(len(X), dtype=int)
    rates, neighbourhoods, individual = [], [], []

    def summarise(method, scope, values, mask):
        for group, chosen in (("top_5_percent", mask), ("remaining", ~mask), ("overall", np.ones(len(X), dtype=bool))):
            rates.append({"anomaly": method, "scope": scope, "group": group,
                          "n": int(chosen.sum()), "rearrests": int(y.iloc[np.flatnonzero(chosen)].sum()),
                          "rearrest_rate": float(y.iloc[np.flatnonzero(chosen)].mean()),
                          "mean_anomaly_score": float(values[chosen].mean())})

    with threadpool_limits(limits=1):
        for fold, (train, val) in enumerate(cv.split(X, y), start=1):
            prep = build_preprocessor(config)
            Z = prep.fit_transform(X.iloc[train], y.iloc[train])
            new = prep.transform(X.iloc[val])
            largest = int(np.unique(Z.round(6), axis=0, return_counts=True)[1].max())
            neighbourhoods.append({"scope": "training_fold", "fold": fold, "training_rows": len(train),
                                   "largest_identical_group": largest, "n_neighbors": config["anomaly"]["n_neighbors"],
                                   "k_larger_than_group": config["anomaly"]["n_neighbors"] > largest})
            fold_number[val] = fold
            for method in methods:
                step = AnomalyScore(method, config["anomaly"]["n_neighbors"], config["anomaly"]["random_state"])
                step.fit(Z)
                values = step.transform(new)[:, -1]
                scores[method][val] = values
                top[method][val] = top_fraction(values)
        # This descriptive notebook-style fit never feeds a classifier or CV estimate.
        prep = build_preprocessor(config)
        Z = prep.fit_transform(X, y)
        largest = int(np.unique(Z.round(6), axis=0, return_counts=True)[1].max())
        neighbourhoods.append({"scope": "development_fit", "fold": 0, "training_rows": len(X),
                               "largest_identical_group": largest, "n_neighbors": config["anomaly"]["n_neighbors"],
                               "k_larger_than_group": config["anomaly"]["n_neighbors"] > largest})
        for method in methods:
            step = AnomalyScore(method, config["anomaly"]["n_neighbors"], config["anomaly"]["random_state"])
            fitted_scores = step.fit_transform(Z)[:, -1]
            summarise(method, "development_fit", fitted_scores, top_fraction(fitted_scores))
            summarise(method, "out_of_fold", scores[method], top[method])
            individual.append(pd.DataFrame({"row_index": X.index, "y_true": y.to_numpy(), "fold": fold_number,
                                            "anomaly": method, "oof_score": scores[method],
                                            "oof_top_5_percent": top[method], "development_fit_score": fitted_scores}))
    pd.DataFrame(rates).to_csv(folder / "anomaly_rates.csv", index=False)
    pd.DataFrame(neighbourhoods).to_csv(folder / "lof_neighbourhoods.csv", index=False)
    local = Path(config["output"]["results_dir"])
    local.mkdir(parents=True, exist_ok=True)
    pd.concat(individual, ignore_index=True).to_csv(local / "anomaly_scores.csv", index=False)
    print(pd.DataFrame(rates).to_string(index=False), flush=True)
