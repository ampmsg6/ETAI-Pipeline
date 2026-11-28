"""Independently verify the published Week 11 metrics and real prediction CLI."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from main import load_config
from scripts.profile_week11 import development_data, top_fraction
from src.artifacts import load_model
from src.data import load_data
from src.predict import predict
from src.model import build_pipeline


def main():
    folder = Path("reports/week11")
    board = pd.read_csv(folder / "scoreboard.csv", dtype={"run": str})
    config = load_config()
    X, y, reserved = development_data(config)
    cv = StratifiedKFold(config["cv"]["n_splits"], shuffle=True, random_state=config["cv"]["random_state"])
    split = list(cv.split(X, y))
    new = load_data("data/compas_new_people.csv")
    prediction_checks = {}
    fold_features = []
    for row in board.to_dict("records"):
        prefix = Path(row["report_path"]).with_suffix("")
        oof = pd.read_csv(f"{prefix}_oof.csv")
        np.testing.assert_array_equal(oof["row_index"], X.index)
        np.testing.assert_array_equal(oof["y_true"], y)
        assert not set(oof["row_index"]) & set(reserved.index)
        folds = pd.read_csv(f"{prefix}_default.csv")
        for i, (_, val) in enumerate(split):
            truth = y.iloc[val]
            probability = oof.iloc[val]["oof_probability"]
            labels = probability.ge(0.5).astype(int)
            values = {
                "accuracy": metrics.accuracy_score(truth, labels),
                "balanced_accuracy": metrics.balanced_accuracy_score(truth, labels),
                "precision": metrics.precision_score(truth, labels),
                "recall": metrics.recall_score(truth, labels),
                "f1": metrics.f1_score(truth, labels),
                "fbeta_2": metrics.fbeta_score(truth, labels, beta=2),
                "roc_auc": metrics.roc_auc_score(truth, probability),
                "average_precision": metrics.average_precision_score(truth, probability),
            }
            for name, value in values.items():
                np.testing.assert_allclose(folds.iloc[i][name], value, atol=1e-14)
        pipe, saved = load_model(row["model_path"])
        method = row["recipe"]
        assert saved["anomaly"]["method"] == method
        assert ("anomaly" in pipe.named_steps) == (method != "none")
        features = pd.read_csv(f"{prefix}_features.csv")
        assert ("anomaly_score" in features["feature"].values) == (method != "none")
        if method != "none":
            assert pipe["anomaly"].n_features_in_ == len(features) - 1
        assert pipe.memory is None
        # Refit each published recipe independently and verify its held-out probabilities.
        with threadpool_limits(limits=1):
            for fold, (train, val) in enumerate(split, start=1):
                fitted = build_pipeline(saved).fit(X.iloc[train], y.iloc[train])
                probability = fitted.predict_proba(X.iloc[val])[:, 1]
                np.testing.assert_array_equal(probability, oof.iloc[val]["oof_probability"].to_numpy(dtype=np.float32))
                names = fitted["prep"].get_feature_names_out()
                if "anomaly" in fitted.named_steps:
                    names = fitted["anomaly"].get_feature_names_out(names)
                selector = fitted["select"]
                for name, vote, kept in zip(names, selector.votes_, selector.get_support()):
                    fold_features.append({"anomaly": method, "fold": fold, "feature": name,
                                          "votes": int(vote), "kept": bool(kept)})
        before = hashlib.sha256(Path(row["model_path"]).read_bytes()).hexdigest()
        output = Path("predictions/week11") / f"{method}.csv"
        completed = subprocess.run([sys.executable, "predict.py", "data/compas_new_people.csv", str(output),
                                    "--model", row["model_path"]], check=True, capture_output=True, text=True)
        print(completed.stdout, end="")
        result = pd.read_csv(output)
        expected = predict(new, pipe, saved)
        assert len(result) == len(new) == 20
        np.testing.assert_array_equal(result["id"], new["id"])
        np.testing.assert_array_equal(result["probability"].to_numpy(dtype=np.float32), expected["probability"])
        np.testing.assert_array_equal(result["prediction"], expected["prediction"])
        assert result.loc[14, "probability"] == result.loc[15, "probability"]
        assert hashlib.sha256(Path(row["model_path"]).read_bytes()).hexdigest() == before
        prediction_checks[method] = {"rows": len(result), "ids_and_order_preserved": True, "no_refit": True}
    pd.DataFrame(fold_features).to_csv(folder / "fold_features.csv", index=False)
    indexed = board.set_index("recipe")
    for row in pd.read_csv(folder / "anomaly_comparisons.csv").to_dict("records"):
        columns = [f"fold_{i}" for i in range(1, 6)]
        difference = (indexed.loc[row["anomaly_a"], columns].astype(float).to_numpy()
                      - indexed.loc[row["anomaly_b"], columns].astype(float).to_numpy())
        statistic = difference.mean() / np.sqrt((1 / 5 + 1 / 4) * difference.var(ddof=1))
        np.testing.assert_allclose(row["mean_diff"], difference.mean(), atol=1e-14)
        np.testing.assert_allclose(row["t"], statistic, atol=1e-12)
        np.testing.assert_allclose(row["p_value"], 2 * student_t.sf(abs(statistic), 4), atol=1e-12)
    local = Path("results/week11/anomaly_scores.csv")
    rates = pd.read_csv(folder / "anomaly_rates.csv")
    individual = pd.read_csv(local)
    for method, part in individual.groupby("anomaly", sort=False):
        np.testing.assert_array_equal(part["row_index"], X.index)
        np.testing.assert_array_equal(part["y_true"], y)
        mask = np.zeros(len(X), dtype=bool)
        for _, val in split:
            mask[val] = top_fraction(part.iloc[val]["oof_score"])
        np.testing.assert_array_equal(mask, part["oof_top_5_percent"])
        for scope, top in (("out_of_fold", mask), ("development_fit", top_fraction(part["development_fit_score"]))):
            for group, chosen in (("top_5_percent", top), ("remaining", ~top), ("overall", np.ones(len(X), dtype=bool))):
                row = rates[(rates["anomaly"] == method) & (rates["scope"] == scope) & (rates["group"] == group)].iloc[0]
                assert row["n"] == int(chosen.sum())
                assert row["rearrests"] == int(y.iloc[np.flatnonzero(chosen)].sum())
                np.testing.assert_allclose(row["rearrest_rate"], y.iloc[np.flatnonzero(chosen)].mean(), atol=1e-14)
    assert pd.read_csv(folder / "lof_neighbourhoods.csv")["k_larger_than_group"].all()
    historical = pd.read_csv("reports/week10/folds.csv")
    historical = historical[historical["imputer"] == "median"]
    current = pd.read_csv(folder / "folds.csv")
    current = current[current["anomaly"] == "none"]
    np.testing.assert_allclose(current[config["cv"]["metrics"]], historical[config["cv"]["metrics"]], atol=1e-14)
    default_model = Path(config["output"]["model_path"])
    default_hash = hashlib.sha256(default_model.read_bytes()).hexdigest()
    completed = subprocess.run([sys.executable, "predict.py", "data/compas_new_people.csv", "predictions.csv"],
                                check=True, capture_output=True, text=True)
    print(completed.stdout, end="")
    assert len(pd.read_csv("predictions.csv")) == 20
    assert hashlib.sha256(default_model.read_bytes()).hexdigest() == default_hash
    validation = {
        "verified_date": "2026-11-28", "development_rows": len(X), "reserved_rows": len(reserved),
        "reserved_test_evaluated": False, "metrics_recomputed": 3 * 5 * 8,
        "independent_fold_refits_matching_oof": 15, "corrected_comparisons_recomputed": 3, "anomaly_rate_groups_recomputed": len(rates),
        "lof_k_checked_on_all_training_folds": True, "predictions": prediction_checks,
        "default_cli_prediction_rows": 20, "week10_baseline_metrics_unchanged": True,
    }
    (folder / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
