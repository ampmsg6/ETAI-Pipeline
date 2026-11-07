"""Regression checks for the Week 7 probability, metric and scoreboard contracts."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from main import load_config, run_pipeline
from src.evaluate import (
    build_scorers, cross_validate_pipeline, fairness_report, oof_classification_report,
)
from src.model import build_pipeline, default_model_config
from src.results import comparison_key, save_scores
from src.tuning import nested_cross_validate


class Week7Tests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        values, labels = make_classification(n_samples=120, n_features=5, random_state=42)
        self.X = pd.DataFrame(values, index=np.arange(1000, 1120)[::-1])
        self.y = pd.Series(labels, index=self.X.index)
        self.cv = StratifiedKFold(3, shuffle=True, random_state=42)
        self.model = DecisionTreeClassifier(max_depth=2, random_state=42)
        self.metrics = load_config()["cv"]["metrics"]

    def test_oof_probabilities_and_every_metric_match_independent_fold_fits(self):
        scorers = build_scorers(self.metrics)
        folds, probabilities = cross_validate_pipeline(
            self.model, self.X, self.y, self.cv, scorers, "roc_auc",
        )
        self.assertEqual(len(probabilities), len(self.X))
        self.assertTrue(((probabilities > 0) & (probabilities < 1)).any())
        for i, (train, val) in enumerate(self.cv.split(self.X, self.y)):
            fitted = clone(self.model).fit(self.X.iloc[train], self.y.iloc[train])
            np.testing.assert_allclose(probabilities[val], fitted.predict_proba(self.X.iloc[val])[:, 1])
            for name, scorer in scorers.items():
                self.assertAlmostEqual(folds.iloc[i][name], scorer(fitted, self.X.iloc[val], self.y.iloc[val]))
            self.assertAlmostEqual(folds.iloc[i]["validation"], roc_auc_score(self.y.iloc[val], probabilities[val]))

    def test_nested_cv_accepts_custom_fbeta_and_returns_probabilities(self):
        scorers = build_scorers(["accuracy", "fbeta_2", "roc_auc"])
        with contextlib.redirect_stdout(io.StringIO()):
            folds, probabilities = nested_cross_validate(
                self.model, self.X, self.y, self.cv, self.cv, scorers, "fbeta_2",
                {"max_depth": {"type": "int", "low": 2, "high": 2}}, 1, 42,
            )
        for i, (_, val) in enumerate(self.cv.split(self.X, self.y)):
            self.assertAlmostEqual(folds.iloc[i]["validation"], fbeta_score(
                self.y.iloc[val], probabilities[val] > 0.5, beta=2,
            ))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_new_models_fit_inside_the_existing_preprocessing_pipeline(self):
        config = load_config()
        config["feature_selection"]["enabled"] = False
        X = pd.DataFrame({
            "age": np.arange(120) % 45 + 18, "priors_count": np.arange(120) % 8,
            "juv_fel_count": np.arange(120) % 2, "juv_misd_count": np.zeros(120),
            "juv_other_count": np.zeros(120), "sex": ["Male", "Female"] * 60,
            "age_cat": ["Less than 25", "25 - 45", "Greater than 45"] * 40,
            "c_charge_degree": ["F", "M"] * 60,
            "priors_count_was_missing": np.zeros(120), "c_charge_degree_was_missing": np.zeros(120),
        })
        y = pd.Series([0, 1, 1, 0] * 30)
        for name in ("gradient_boosting", "hist_gradient_boosting", "naive_bayes"):
            with self.subTest(model=name):
                pipeline = build_pipeline(config, default_model_config(name))
                pipeline.fit(X, y)
                probabilities = pipeline.predict_proba(X)
                self.assertEqual(probabilities.shape, (120, 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                pipeline.set_params(**{key: spec.get("low", spec.get("choices", [None])[0])
                                       for key, spec in config["tuning"]["search_spaces"][name].items()})

    def test_reports_use_threshold_and_do_not_treat_unknown_compas_as_high_risk(self):
        y = pd.Series([0, 0, 1, 1])
        extras = pd.DataFrame({"race": ["A", "A", None, "B"], "score_text": ["Low", None, "High", "Medium"]})
        probabilities = np.array([0.6, 0.2, 0.8, 0.7])
        report = fairness_report(y, probabilities, extras)
        self.assertIn("A                    FPR = 0.50  (n=2)", report)
        self.assertIn("A                    FPR = 0.00  (n=1)", report)
        self.assertIn("predicted 1", oof_classification_report(y, probabilities, threshold=0.75))
        self.assertNotEqual(report, fairness_report(y, probabilities, extras, threshold=0.75))

    def test_scoreboard_append_and_experiment_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "runs.csv")
            save_scores(path, {"run": "0001", "mean": 0.5})
            rows = save_scores(path, {"run": "0002", "mean": 0.7})
            self.assertEqual(rows["run"].tolist(), ["0001", "0002"])
        config = load_config()
        other = deepcopy(config)
        other["model"] = default_model_config("naive_bayes")
        other["cv"]["n_jobs"] = 5
        self.assertEqual(comparison_key(config), comparison_key(other))
        other["cv"]["random_state"] = 123
        self.assertNotEqual(comparison_key(config), comparison_key(other))

    def test_gradient_boosting_no_tuning_writes_complete_week7_artifacts(self):
        config = load_config()
        config["model"] = default_model_config("gradient_boosting")
        config["feature_selection"]["enabled"] = False
        config["tuning"]["enabled"] = False
        with tempfile.TemporaryDirectory() as folder:
            config["output"] = {"results_dir": folder, "scoreboard": str(Path(folder) / "runs.csv")}
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_pipeline(config)
            report = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertEqual(set(self.metrics) - set(result["default_scores"].columns), set())
            self.assertEqual(len(result["oof_probabilities"]), 5771)
            self.assertIn("1443 rows, not evaluated", report)
            self.assertIn("predicted 0", report)
            self.assertIn("False positive rate", report)
            self.assertEqual(len(pd.read_csv(config["output"]["scoreboard"])), 1)


if __name__ == "__main__":
    unittest.main()
