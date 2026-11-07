"""Leakage, caching and reporting checks for Week 8's learned selection."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from main import load_config, run_pipeline
from src.evaluate import build_scorers
from src.model import VoteSelector, build_pipeline, default_model_config
from src.results import comparison_key
from src.tuning import nested_cross_validate


class NamedIdentity(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self):
        return self.feature_names_in_


class RecordingSelector(VoteSelector):
    seen = []

    def fit(self, X, y):
        type(self).seen.append(set(X.index))
        return super().fit(X, y)


class Week8Tests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        values, labels = make_classification(n_samples=120, n_features=5, n_informative=3,
                                              n_redundant=0, random_state=42)
        self.X = pd.DataFrame(values, columns=list("abcde"), index=np.arange(1000, 1120))
        self.y = pd.Series(labels, index=self.X.index)

    def test_selector_names_clone_and_fitted_state(self):
        selector = VoteSelector(min_votes=1, random_state=42)
        with self.assertRaises(NotFittedError):
            selector.get_support()
        selector.fit(self.X, self.y)
        np.testing.assert_array_equal(selector.get_feature_names_out(),
                                      self.X.columns[selector.get_support()])
        self.assertEqual(selector.transform(self.X).shape[1], selector.get_support().sum())
        self.assertTrue(((selector.votes_ >= 0) & (selector.votes_ <= 3)).all())
        other = clone(selector).fit(self.X, self.y)
        np.testing.assert_array_equal(selector.votes_, other.votes_)
        with self.assertRaisesRegex(ValueError, "min_votes"):
            VoteSelector(min_votes=4).fit(self.X, self.y)

    def test_cached_selection_is_refitted_inside_nested_training_folds(self):
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        inner = StratifiedKFold(2, shuffle=True, random_state=42)
        RecordingSelector.seen.clear()
        with tempfile.TemporaryDirectory() as folder:
            pipeline = Pipeline([
                ("prep", NamedIdentity()),
                ("select", RecordingSelector(min_votes=1, random_state=42)),
                ("model", DecisionTreeClassifier(random_state=42)),
            ], memory=folder)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, proba = nested_cross_validate(
                    pipeline, self.X, self.y, cv, inner, build_scorers(["roc_auc"]), "roc_auc",
                    {"model__max_depth": {"type": "int", "low": 2, "high": 2},
                     "select__min_votes": {"type": "int", "low": 1, "high": 1}}, 2, 42,
                )
        # The repeated trial uses cached prep/selection: two inner fits + one
        # outer refit per outer fold, rather than five selection fits per fold.
        self.assertEqual(len(RecordingSelector.seen), 9)
        for fold, (train, val) in enumerate(cv.split(self.X, self.y)):
            allowed, forbidden = set(self.X.iloc[train].index), set(self.X.iloc[val].index)
            for seen in RecordingSelector.seen[fold * 3:(fold + 1) * 3]:
                self.assertTrue(seen <= allowed)
                self.assertFalse(seen & forbidden)
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue((scores["select__min_votes"] == 1).all())
        self.assertTrue((scores["n_features"] > 0).all())

    def test_xgboost_no_tuning_records_features_and_selection_setting(self):
        config = load_config()
        config["tuning"]["enabled"] = False
        config["cv"]["n_splits"] = 2
        config["model"] = default_model_config("xgboost")
        config["model"]["params"].update(n_estimators=5, n_jobs=1)
        config["feature_selection"].update(enabled=True, min_votes=2)
        with tempfile.TemporaryDirectory() as folder:
            config["output"] = {"results_dir": folder, "scoreboard": str(Path(folder) / "runs.csv")}
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_pipeline(config)
            report = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertIn("Feature selection: VoteSelector", report)
            self.assertIn("Features kept:", report)
            self.assertIn("1443 rows, not evaluated", report)
            self.assertEqual(result["score_row"]["feature_selection"], "min_votes 2")
            self.assertEqual(result["model"].memory, None)
            features = pd.read_csv(Path(result["report_path"]).with_suffix("").as_posix() + "_features.csv")
            self.assertEqual(result["kept_features"], features.loc[features["kept"], "feature"].tolist())
        # A no-selection dummy remains a true no-information floor.
        floor = build_pipeline(config, default_model_config("dummy"), selection_config={})
        self.assertNotIn("select", floor.named_steps)
        other = deepcopy(config)
        other["feature_selection"]["enabled"] = False
        self.assertEqual(comparison_key(other), comparison_key(config))


if __name__ == "__main__":
    unittest.main()
