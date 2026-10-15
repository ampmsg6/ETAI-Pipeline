"""Checks for leakage boundaries, feature missingness and ensemble parameter paths."""
import contextlib
import io
import unittest

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from main import load_config
from src.evaluate import corrected_paired_test
from src.model import build_pipeline, default_model_config
from src.preprocessing import add_engineered_features
from src.tuning import nested_cross_validate, tune_pipeline


class RecordingTransformer(TransformerMixin, BaseEstimator):
    seen = []

    def fit(self, X, y=None):
        self.fitted_ = True
        type(self).seen.append(set(X.index))
        return self

    def transform(self, X):
        return X


class Week6Tests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def test_inner_preprocessing_never_sees_outer_validation(self):
        X = pd.DataFrame({"x": np.arange(36)}, index=np.arange(100, 136))
        y = pd.Series([0, 1] * 18, index=X.index)
        pipeline = Pipeline([
            ("prep", RecordingTransformer()),
            ("model", DecisionTreeClassifier(random_state=42)),
        ])
        outer = StratifiedKFold(3, shuffle=True, random_state=42)
        inner = StratifiedKFold(2, shuffle=True, random_state=42)
        space = {"model__max_depth": {"type": "int", "low": 2, "high": 2}}
        RecordingTransformer.seen.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            folds, oof = nested_cross_validate(pipeline, X, y, outer, inner, "accuracy", space, 2, 42)
        # Two trials * two inner fits + the chosen pipeline's outer refit.
        self.assertEqual(len(RecordingTransformer.seen), 15)
        for fold, (train, val) in enumerate(outer.split(X, y)):
            allowed = set(X.iloc[train].index)
            forbidden = set(X.iloc[val].index)
            observed = RecordingTransformer.seen[fold * 5:(fold + 1) * 5]
            for fit_rows in observed:
                self.assertTrue(fit_rows <= allowed)
                self.assertFalse(fit_rows & forbidden)
            self.assertEqual(observed[-1], allowed)
            expected = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X.iloc[train], y.iloc[train])
            np.testing.assert_array_equal(oof[val], expected.predict(X.iloc[val]))
        self.assertEqual(len(folds), 3)
        best, _ = tune_pipeline(pipeline, X, y, inner, "accuracy", space, 1, 42)
        with self.assertRaises(NotFittedError):
            check_is_fitted(best.named_steps["prep"])

    def test_features_preserve_rows_and_missing_components(self):
        df = pd.DataFrame({
            "age": [18, 19, np.nan, 30], "priors_count": [2, 4, 1, np.nan],
            "juv_fel_count": [1, np.nan, 1, 1], "juv_misd_count": [2, 2, 2, 2],
            "juv_other_count": [3, 3, 3, 3],
        }, index=[7, 3, 5, 9])
        original = df.copy(deep=True)
        out = add_engineered_features(df)
        pd.testing.assert_frame_equal(df, original)
        self.assertEqual(list(out.index), list(df.index))
        self.assertEqual(out.loc[7, "priors_per_adult_year"], 2)
        self.assertEqual(out.loc[3, "priors_per_adult_year"], 2)
        self.assertEqual(out.loc[7, "juvenile_per_adult_year"], 6)
        self.assertTrue(pd.isna(out.loc[3, "juvenile_per_adult_year"]))
        self.assertTrue(pd.isna(out.loc[5, "priors_per_adult_year"]))
        self.assertTrue(pd.isna(out.loc[9, "priors_age_interaction"]))

    def test_ensembles_default_and_nested_tuning(self):
        config = load_config()
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "age": rng.integers(18, 65, 80), "priors_count": rng.integers(0, 10, 80),
            "juv_fel_count": rng.integers(0, 3, 80), "juv_misd_count": rng.integers(0, 3, 80),
            "juv_other_count": rng.integers(0, 3, 80), "sex": ["Male", "Female"] * 40,
            "age_cat": ["Less than 25", "25 - 45"] * 40, "c_charge_degree": ["F", "M"] * 40,
            "priors_count_was_missing": np.zeros(80), "c_charge_degree_was_missing": np.zeros(80),
        })
        y = pd.Series([0, 1] * 40)
        cv = StratifiedKFold(2, shuffle=True, random_state=42)
        for name in ("random_forest", "oblique_random_forest", "adaboost"):
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()):
                model_config = default_model_config(name)
                model_config["params"].update(n_estimators=4)
                if name != "adaboost":
                    model_config["params"]["n_jobs"] = 1
                pipeline = build_pipeline(config, model_config)
                pipeline.fit(X, y)
                self.assertEqual(len(pipeline.predict(X)), len(X))
                if name == "adaboost":
                    space = {"model__estimator__max_depth": {"type": "int", "low": 1, "high": 2}}
                else:
                    space = config["tuning"]["search_spaces"][name]
                folds, oof = nested_cross_validate(pipeline, X, y, cv, cv, "accuracy", space, 1, 42)
                self.assertEqual(len(oof), len(X))
                self.assertTrue(folds["validation"].between(0, 1).all())
                pipeline.set_params(**{key: folds.iloc[0][key] for key in space})

    def test_corrected_test_known_values_and_pairing(self):
        a = [0.674, 0.658, 0.701, 0.688, 0.691]
        b = [0.675, 0.658, 0.693, 0.684, 0.696]
        result = corrected_paired_test(a, b, 1 / 4)
        self.assertAlmostEqual(result["mean_diff"], 0.0012)
        self.assertGreater(result["p_value"], 0.05)
        self.assertEqual(corrected_paired_test(a, a, 1 / 4)["p_value"], 1.0)
        with self.assertRaises(ValueError):
            corrected_paired_test(a, b[:-1], 1 / 4)


if __name__ == "__main__":
    unittest.main()
