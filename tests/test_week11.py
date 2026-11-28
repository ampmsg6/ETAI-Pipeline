"""Fold-isolated anomaly scores and saved-model novelty prediction."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import LocalOutlierFactor

from main import load_config
from src.artifacts import load_model, save_model
from src.data import load_data
from src.model import build_pipeline, default_model_config
from src.predict import predict
from src.preprocessing import AnomalyScore, clean_dataset, split_features_target


class Week11Tests(unittest.TestCase):
    def test_scores_match_detectors_and_lof_training_excludes_self(self):
        X = np.array([[0.], [0.], [0.1], [0.2], [1.], [2.], [10.], [15.]])
        for method in ("isolation_forest", "lof"):
            with self.subTest(method=method):
                step = AnomalyScore(method, n_neighbors=3, random_state=42)
                transformed = step.fit_transform(X, np.arange(len(X)))
                reference = (IsolationForest(random_state=42).fit(X) if method == "isolation_forest"
                             else LocalOutlierFactor(n_neighbors=3, novelty=True).fit(X))
                expected = (-reference.score_samples(X) if method == "isolation_forest"
                            else -reference.negative_outlier_factor_)
                np.testing.assert_array_equal(transformed[:, :-1], X)
                np.testing.assert_allclose(transformed[:, -1], expected)
                query = np.array([[100.], [0.], [100.]])
                np.testing.assert_allclose(step.transform(query)[:, -1], -reference.score_samples(query))
                self.assertEqual(step.transform(query).shape, (3, 2))
                self.assertEqual(step.transform(query)[0, -1], step.transform(query)[2, -1])
                if method == "lof":
                    self.assertFalse(np.allclose(expected, -reference.score_samples(X)))

    def test_fitted_contract_clone_names_and_invalid_method(self):
        step = AnomalyScore("lof", 3, 42)
        with self.assertRaises(NotFittedError):
            step.transform([[0., 1.]])
        X = pd.DataFrame({"first": np.arange(8.), "second": np.arange(8.) ** 2})
        step.fit(X)
        np.testing.assert_array_equal(step.get_feature_names_out(), ["first", "second", "anomaly_score"])
        with self.assertRaisesRegex(ValueError, "input_features"):
            step.get_feature_names_out(["one"])
        with self.assertRaises(ValueError):
            step.transform([[1.]])
        self.assertEqual(clone(step).get_params(), step.get_params())
        with self.assertRaisesRegex(ValueError, "Unknown anomaly method"):
            AnomalyScore("invalid").fit(X)

    def test_step_order_full_config_override_and_legacy_disabled(self):
        config = load_config()
        config["anomaly"]["method"] = "lof"
        self.assertEqual(list(build_pipeline(config).named_steps), ["prep", "anomaly", "select", "model"])
        self.assertEqual(list(build_pipeline(config, selection_config={}).named_steps), ["prep", "anomaly", "model"])
        self.assertNotIn("anomaly", build_pipeline(config, anomaly_config={}).named_steps)
        legacy = deepcopy(config)
        legacy.pop("anomaly")
        self.assertNotIn("anomaly", build_pipeline(legacy).named_steps)

    def test_cv_detector_sees_only_its_training_fold(self):
        config = load_config()
        config["preprocessing"].update(numeric_features=["a", "b"], categorical_features=[],
                                      imputation={"numeric_strategy": "median"})
        config["anomaly"].update(method="lof", n_neighbors=3)
        X = pd.DataFrame({"a": np.arange(30.), "b": np.arange(30.) ** 2})
        y = np.tile([0, 1], 15)
        cv = KFold(3, shuffle=True, random_state=42)
        pipe = build_pipeline(config, default_model_config("decision_tree"), selection_config={})
        result = cross_validate(pipe, X, y, cv=cv, return_estimator=True, error_score="raise")
        for estimator, (train, val) in zip(result["estimator"], cv.split(X)):
            expected = estimator["prep"].transform(X.iloc[train])
            np.testing.assert_allclose(estimator["anomaly"].detector_._fit_X, expected)
            self.assertEqual(len(estimator["anomaly"].detector_._fit_X), len(train))
            before = estimator["anomaly"].detector_._fit_X.copy()
            self.assertEqual(len(estimator.predict(X.iloc[val])), len(val))
            np.testing.assert_array_equal(before, estimator["anomaly"].detector_._fit_X)

    def test_saved_detectors_predict_every_new_row_without_refitting(self):
        config = load_config()
        X, y, _ = split_features_target(clean_dataset(load_data(config["data"]["path"]), config), config)
        raw = load_data("data/compas_new_people.csv")
        raw = pd.concat([raw.iloc[::-1], raw.iloc[[14]]])
        for method in ("isolation_forest", "lof"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                config["anomaly"]["method"] = method
                config["output"]["model_path"] = str(Path(folder) / "model.joblib")
                pipe = build_pipeline(config, default_model_config("decision_tree"), selection_config={})
                pipe.fit(X.iloc[:600], y.iloc[:600])
                expected = predict(raw, pipe, config)
                model_path = save_model(pipe, config, "test_week11", 600, 0)
                loaded, saved = load_model(model_path)
                with patch.object(loaded["anomaly"], "fit", side_effect=AssertionError("No inference fit")):
                    result = predict(raw, loaded, saved)
                pd.testing.assert_frame_equal(result, expected)
                self.assertEqual(len(result), 21)
                np.testing.assert_array_equal(result["id"], raw["id"])
                self.assertTrue(np.isfinite(result["probability"]).all())


if __name__ == "__main__":
    unittest.main()
