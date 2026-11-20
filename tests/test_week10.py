"""Training-only imputation, saved pipelines and row-preserving deployment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from main import load_config
from src.artifacts import load_model, save_model
from src.data import load_data
from src.model import build_pipeline, default_model_config
from src.predict import predict
from src.preprocessing import build_preprocessor, clean_dataset, split_features_target


class Week10Tests(unittest.TestCase):
    def numeric_config(self, strategy):
        config = load_config()
        config["preprocessing"].update(numeric_features=["a", "b"], categorical_features=[],
            imputation={"numeric_strategy": strategy, "max_iter": 3, "n_estimators": 5})
        return config

    def test_all_imputers_fit_only_training_values_and_preserve_empty_columns(self):
        X = pd.DataFrame({"a": [1., 2., 3., np.nan, 30., 60., 90., np.nan],
                          "b": [2., np.nan, 6., 8., 10., 12., np.nan, 16.]})
        for strategy in ("median", "mean", "knn", "iterative", "random_forest"):
            with self.subTest(strategy=strategy):
                prep = build_preprocessor(self.numeric_config(strategy))
                for train, val in KFold(2).split(X):
                    fitted = clone(prep).fit(X.iloc[train])
                    numeric = fitted.named_transformers_["numeric"]
                    np.testing.assert_allclose(numeric["scale"].center_, X.iloc[train].median())
                    if strategy == "median":
                        expected = np.nanmedian(numeric["scale"].transform(X.iloc[train]), axis=0)
                        np.testing.assert_allclose(numeric["impute"].statistics_, expected)
                    self.assertTrue(np.isfinite(fitted.transform(X.iloc[val])).all())
                empty = X.copy()
                empty["b"] = np.nan
                fitted = clone(prep).fit(empty)
                transformed = fitted.transform(X)
                self.assertEqual(transformed.shape, X.shape)
                self.assertTrue(np.isfinite(transformed).all())

    def test_knn_uses_scaled_distances(self):
        config = self.numeric_config("knn")
        config["preprocessing"]["numeric_features"] = ["a", "b", "c"]
        config["preprocessing"]["imputation"]["n_neighbors"] = 1
        train = pd.DataFrame({"a": [0., 1000., 0., 1000.], "b": [0., 1., 1., 0.],
                              "c": [10., 20., 30., 40.]})
        query = pd.DataFrame({"a": [100.], "b": [0.9], "c": [np.nan]})
        numeric = build_preprocessor(config).fit(train).named_transformers_["numeric"]
        filled = numeric["scale"].inverse_transform(numeric.transform(query))
        self.assertAlmostEqual(filled[0, 2], 30.)
        self.assertEqual(list(numeric.named_steps), ["scale", "impute"])

    def test_legacy_imputer_override_and_unknown_strategy(self):
        config = self.numeric_config("median")
        config["preprocessing"]["numeric_imputer"] = "knn"
        prep = build_preprocessor(config)
        self.assertEqual(type(prep.transformers[0][1]["impute"]).__name__, "KNNImputer")
        config["preprocessing"].pop("numeric_imputer")
        config["preprocessing"]["imputation"]["numeric_strategy"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Unknown numeric_strategy"):
            build_preprocessor(config)

    def test_saved_pipeline_predicts_dirty_compas_without_refitting(self):
        config = load_config()
        X, y, _ = split_features_target(clean_dataset(load_data(config["data"]["path"]), config), config)
        pipeline = build_pipeline(config, default_model_config("decision_tree"), selection_config={})
        pipeline.fit(X.iloc[:1200], y.iloc[:1200])
        raw = load_data("data/compas_new_people.csv")
        # Also exercise identical IDs, identical facts and a non-default input index.
        raw = pd.concat([raw.iloc[::-1], raw.iloc[[14]]])
        expected = predict(raw, pipeline, config)
        with tempfile.TemporaryDirectory() as folder:
            config["output"]["model_path"] = str(Path(folder) / "model.joblib")
            path = save_model(pipeline, config, "test", 1200, 0)
            loaded, saved_config = load_model(path)
            np.testing.assert_array_equal(loaded.predict_proba(X.iloc[:20]), pipeline.predict_proba(X.iloc[:20]))
            before = loaded["prep"].named_transformers_["numeric"]["impute"].statistics_.copy()
            with patch.object(loaded, "fit", side_effect=AssertionError("Inference must not fit")):
                result = predict(raw, loaded, saved_config)
            pd.testing.assert_frame_equal(result, expected)
            self.assertEqual(len(result), 21)
            np.testing.assert_array_equal(result["id"], raw["id"])
            np.testing.assert_array_equal(before, loaded["prep"].named_transformers_["numeric"]["impute"].statistics_)
            self.assertIsNone(loaded.memory)
            # A different environment is rejected before any pickle is loaded.
            meta_path = Path(path).with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["versions"]["scikit-learn"] = "0.0"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with patch("src.artifacts.joblib.load", side_effect=AssertionError("Must not load")):
                with self.assertRaisesRegex(ValueError, "library versions differ"):
                    load_model(path)

    def test_missing_required_columns_fail_clearly_before_prediction(self):
        config = load_config()
        raw = load_data("data/compas_new_people.csv").drop(columns=["age", "priors_count"])
        with self.assertRaisesRegex(ValueError, "missing required columns: age, priors_count"):
            predict(raw, None, config)


if __name__ == "__main__":
    unittest.main()
