"""Adult portability, cleaning and row-preserving unlabelled prediction contracts."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from main import load_config
from src.data import load_data
from src.evaluate import fairness_report
from src.model import build_pipeline, default_model_config
from src.predict import predict_new_data
from src.preprocessing import clean_dataset, split_features_target, add_engineered_features


class Week9Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config_adult.yaml")
        self.raw = load_data(self.config["data"]["path"])

    def test_adult_cleaning_matches_the_course_card_and_keeps_valid_extremes(self):
        clean = clean_dataset(self.raw, self.config)
        self.assertEqual(len(clean), 6000)
        np.testing.assert_array_equal(clean["id"], self.raw["id"])
        for column, expected in (("workclass", 379), ("occupation", 381), ("native_country", 86)):
            self.assertEqual(clean[column].isna().sum(), expected)
            self.assertFalse(clean[column].eq("?").any())
        self.assertEqual(clean["age"].eq(17).sum(), 74)
        self.assertEqual(clean["capital_gain"].eq(99999).sum(), 29)
        self.assertNotIn("education", clean)
        self.assertNotIn("fnlwgt", clean)

    def test_every_undocumented_feature_reaches_preprocessing_without_protected_fields(self):
        X, y, extras = split_features_target(clean_dataset(self.raw, self.config), self.config)
        self.assertTrue({f"v{i:02d}" for i in range(1, 26)} <= set(X.columns))
        self.assertEqual(len(X.columns), 37)
        self.assertFalse({"id", "sex", "race", "education", "fnlwgt", "income_over_50k"} & set(X.columns))
        self.assertEqual(extras.columns.tolist(), ["sex"])
        self.assertEqual(y.eq(1).sum(), 1436)
        self.assertEqual(X["workclass_was_missing"].sum(), 379)
        self.assertEqual(X["occupation_was_missing"].sum(), 381)
        pipeline = build_pipeline(self.config, default_model_config("decision_tree"), selection_config={})
        pipeline.fit(X.iloc[:1200], y.iloc[:1200])
        names = pipeline.named_steps["prep"].get_feature_names_out()
        for name in (f"v{i:02d}" for i in range(1, 26)):
            self.assertTrue(any(feature.endswith("__" + name) for feature in names))

    def test_predictions_keep_ids_order_duplicates_and_handle_unseen_categories(self):
        X, y, _ = split_features_target(clean_dataset(self.raw, self.config), self.config)
        pipeline = build_pipeline(self.config, default_model_config("decision_tree"), selection_config={})
        pipeline.fit(X.iloc[:1200], y.iloc[:1200])
        new = load_data("data/adult_new.csv")
        new.loc[0, "native_country"] = "Previously-unseen-country"
        new = pd.concat([new.iloc[::-1], new.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            config = deepcopy(self.config)
            config["prediction"].update(input_path=str(Path(folder) / "input.csv"),
                                         output_path=str(Path(folder) / "output.csv"))
            new.to_csv(config["prediction"]["input_path"], index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                path = predict_new_data(pipeline, config)
            predictions = pd.read_csv(path)
            self.assertEqual(len(predictions), 2001)
            np.testing.assert_array_equal(predictions["id"], new["id"])
            self.assertTrue(predictions["probability"].between(0, 1).all())
            np.testing.assert_array_equal(predictions["prediction"], predictions["probability"].ge(0.5).astype(int))

    def test_audit_needs_no_external_reference_and_empty_engineering_needs_no_age(self):
        report = fairness_report(pd.Series([0, 0, 1, 1]), np.array([0.7, 0.2, 0.6, 0.8]),
                                  pd.DataFrame({"sex": ["Female", "Female", "Male", None]}),
                                  sensitive_attr="sex")
        self.assertIn("Female               FPR = 0.50  (n=2)", report)
        self.assertNotIn("COMPAS", report)
        self.assertNotIn("reoffend", report)
        frame = pd.DataFrame({"unrelated_feature": [1, 2]})
        pd.testing.assert_frame_equal(frame, add_engineered_features(frame, []))


if __name__ == "__main__":
    unittest.main()
