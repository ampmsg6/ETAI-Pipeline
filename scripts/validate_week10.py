"""Independently verify Week 10 folds, corrected tests and the prediction CLI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from joblib import parallel_config
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from main import load_config, run_pipeline
from src.artifacts import load_model
from src.data import load_data
from src.predict import predict
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test


def main():
    folder = Path("reports/week10")
    historical_predictions = Path("predictions/adult_new_predictions.csv")
    historical_hash = hashlib.sha256(historical_predictions.read_bytes()).hexdigest()
    board = pd.read_csv(folder / "scoreboard.csv", dtype={"run": str})
    config = load_config()
    raw = load_data(config["data"]["path"])
    # The supplied Week 10 training CSV differs only in file formatting.
    course_raw = load_data("../../aula 10/data/compas_two_year_recidivism.csv")
    pd.testing.assert_frame_equal(raw, course_raw)
    clean = drop_duplicate_rows(clean_dataset(raw, config), config["diagnostics"]["id_column"])
    X, y, extras = split_features_target(clean, config)
    X_dev, X_test, y_dev, _, _, _ = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
    cv = StratifiedKFold(config["cv"]["n_splits"], shuffle=True, random_state=config["cv"]["random_state"])
    fold_indices = list(cv.split(X_dev, y_dev))
    for row in board.to_dict("records"):
        prefix = Path(row["report_path"]).with_suffix("")
        oof = pd.read_csv(f"{prefix}_oof.csv")
        np.testing.assert_array_equal(oof["row_index"], X_dev.index)
        np.testing.assert_array_equal(oof["y_true"], y_dev)
        assert not set(oof["row_index"]) & set(X_test.index)
        folds = pd.read_csv(f"{prefix}_default.csv")
        for i, (_, val) in enumerate(fold_indices):
            truth = y_dev.iloc[val]
            probability = oof.iloc[val]["oof_probability"]
            labels = probability.ge(0.5).astype(int)
            recomputed = {
                "accuracy": metrics.accuracy_score(truth, labels),
                "balanced_accuracy": metrics.balanced_accuracy_score(truth, labels),
                "precision": metrics.precision_score(truth, labels),
                "recall": metrics.recall_score(truth, labels),
                "f1": metrics.f1_score(truth, labels),
                "fbeta_2": metrics.fbeta_score(truth, labels, beta=2),
                "roc_auc": metrics.roc_auc_score(truth, probability),
                "average_precision": metrics.average_precision_score(truth, probability),
            }
            for name, value in recomputed.items():
                np.testing.assert_allclose(folds.iloc[i][name], value, atol=1e-14)
        pipeline, saved_config = load_model(row["model_path"])
        assert pipeline.memory is None
        assert saved_config["preprocessing"]["imputation"]["numeric_strategy"] == row["imputer"]
    indexed = board.set_index("recipe")
    for row in pd.read_csv(folder / "imputer_comparisons.csv").to_dict("records"):
        columns = [f"fold_{i}" for i in range(1, 6)]
        difference = (indexed.loc[row["imputer_a"], columns].astype(float).to_numpy()
                      - indexed.loc[row["imputer_b"], columns].astype(float).to_numpy())
        statistic = difference.mean() / np.sqrt((1 / 5 + 1 / 4) * difference.var(ddof=1))
        np.testing.assert_allclose(row["mean_diff"], difference.mean(), atol=1e-14)
        np.testing.assert_allclose(row["t"], statistic, atol=1e-12)
        np.testing.assert_allclose(row["p_value"], 2 * student_t.sf(abs(statistic), 4), atol=1e-12)

    model_path = config["output"]["model_path"]
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    pipeline, saved_config = load_model(model_path)
    new = load_data("data/compas_new_people.csv")
    completed = subprocess.run([sys.executable, "predict.py", "data/compas_new_people.csv", "predictions.csv"],
                               check=True, capture_output=True, text=True)
    print(completed.stdout, end="")
    output = pd.read_csv("predictions.csv")
    expected = predict(new, pipeline, saved_config)
    assert len(output) == len(new) == 20
    np.testing.assert_array_equal(output["id"], new["id"])
    np.testing.assert_array_equal(output["probability"].to_numpy(dtype=np.float32), expected["probability"])
    np.testing.assert_array_equal(output["prediction"], expected["prediction"])
    assert output.loc[14, "probability"] == output.loc[15, "probability"]
    assert hashlib.sha256(Path(model_path).read_bytes()).hexdigest() == model_hash
    output.to_csv("predictions/compas_new_predictions.csv", index=False)
    assert pd.isna(clean_dataset(new, config).loc[3, "priors_count"])

    # Exercise B4's error at the real CLI boundary; no output should be created.
    invalid = Path("results/week10/missing_columns.csv")
    new.drop(columns=["age"]).to_csv(invalid, index=False)
    failed_output = invalid.with_name("invalid_predictions.csv")
    failed = subprocess.run([sys.executable, "predict.py", str(invalid), str(failed_output)],
                            capture_output=True, text=True)
    assert failed.returncode == 2
    assert "missing required columns: age" in failed.stderr
    assert not failed_output.exists()

    # Adult still runs with the shared preprocessing and separate model/output paths.
    adult = deepcopy(load_config("config_adult.yaml"))
    adult["tuning"]["enabled"] = False
    adult["cv"]["n_jobs"] = 5
    adult["model"]["params"]["n_jobs"] = 1
    adult["output"].update(results_dir="results/week10/adult", scoreboard="results/week10/adult/runs.csv")
    adult["prediction"]["output_path"] = "results/week10/adult/new_predictions.csv"
    with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
        adult_result = run_pipeline(adult)
    adult_new = load_data(adult["prediction"]["input_path"])
    adult_output = pd.read_csv(adult_result["prediction_path"])
    np.testing.assert_array_equal(adult_output["id"], adult_new["id"])
    assert len(adult_output) == 2000
    adult_model, adult_saved = load_model(adult_result["model_path"])
    np.testing.assert_array_equal(predict(adult_new, adult_model, adult_saved)["probability"],
                                 adult_output["probability"].to_numpy(dtype=np.float32))
    assert hashlib.sha256(historical_predictions.read_bytes()).hexdigest() == historical_hash

    validation = {
        "date": "2026-11-20",
        "course_training_data_equal_as_dataframes": True,
        "imputers": board["recipe"].tolist(), "folds_per_imputer": 5,
        "development_rows": len(X_dev), "reserved_rows": len(X_test),
        "eight_metrics_recomputed_from_oof_on_all_20_folds": True,
        "reserved_rows_absent_from_oof": True, "six_corrected_tests_recomputed": True,
        "model_path": model_path, "model_sha256": model_hash,
        "cli_command": "python predict.py data/compas_new_people.csv predictions.csv",
        "prediction_rows": len(output), "input_ids_and_order_preserved": True,
        "duplicate_facts_receive_equal_predictions": True,
        "cli_predictions_equal_saved_model": True, "model_file_unchanged_after_inference": True,
        "missing_column_cli_error_verified": True, "threshold": 0.5,
        "adult_no_tuning_smoke": {"rows_predicted": len(adult_output),
            "scores": {name: adult_result["score_row"][f"mean_{name}"] for name in ("roc_auc", "accuracy", "f1")},
            "historical_week9_predictions_unchanged": True,
            "report_path": adult_result["report_path"]},
    }
    (folder / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print("Week 10 independent verification passed.")


if __name__ == "__main__":
    main()
