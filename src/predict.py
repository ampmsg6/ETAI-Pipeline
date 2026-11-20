"""Row-preserving inference on unlabelled data using a fitted complete pipeline."""
from pathlib import Path

import pandas as pd

from src.data import load_data
from src.evaluate import positive_probabilities
from src.preprocessing import (
    ENGINEERED_FEATURES, clean_dataset, add_engineered_features, split_features_target,
)


def predict(raw: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """One prediction per input row, using only previously fitted transforms."""
    id_column = config.get("prediction", {}).get(
        "id_column", config["diagnostics"].get("id_column", "id"))
    options = config["preprocessing"]
    required = set(options["numeric_features"] + options["categorical_features"]) - set(ENGINEERED_FEATURES)
    engineered = options.get("engineered_features", [])
    if engineered:
        required.add("age")
    if {"priors_per_adult_year", "priors_age_interaction"} & set(engineered):
        required.add("priors_count")
    if "juvenile_per_adult_year" in engineered:
        required.update(("juv_fel_count", "juv_misd_count", "juv_other_count"))
    missing = sorted((required | {id_column}) - set(raw.columns))
    if missing:
        raise ValueError("Prediction input is missing required columns: " + ", ".join(missing))
    if raw[id_column].isna().any():
        raise ValueError(f"Prediction data needs a nonmissing identifier column: {id_column}")
    if raw.empty:
        return pd.DataFrame(columns=[id_column, "probability", "prediction"])
    clean = clean_dataset(raw, config)
    # Repeated new records also need a prediction each.
    clean = add_engineered_features(clean, engineered)
    X, _, _ = split_features_target(clean, config)
    probabilities = positive_probabilities(pipeline, X)
    result = pd.DataFrame({id_column: raw[id_column].to_numpy(), "probability": probabilities,
                           "prediction": (probabilities >= 0.5).astype(int)})
    if len(result) != len(raw):
        raise ValueError("Prediction must preserve the number of input rows")
    return result


def predict_new_data(pipeline, config: dict) -> str:
    """Keep every input row, ID and order; apply the training recipe without refitting."""
    settings = config["prediction"]
    raw = load_data(settings["input_path"])
    result = predict(raw, pipeline, config)
    path = Path(settings["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, lineterminator="\n")
    print(f"Predictions: {len(result)} input rows -> {len(result)} output rows: {path}")
    return str(path)
