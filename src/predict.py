"""Row-preserving inference on unlabelled data using a fitted complete pipeline."""
from pathlib import Path

import pandas as pd

from src.data import load_data
from src.evaluate import positive_probabilities
from src.preprocessing import clean_dataset, add_engineered_features, split_features_target


def predict_new_data(pipeline, config: dict) -> str:
    """Keep every input row, ID and order; apply the training recipe without refitting."""
    settings = config["prediction"]
    raw = load_data(settings["input_path"])
    id_column = settings.get("id_column", config["diagnostics"].get("id_column", "id"))
    if id_column not in raw or raw[id_column].isna().any():
        raise ValueError(f"Prediction data needs a nonmissing identifier column: {id_column}")
    clean = clean_dataset(raw, config)
    # No deduplication here: repeated new records also need a prediction each.
    clean = add_engineered_features(clean, config["preprocessing"].get("engineered_features", []))
    X, _, _ = split_features_target(clean, config)
    probabilities = positive_probabilities(pipeline, X)
    result = pd.DataFrame({id_column: raw[id_column].to_numpy(), "probability": probabilities,
                           "prediction": (probabilities >= 0.5).astype(int)})
    if len(result) != len(raw):
        raise ValueError("Prediction must preserve the number of input rows")
    path = Path(settings["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"Predictions: {len(result)} input rows -> {len(result)} output rows: {path}")
    return str(path)
