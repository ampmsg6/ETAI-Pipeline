"""Predict a CSV with a trusted pipeline saved by main.py, without training."""
import argparse
from pathlib import Path

from main import load_config
from src.artifacts import load_model
from src.data import load_data
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="New CSV; the target is not required")
    parser.add_argument("output", nargs="?", default="predictions.csv")
    parser.add_argument("--config", default="config.yaml", help="Config locating the saved model")
    parser.add_argument("--model", help="Explicit saved model path")
    args = parser.parse_args()
    try:
        model_path = args.model or load_config(args.config).get("output", {}).get("model_path", "models/model.joblib")
        pipeline, training_config = load_model(model_path)
        predictions = predict(load_data(args.input), pipeline, training_config)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False, lineterminator="\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")
    print(f"{len(predictions)} rows predicted -> {path}")


if __name__ == "__main__":
    main()
