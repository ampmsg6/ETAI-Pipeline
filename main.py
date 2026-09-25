"""Run the complete COMPAS pipeline from configuration to saved report."""

import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, split_train_test, build_preprocessor
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    df_raw = load_data(config["data"]["path"])
    df = clean_dataset(df_raw, config)
    X, y, extras = split_features_target(df, config)
    X_train, X_test, y_train, y_test, _, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(config)),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = f"Rows: {len(df_raw)} raw -> {len(df)} after cleaning; {len(X_train)} train / {len(X_test)} test\n\n"
    report += evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
