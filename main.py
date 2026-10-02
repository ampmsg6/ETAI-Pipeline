"""Run the Week 4 COMPAS pipeline and save its development-set report."""

import yaml
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor, clean_dataset, drop_duplicate_rows,
    split_dev_test, split_features_target,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline, cv_report, fairness_report, oof_classification_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    raw = load_data(config["data"]["path"])
    clean = clean_dataset(raw, config)
    labelled = drop_duplicate_rows(clean, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(labelled, config)

    X_dev, X_test, y_dev, _, extras_dev, _ = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor(config)),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"], shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )

    # A single holdout on development rows makes the Week 3 style estimate
    # visible without scoring the reserved test set again.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25,
        random_state=cv_config["random_state"], stratify=y_dev,
    )
    holdout = clone(pipeline).fit(X_train, y_train)
    holdout_train = accuracy_score(y_train, holdout.predict(X_train))
    holdout_val = accuracy_score(y_val, holdout.predict(X_val))
    holdout_text = (
        "Development holdout (75% train / 25% validation)\n"
        f"Train accuracy: {holdout_train:.3f}\n"
        f"Validation accuracy: {holdout_val:.3f}\n"
        f"Gap: {holdout_train - holdout_val:+.3f}"
    )
    print(holdout_text)

    folds, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv,
        scoring=cv_config.get("scoring", "accuracy"),
        n_jobs=cv_config.get("n_jobs", 1),
    )
    report = (
        f"Rows: {len(raw)} raw -> {len(labelled)} distinct; "
        f"{len(X_dev)} development / {len(X_test)} reserved test\n\n"
        + holdout_text + "\n\n"
    )
    report += cv_report(folds, cv_config.get("scoring", "accuracy"))
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
        dataset_label="out-of-fold, development set",
    )

    final_model = clone(pipeline).fit(X_dev, y_dev)
    final_text = (
        f"Final model: {config['model']['type']} refit on {len(X_dev)} development rows.\n"
        f"Reserved test set: {len(X_test)} rows; not evaluated in this run."
    )
    print(final_text)
    report += "\n" + final_text + "\n"
    path = save_run(config.get("output", {}).get("results_dir", "results"), config, report)
    print(f"Full results saved to {path}")
    return final_model


if __name__ == "__main__":
    main()
