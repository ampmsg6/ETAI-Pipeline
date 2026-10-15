"""Run the COMPAS pipeline: default CV, nested tuning, OOF reports, final refit."""
import argparse
from pathlib import Path

import optuna
import yaml
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import load_data
from src.preprocessing import (
    ENGINEERED_FEATURES, add_engineered_features, configure_engineered_features,
    clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target,
)
from src.model import build_pipeline, default_model_config
from src.evaluate import (
    cross_validate_pipeline, cv_report, fairness_report, oof_classification_report,
)
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_pipeline(config: dict) -> dict:
    """Execute one recipe and return its model, fold tables and report path."""
    tuning = config.get("tuning", {})
    tuning_enabled = tuning.get("enabled", False)
    model_type = config["model"]["type"]
    spaces = tuning.get("search_spaces") or {}
    if tuning_enabled and model_type not in spaces:
        raise ValueError(
            f"Tuning is enabled but no search space exists for '{model_type}'. "
            "Add one or set tuning.enabled: false."
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    raw = load_data(config["data"]["path"])
    clean = clean_dataset(raw, config)
    labelled = drop_duplicate_rows(clean, config["diagnostics"].get("id_column"))
    labelled = add_engineered_features(
        labelled, config["preprocessing"].get("engineered_features", []),
    )
    X, y, extras = split_features_target(labelled, config)
    X_dev, X_test, y_dev, _, extras_dev, _ = split_dev_test(
        X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"],
    )
    pipeline = build_pipeline(config)
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"], shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # Preserve the Week 4 development holdout, without scoring the reserved test.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25,
        random_state=cv_config["random_state"], stratify=y_dev,
    )
    holdout = clone(pipeline).fit(X_train, y_train)
    train_score = accuracy_score(y_train, holdout.predict(X_train))
    val_score = accuracy_score(y_val, holdout.predict(X_val))
    holdout_text = (
        "Development holdout (75% train / 25% validation, configured parameters)\n"
        f"Train accuracy: {train_score:.3f}\nValidation accuracy: {val_score:.3f}\n"
        f"Gap: {train_score - val_score:+.3f}"
    )
    print(holdout_text, flush=True)
    folds, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs)
    report = (
        f"Rows: {len(raw)} raw -> {len(labelled)} distinct; "
        f"{len(X_dev)} development / {len(X_test)} reserved test\n\n"
        + holdout_text + "\n\nDefault/configured hyperparameters:\n"
        + cv_report(folds, scoring)
    )
    nested_scores = study = None
    if tuning_enabled:
        inner_cv = StratifiedKFold(
            n_splits=tuning["n_splits"], shuffle=True, random_state=tuning["random_state"],
        )
        print(
            f"Nested CV: {cv.get_n_splits()} outer folds x {tuning['n_trials']} trials "
            f"x {inner_cv.get_n_splits()} inner folds", flush=True,
        )
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, spaces[model_type],
            tuning["n_trials"], tuning["random_state"], n_jobs,
        )
        report += "\n\nNested CV (honest estimate of tuning):\n" + cv_report(nested_scores, scoring)
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, spaces[model_type],
            tuning["n_trials"], tuning["random_state"], n_jobs,
        )
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    oof_source = "nested CV" if tuning_enabled else "default CV"
    report += f"\n\nOOF prediction source: {oof_source}\n"
    report += oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"],
        dataset_label=f"{oof_source}, development set",
    )
    final_model = clone(pipeline).fit(X_dev, y_dev)
    final_text = (
        f"Final model: {model_type} refit on {len(X_dev)} development rows.\n"
        f"Tuned hyperparameters: {study.best_params if study else 'tuning off'}\n"
        f"Reserved test set: {len(X_test)} rows; not evaluated in this run."
    )
    print(final_text, flush=True)
    report += "\n" + final_text + "\n"
    path = save_run(config.get("output", {}).get("results_dir", "results"), config, report)
    prefix = Path(path).with_suffix("")
    folds.to_csv(f"{prefix}_default.csv", index=False)
    if nested_scores is not None:
        nested_scores.to_csv(f"{prefix}_nested.csv", index=False)
        study.trials_dataframe().to_csv(f"{prefix}_trials.csv", index=False)
    print(f"Full results saved to {path}", flush=True)
    return {
        "model": final_model, "default_scores": folds, "nested_scores": nested_scores,
        "study": study, "report_path": path, "oof_predictions": y_oof,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=[
        "dummy", "logistic_regression", "decision_tree", "random_forest",
        "oblique_random_forest", "adaboost",
    ])
    parser.add_argument("--trials", type=int, help="Override the number of Optuna trials")
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--features", nargs="*", choices=ENGINEERED_FEATURES,
                        help="Fixed engineered features; an empty list selects the baseline")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model:
        config["model"] = default_model_config(args.model, config["cv"]["random_state"])
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        config.setdefault("tuning", {})["n_trials"] = args.trials
    if args.no_tuning:
        config.setdefault("tuning", {})["enabled"] = False
    if args.features is not None:
        configure_engineered_features(config, args.features)
    return run_pipeline(config)["model"]


if __name__ == "__main__":
    main()
