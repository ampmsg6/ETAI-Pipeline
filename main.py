"""Configurable binary pipeline: COMPAS or Adult, with development-only evaluation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    ENGINEERED_FEATURES, add_engineered_features, configure_engineered_features,
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test,
)
from src.model import build_pipeline, default_model_config, _MODELS
from src.evaluate import (build_scorers, cross_validate_pipeline, cv_report, oof_classification_report,
                          fairness_report)
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import comparison_key, save_run, save_scores
from src.predict import predict_new_data


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> dict:
    """Execute one complete recipe without evaluating the reserved test set."""
    with TemporaryDirectory(prefix="etai_pipeline_cache_") as cache_dir:
        result = _run_pipeline(config, cache_dir)
        # The fitted estimator stays usable after the run's cache is removed.
        result["model"].memory = None
        return result


def _run_pipeline(config: dict, cache_dir: str) -> dict:
    run = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    model_type = config["model"]["type"]

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config)
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    df_clean = add_engineered_features(df_clean, config["preprocessing"].get("engineered_features", []))
    X, y, extras = split_features_target(df_clean, config)

    # From here on, every decision uses the development set only; the test set stays locked.
    X_dev, X_test, y_dev, _, extras_dev, _ = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    if selection_metric not in scorers:
        raise ValueError("cv.selection_metric must be included in cv.metrics")
    n_jobs = cv_config.get("n_jobs", 1)
    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV   |   feature selection: "
          f"{'min_votes ' + str(selection_config.get('min_votes', 2)) if selection_enabled else 'off'}")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    pipeline = build_pipeline(config, selection_config=selection_config, memory=cache_dir)
    floor_pipeline = build_pipeline(config, default_model_config("dummy"), selection_config={})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores
    study = nested_scores = None

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        if selection_enabled:
            search_space.update(selection_config.get("search_space") or {})
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold probabilities feed the reports.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        nested_scores = final_scores
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    audit = config.get("audit") or {}
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"],
                                    reference=audit.get("reference"),
                                    negative_description=audit.get("negative_description"))

    report += f"\nOOF source: {'nested CV' if tuning_enabled else 'default CV'}; threshold: 0.5\n"
    pipeline.fit(X_dev, y_dev)
    all_features = pipeline.named_steps["prep"].get_feature_names_out()
    kept_features = (pipeline.named_steps["select"].get_feature_names_out(all_features)
                     if selection_enabled else all_features)
    refit = (f"Final model: {model_type} refit on all {len(X_dev)} development rows. "
             f"Reserved test: {len(X_test)} rows, not evaluated.")
    refit += f"\nFeatures kept: {len(kept_features)} of {len(all_features)}: {', '.join(kept_features)}"
    if selection_enabled:
        refit += f"\nFinal min_votes: {pipeline.named_steps['select'].min_votes}"
    feature_table = pd.DataFrame({"feature": all_features,
                                  "kept": np.isin(all_features, kept_features)})
    if selection_enabled:
        feature_table["votes"] = pipeline.named_steps["select"].votes_
    report += "\n" + refit + "\n"
    prediction_path = None
    if config.get("prediction"):
        prediction_path = predict_new_data(pipeline, config)
        report += f"\nUnlabelled predictions saved to: {prediction_path}\n"

    row = {"comparison_key": comparison_key(config), "run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "feature_selection": (f"min_votes {pipeline.named_steps['select'].min_votes}" if selection_enabled else "off"),
           "n_features": len(kept_features), "kept_features": ", ".join(kept_features),
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric,
           "trials": config.get("tuning", {}).get("n_trials", 0) if tuning_enabled else 0,
           "outer_folds": cv_config["n_splits"],
           "inner_folds": config.get("tuning", {}).get("n_splits", 0) if tuning_enabled else 0,
           "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "untuned_std": fold_scores["validation"].std(ddof=1), "floor": floor,
           "report_path": str(Path(config.get("output", {}).get("results_dir", "results")) / f"run_{run}.txt"),
           **{f"mean_{name}": final_scores[name].mean() for name in metrics},
           **{f"default_fold_{i}": score for i, score in enumerate(fold_scores["validation"], start=1)},
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["comparison_key"] == row["comparison_key"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "feature_selection", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs with matching data, cleaning, folds and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(config.get("output", {}).get("results_dir", "results"), config, report, run=run)
    prefix = Path(path).with_suffix("")
    fold_scores.to_csv(f"{prefix}_default.csv", index=False)
    feature_table.to_csv(f"{prefix}_features.csv", index=False)
    if nested_scores is not None:
        nested_scores.to_csv(f"{prefix}_nested.csv", index=False)
        study.trials_dataframe().to_csv(f"{prefix}_trials.csv", index=False)
    pd.DataFrame({"row_index": X_dev.index, "y_true": y_dev.to_numpy(),
                  "oof_probability": oof_proba, "prediction": (oof_proba >= 0.5).astype(int)}).to_csv(
        f"{prefix}_oof.csv", index=False,
    )
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")

    return {"model": pipeline, "default_scores": fold_scores, "nested_scores": nested_scores,
            "study": study, "report_path": path, "oof_probabilities": oof_proba,
            "oof_predictions": (oof_proba >= 0.5).astype(int), "score_row": row,
            "kept_features": kept_features.tolist(), "prediction_path": prediction_path}


def main():
    parser = argparse.ArgumentParser(description="Binary predictive pipeline (COMPAS by default)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=list(_MODELS))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--features", nargs="*", choices=ENGINEERED_FEATURES)
    parser.add_argument("--selection", choices=("on", "off"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model:
        config["model"] = default_model_config(args.model, config["cv"]["random_state"])
    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        config["tuning"]["n_trials"] = args.trials
    if args.no_tuning or config["model"]["type"] == "dummy":
        config["tuning"]["enabled"] = False
    if args.features is not None:
        configure_engineered_features(config, args.features)
    if args.selection is not None:
        config.setdefault("feature_selection", {})["enabled"] = args.selection == "on"
    return run_pipeline(config)["model"]


if __name__ == "__main__":
    main()
