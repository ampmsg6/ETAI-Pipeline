"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline
on several metrics, out-of-fold probabilities and reports, a fairness check and a paired
comparison of two models. The locked test set is never used here.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    if not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("Provide a nonempty list of unique metric names")
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            if not math.isfinite(beta) or beta <= 0:
                raise ValueError("F-beta requires a positive finite beta")
            scorers[name] = make_scorer(fbeta_score, beta=beta, zero_division=0)
        elif name in ("precision", "recall", "f1"):
            from sklearn import metrics as skmetrics
            scorers[name] = make_scorer(getattr(skmetrics, f"{name}_score"), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probability of the positive class for every row, from the
        fold model that did not train on it. `cv` must be a partition (each row validated
        exactly once).
    """
    if selection_metric not in scorers:
        raise ValueError("selection_metric must be included in cv.metrics")
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs, error_score="raise")
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.full(len(X), np.nan)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = positive_probabilities(model, X.iloc[val_idx])
    if not np.isfinite(oof_proba).all():
        raise ValueError("CV must validate every row exactly once")
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Provide equal-length one-dimensional scores for at least two paired folds")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or test_train_ratio <= 0:
        raise ValueError("Scores must be finite and test_train_ratio must be positive")
    diff = a - b
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else (0.0 if diff.mean() == 0 else float(np.copysign(np.inf, diff.mean())))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}",
            "t": t, "p_value": float(2 * stats.t.sf(abs(t), df=k - 1))}


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5, dataset_label="development set, out-of-fold") -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = df["score_text"].map({"Low": 0, "Medium": 1, "High": 1})

    lines = [
        f"False positive rate by {sensitive_attr} ({dataset_label}; threshold {threshold})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.dropna(subset=[col]).groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)


def positive_probabilities(estimator, X):
    """Return P(class 1), using the fitted class order rather than assuming column 1."""
    classes = np.asarray(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("This pipeline requires binary labels 0 and 1")
    return estimator.predict_proba(X)[:, int(np.flatnonzero(classes == 1)[0])]
