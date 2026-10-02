"""Holdout and cross-validation reports for the complete model pipeline."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """
    Prints -- and returns as text, so it can also be saved to disk -- train accuracy and test accuracy side by side, plus the usual classification report on the test set.
    """
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy

    lines = [
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {gap:+.3f}",
    ]
    lines.append("")
    lines.append("Classification report (test set):")
    lines.append(classification_report(y_test, y_pred))

    text = "\n".join(lines)
    print(text)
    return text


def cross_validate_pipeline(pipeline, X, y, cv, scoring="accuracy", n_jobs=1):
    """Score the whole pipeline and predict each row from its held-out fold."""
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        return_train_score=True, return_estimator=True, return_indices=True,
    )
    folds = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    folds["gap"] = folds["train"] - folds["validation"]
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return folds, y_oof


def cv_report(folds, scoring="accuracy"):
    """Report each fold and the spread of train, validation and gap scores."""
    lines = [
        f"Cross-validation ({len(folds)} stratified folds, metric: {scoring})",
        "",
        folds.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    for column in ("train", "validation", "gap"):
        lines.append(
            f"{column.capitalize():<11s} mean = {folds[column].mean():.3f}   "
            f"std = {folds[column].std(ddof=1):.3f}"
        )
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, y_pred):
    text = "Classification report (out-of-fold, development set):\n" + classification_report(
        y_true, y_pred, zero_division=0
    )
    print(text)
    return text


def fairness_report(y_test, y_pred, extras_test: pd.DataFrame, sensitive_attr: str = "race", dataset_label="test set") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = y_pred
    # An unknown COMPAS score must not silently count as high risk.
    df["y_pred_compas"] = df["score_text"].map({"Low": 0, "Medium": 1, "High": 1})

    lines = [
        f"False positive rate by race ({dataset_label})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.dropna(subset=[sensitive_attr, col]).groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
