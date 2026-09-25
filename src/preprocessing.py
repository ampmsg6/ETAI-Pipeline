"""Deterministic cleaning followed by train-only learned preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)

from src.data_diagnostics import (
    flag_invalid_values, inconsistent_age_category, inconsistent_score_category,
)


class CountEncoder(BaseEstimator, TransformerMixin):
    """Training frequency encoding, with zero for an unseen category."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.counts_ = [frame.iloc[:, i].value_counts().to_dict() for i in range(frame.shape[1])]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return np.column_stack([
            frame.iloc[:, i].map(counts).fillna(0).to_numpy(dtype=float)
            for i, counts in enumerate(self.counts_)
        ])


def recover_juvenile_felony_count(df):
    """Recover a missing component from a valid recorded total, when possible."""
    out = df.copy()
    required = {"juv_fel_count", "juv_misd_count", "juv_other_count", "juvenile_total"}
    if not required.issubset(out.columns):
        return out
    candidate = out["juvenile_total"] - out["juv_misd_count"] - out["juv_other_count"]
    recoverable = (
        out["juv_fel_count"].isna()
        & out["juvenile_total"].ge(0)
        & out["juv_misd_count"].ge(0)
        & out["juv_other_count"].ge(0)
        & candidate.ge(0)
        & candidate.eq(candidate.round())
    )
    out.loc[recoverable, "juv_fel_count"] = candidate.loc[recoverable]
    return out


def clean_dataset(df, config):
    """Apply fixed rules; no statistics are fitted from test rows."""
    # Drop only records that are identical in every raw column. A repeated
    # identifier with different case facts is retained for investigation.
    out = df.drop_duplicates().copy()
    placeholders = {str(value).strip().lower() for value in config["data"]["placeholder_tokens"]}
    for column in config["data"]["numeric_columns"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column, mapping in config["data"]["canonical_maps"].items():
        if column not in out:
            continue
        values = out[column].astype("string").str.strip()
        values = values.mask(values.str.lower().isin(placeholders))
        out[column] = values.str.lower().map(mapping).fillna(values).astype("object")

    out, _ = flag_invalid_values(out, config["diagnostics"]["validity_rules"])
    # The total is discarded only after using it to recover a missing count.
    out = recover_juvenile_felony_count(out)
    if {"age", "age_cat"}.issubset(out.columns):
        bad_age_cat = inconsistent_age_category(out)
        age = out.loc[bad_age_cat, "age"]
        out.loc[bad_age_cat, "age_cat"] = np.select(
            [age.lt(25), age.lt(45)], ["Less than 25", "25 - 45"], default="Greater than 45"
        )
    if {"decile_score", "score_text"}.issubset(out.columns):
        bad_score = inconsistent_score_category(out)
        score = out.loc[bad_score, "decile_score"]
        out.loc[bad_score, "score_text"] = np.select(
            [score.le(4), score.le(7)], ["Low", "Medium"], default="High"
        )
        # A label backed by an invalid score cannot enter the COMPAS comparison.
        out.loc[out["decile_score"].isna(), "score_text"] = np.nan

    return out.drop(columns=config["data"]["redundant_columns"], errors="ignore")


def add_missingness_indicators(df, config):
    out = df.copy()
    for column, plan in config["preprocessing"]["imputation"].items():
        if plan.get("indicator") and column in out:
            out[f"{column}_was_missing"] = out[column].isna().astype(int)
    return out


def split_features_target(df, config):
    """Return model inputs, optional target, and separate fairness attributes."""
    out = add_missingness_indicators(df, config)
    data = config["data"]
    target = data["target"]
    y = out[target].copy() if target in out else None
    extras_columns = [c for c in [data["sensitive_attr"], "score_text"] if c in out]
    extras = out[extras_columns].copy()
    feature_columns = (
        config["preprocessing"]["numeric_features"]
        + config["preprocessing"]["categorical_features"]
        + [f"{c}_was_missing" for c, plan in config["preprocessing"]["imputation"].items() if plan.get("indicator")]
    )
    return out[feature_columns].copy(), y, extras


def split_train_test(X, y, extras, test_size, random_state):
    if y is None:
        raise ValueError("A target is required for the training split")
    return train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(config):
    options = config["preprocessing"]
    imputers = {
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=options.get("knn_neighbors", 5)),
        "constant": SimpleImputer(strategy="constant", fill_value=options.get("numeric_fill_value", 0)),
    }
    scalers = {
        "none": "passthrough", "standard": StandardScaler(),
        "minmax": MinMaxScaler(), "robust": RobustScaler(),
    }
    encoders = {
        "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "count": CountEncoder(),
        # scikit-learn cross-fits training rows in fit_transform, preventing
        # each row's own target from being used in its encoded value.
        "target": TargetEncoder(target_type="binary", cv=StratifiedKFold(
            n_splits=5, shuffle=True, random_state=options.get("encoder_random_state", 42)
        )),
    }
    numeric = Pipeline([("impute", imputers[options["numeric_imputer"]]), ("scale", scalers[options["scaler"]])])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", encoders[options["encoder"]]),
    ])
    indicators = [f"{c}_was_missing" for c, plan in options["imputation"].items() if plan.get("indicator")]
    return ColumnTransformer([
        ("numeric", numeric, options["numeric_features"]),
        ("categorical", categorical, options["categorical_features"]),
        ("indicators", "passthrough", indicators),
    ])
