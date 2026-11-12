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
    """Apply fixed rules without changing the number or order of rows.

    The same cleaning can therefore be used on unlabelled prediction data,
    where every input row must receive a prediction.
    """
    out = df.copy()
    placeholders = {str(value).strip().lower() for value in config["data"]["placeholder_tokens"]}
    for column in config["data"]["numeric_columns"]:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    mappings = config["data"].get("canonical_maps", {})
    # Missing-token cleanup also applies to categories with no spelling map.
    categorical_columns = dict.fromkeys([
        *mappings, *config["preprocessing"]["categorical_features"],
    ])
    for column in categorical_columns:
        if column not in out:
            continue
        mapping = mappings.get(column, {})
        values = out[column].astype("string").str.strip()
        values = values.mask(values.str.lower().isin(placeholders))
        canonical = values.str.lower().map(mapping).fillna(values).astype("object")
        # SimpleImputer's object-array mask expects np.nan, not nullable pd.NA.
        out[column] = canonical.where(canonical.notna(), np.nan)

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


def drop_duplicate_rows(df, id_column=None):
    """Remove exact copies from labelled data before the dev/test split.

    A repeated id with different case facts is ambiguous: do not silently
    discard a record merely because its identifier matches another one.
    """
    out = df.drop_duplicates().copy()
    if id_column in out and out[id_column].duplicated().any():
        raise ValueError(f"Conflicting records share {id_column}; inspect them before splitting")
    return out


def add_missingness_indicators(df, config):
    out = df.copy()
    for column, plan in config["preprocessing"]["imputation"].items():
        if plan.get("indicator") and column in out:
            out[f"{column}_was_missing"] = out[column].isna().astype(int)
    return out


ENGINEERED_FEATURES = (
    "priors_per_adult_year", "juvenile_per_adult_year", "priors_age_interaction",
)


def add_engineered_features(df, features=ENGINEERED_FEATURES):
    """Fixed, row-preserving formulas; missing components remain missing.

    No statistic is learned, so these features can be made before splitting.
    The denominator follows the notebook's age-17 convention (minimum 1).
    """
    unknown = set(features) - set(ENGINEERED_FEATURES)
    if unknown:
        raise ValueError(f"Unknown engineered features: {sorted(unknown)}")
    out = df.copy()
    if not features:
        return out
    adult_years = (out["age"] - 17).clip(lower=1)
    if "priors_per_adult_year" in features:
        out["priors_per_adult_year"] = out["priors_count"] / adult_years
    if "juvenile_per_adult_year" in features:
        # Do not silently interpret a missing juvenile count as zero.
        juvenile_count = out[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(
            axis=1, min_count=3
        )
        out["juvenile_per_adult_year"] = juvenile_count / adult_years
    if "priors_age_interaction" in features:
        out["priors_age_interaction"] = out["priors_count"] * out["age"]
    return out


def configure_engineered_features(config, features):
    """Select a feature ablation in the run config, without changing the base recipe."""
    unknown = set(features) - set(ENGINEERED_FEATURES)
    if unknown:
        raise ValueError(f"Unknown engineered features: {sorted(unknown)}")
    options = config["preprocessing"]
    options["engineered_features"] = list(dict.fromkeys(features))
    baseline = [name for name in options["numeric_features"] if name not in ENGINEERED_FEATURES]
    options["numeric_features"] = baseline + options["engineered_features"]


def split_features_target(df, config):
    """Return model inputs, optional target, and separate fairness attributes."""
    out = add_missingness_indicators(df, config)
    data = config["data"]
    target = data["target"]
    y = out[target].copy() if target in out else None
    reference_column = (config.get("audit", {}).get("reference") or {}).get("column")
    extras_columns = list(dict.fromkeys(c for c in [data["sensitive_attr"], reference_column] if c in out))
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


def split_dev_test(X, y, extras, test_size, random_state):
    """Reserve a stratified test set, aligned with features and audit fields."""
    return split_train_test(X, y, extras, test_size, random_state)


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
