"""Deterministic cleaning followed by train-only learned preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)

from src.data_diagnostics import (
    flag_invalid_values, inconsistent_age_category, inconsistent_score_category,
)


# Every imputer learns only from numeric inputs, inside each training fold.
_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median", keep_empty_features=True),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean", keep_empty_features=True),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5), keep_empty_features=True),
    "iterative": lambda cfg, seed: IterativeImputer(
        max_iter=cfg.get("max_iter", 10), random_state=seed, keep_empty_features=True),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50),
                                        min_samples_leaf=5, random_state=seed, n_jobs=1),
        max_iter=cfg.get("max_iter", 10), random_state=seed, keep_empty_features=True),
    # Retained for the earlier preprocessing experiments.
    "constant": lambda cfg, seed: SimpleImputer(
        strategy="constant", fill_value=cfg.get("fill_value", 0), keep_empty_features=True),
}


def numeric_imputation_settings(config):
    """Course settings, with the explicit legacy overrides used by Weeks 3–4."""
    options = config["preprocessing"]
    settings = dict(options.get("imputation", {}))
    strategy = options.get("numeric_imputer", settings.get("numeric_strategy", "median"))
    if strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {strategy}. Options: {list(_NUMERIC_IMPUTERS)}")
    settings.setdefault("n_neighbors", options.get("knn_neighbors", 5))
    settings.setdefault("fill_value", options.get("numeric_fill_value", 0))
    return strategy, settings


def missingness_sources(config):
    """Keep the per-column indicator plans alongside the new imputer settings."""
    return [column for column, plan in config["preprocessing"].get("imputation", {}).items()
            if isinstance(plan, dict) and plan.get("indicator")]


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
    for column in missingness_sources(config):
        if column in out:
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
        + [f"{c}_was_missing" for c in missingness_sources(config)]
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
    strategy, imputation = numeric_imputation_settings(config)
    seed = options.get("random_state", options.get("encoder_random_state", 42))
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
    # Scalers learn from observed training values and leave NaN for the imputer.
    # Scaling first gives KNN meaningful distances across numeric columns.
    numeric = Pipeline([("scale", scalers[options["scaler"]]),
                        ("impute", _NUMERIC_IMPUTERS[strategy](imputation, seed))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"),
                                 keep_empty_features=True)),
        ("encode", encoders[options["encoder"]]),
    ])
    indicators = [f"{c}_was_missing" for c in missingness_sources(config)]
    return ColumnTransformer([
        ("numeric", numeric, options["numeric_features"]),
        ("categorical", categorical, options["categorical_features"]),
        ("indicators", "passthrough", indicators),
    ])


class AnomalyScore(TransformerMixin, BaseEstimator):
    """Append a training-fitted anomaly score; higher means more unusual.

    LOF uses proper training scores in fit_transform and novelty scores for
    validation or new rows. Neither method drops rows or uses the target.
    """

    def __init__(self, method="isolation_forest", n_neighbors=20, random_state=None):
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        values = validate_data(self, X, dtype=float)
        if self.method == "isolation_forest":
            self.detector_ = IsolationForest(random_state=self.random_state, n_jobs=1)
        elif self.method == "lof":
            self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, n_jobs=1)
        else:
            raise ValueError(f"Unknown anomaly method: {self.method}. Options: isolation_forest, lof")
        self.detector_.fit(values)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        if self.method == "lof":
            return np.column_stack([X, -self.detector_.negative_outlier_factor_])
        return self.transform(X)

    def transform(self, X):
        check_is_fitted(self, "detector_")
        values = validate_data(self, X, reset=False, dtype=float)
        return np.column_stack([values, -self.detector_.score_samples(values)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "detector_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or len(names) != self.n_features_in_:
            raise ValueError("input_features must match the fitted number of features")
        return np.append(names, "anomaly_score")
