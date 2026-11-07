"""Model construction and the complete, fold-fitted estimator."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, validate_data
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def adaboost(max_depth: int = 1, **params):
    """AdaBoost over shallow trees; depth is model__estimator__max_depth."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth), **params
    )


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10,
                          density="auto", random_state=None, n_jobs=None, **tree_params):
    """Bootstrap trees, each with its own random sparse feature projections."""
    tree_params.setdefault("max_features", "sqrt")
    estimator = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=estimator, n_estimators=n_estimators,
        random_state=random_state, n_jobs=n_jobs,
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def default_model_config(model_type: str, random_state: int = 42) -> dict:
    """Notebook defaults used by the CLI and reproducible comparisons."""
    params = {
        "dummy": {"strategy": "most_frequent"},
        "logistic_regression": {"max_iter": 2000},
        "decision_tree": {"random_state": random_state},
        "random_forest": {"n_estimators": 200, "random_state": random_state, "n_jobs": -1},
        "oblique_random_forest": {
            "n_estimators": 200, "n_components": 10, "random_state": random_state, "n_jobs": -1,
        },
        "adaboost": {
            "max_depth": 1, "n_estimators": 200, "learning_rate": 1.0,
            "random_state": random_state,
        },
        "gradient_boosting": {"random_state": random_state},
        "hist_gradient_boosting": {"random_state": random_state},
        "naive_bayes": {},
        "xgboost": {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
            "random_state": random_state, "n_jobs": -1,
        },
    }
    if model_type not in params:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(params)}")
    return {"type": model_type, "params": params[model_type]}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep features chosen by at least min_votes of the course's three voters.

    Forest/boosting vote for above-mean importance; L1 logistic regression
    votes for nonzero coefficients. Each fit uses only its training rows.
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        if self.min_votes not in (1, 2, 3):
            raise ValueError("min_votes must be 1, 2 or 3")
        X, y = validate_data(self, X, y)
        # CV already parallelises folds; avoid parallel forests within each worker.
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                        random_state=self.random_state, n_jobs=1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear",
                                   random_state=self.random_state).fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            raise ValueError("VoteSelector kept no features; reduce min_votes or disable selection")
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def build_pipeline(config: dict, model_config: dict | None = None,
                   selection_config: dict | None = None, memory=None) -> Pipeline:
    """Keep the full-config API; fit preprocessing and selection inside every fold.

    memory caches fitted steps only for identical training data, labels and
    parameters. Explicit selection_config overrides config['feature_selection'].
    """
    selection = config.get("feature_selection", {}) if selection_config is None else selection_config
    steps = [("prep", build_preprocessor(config))]
    if selection.get("enabled", False):
        steps.append(("select", VoteSelector(selection.get("min_votes", 2),
                                             selection.get("random_state", 42))))
    steps.append(("model", build_model(model_config or config["model"])))
    return Pipeline(steps, memory=memory)
