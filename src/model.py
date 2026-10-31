"""Model construction and the complete, fold-fitted estimator."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

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


def build_pipeline(config: dict, model_config: dict | None = None) -> Pipeline:
    """Keep the project's full-config preprocessing API and expose prep__/model__ paths."""
    return Pipeline([
        ("prep", build_preprocessor(config)),
        ("model", build_model(model_config or config["model"])),
    ])
