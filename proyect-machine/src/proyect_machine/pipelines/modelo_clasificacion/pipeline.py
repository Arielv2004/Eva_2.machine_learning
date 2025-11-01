# pipeline.py
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_random_forest
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_regression,
            inputs="movies_metadata",
            outputs="logistic_model",
            name="train_logistic_regression_node",
        ),
        node(
            func=train_decision_tree,
            inputs="movies_metadata",
            outputs="decision_tree_model",
            name="train_decision_tree_node",
        ),
        node(
            func=train_knn,
            inputs="movies_metadata",
            outputs="knn_model",
            name="train_knn_node",
        ),
        node(
            func=train_random_forest,
            inputs="movies_metadata",
            outputs="random_forest_model",
            name="train_random_forest_node",
        ),
    ])
