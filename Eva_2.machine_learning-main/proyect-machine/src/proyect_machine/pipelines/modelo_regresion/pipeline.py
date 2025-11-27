from kedro.pipeline import Pipeline, node
from .nodes import (
    train_linear_simple,
    train_linear_multiple,
    train_decision_tree,
    train_random_forest,
    train_knn_regressor
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_linear_simple,
            inputs="movies_metadata",
            outputs=["linear_simple_model", "linear_simple_metrics"],
            name="train_linear_simple_node"
        ),
        node(
            func=train_linear_multiple,
            inputs="movies_metadata",
            outputs=["linear_multiple_model", "linear_multiple_metrics"],
            name="train_linear_multiple_node"
        ),
        node(
            func=train_decision_tree,
            inputs="movies_metadata",
            outputs=["decision_tree_model", "decision_tree_metrics"],
            name="train_decision_tree_node"
        ),
        node(
            func=train_random_forest,
            inputs="movies_metadata",
            outputs=["random_forest_model", "random_forest_metrics"],
            name="train_random_forest_node"
        ),
        node(
            func=train_knn_regressor,
            inputs="movies_metadata",
            outputs=["knn_regressor_model", "knn_regressor_metrics"],
            name="train_knn_regressor_node"
        ),
    ])
