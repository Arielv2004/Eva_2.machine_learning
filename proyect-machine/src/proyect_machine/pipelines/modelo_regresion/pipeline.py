"""
Pipeline para los modelos de regresión.
Conecta los nodes definidos en nodes.py para entrenar los distintos modelos.
"""

from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=nodes.train_linear_regression_imputer,  # o train_linear_regression_dropna
            inputs=["movies_metadata", "params:target_col", "params:feature_cols", "params:test_size", "params:random_state"],
            outputs=["linear_model", "linear_metrics"],
            name="train_linear_regression"
        ),
        node(
            func=nodes.train_multiple_linear_regression,
            inputs=["movies_metadata", "params:target_col", "params:feature_cols", "params:test_size", "params:random_state"],
            outputs=["multiple_linear_model", "multiple_linear_metrics"],
            name="train_multiple_linear_regression"
        ),
        node(
            func=nodes.train_decision_tree,
            inputs=["movies_metadata", "params:target_col", "params:feature_cols", "params:test_size", "params:random_state"],
            outputs=["decision_tree_model", "decision_tree_metrics"],
            name="train_decision_tree"
        ),
        
        node(
            func=nodes.train_random_forest,
            inputs=["movies_metadata", "params:target_col", "params:feature_cols", "params:test_size", "params:random_state"],
            outputs=["random_forest_model", "random_forest_metrics"],
            name="train_random_forest"
        ),
    ])
