from kedro.pipeline import Pipeline, node, pipeline
from .nodes import comparar_modelos

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=comparar_modelos,
            inputs=[
                "linear_simple_metrics",
                "linear_multiple_metrics",
                "decision_tree_metrics",
                "random_forest_metrics",
                "knn_regressor_metrics"
            ],
            outputs="comparacion_modelos_df",
            name="comparar_modelos_node"
        )
    ])
