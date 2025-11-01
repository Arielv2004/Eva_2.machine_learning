from kedro.pipeline import Pipeline, node
from .nodes import load_linear_model, predict_with_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=load_linear_model,
            inputs="params:model_path",
            outputs="loaded_model",
            name="load_linear_model_node"
        ),
        node(
            func=lambda model, data: predict_with_model(model, data, ["revenue"]),
            inputs=["loaded_model", "movies_metadata"],
            outputs="movies_with_predictions",
            name="predict_linear_node"     
        ),
    ])
