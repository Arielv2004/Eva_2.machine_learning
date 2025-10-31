from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs="movies_metadata",  
            outputs="model_clasificacion",
            name="train_svc_model_node",
        ),
    ])
