from .pipeline import create_pipeline

def register_pipelines():
    return {
        "modelo_regresion": create_pipeline(),
        "__default__": create_pipeline(),
    }
