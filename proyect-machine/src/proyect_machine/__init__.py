"""Proyect_machine
"""

__version__ = "0.1"
from proyect_machine.pipelines.modelo_regresion import pipeline as modelo_regresion_pipeline

def register_pipelines():
    return {
        "modelo_regresion": modelo_regresion_pipeline.create_pipeline(),
        "__default__": modelo_regresion_pipeline.create_pipeline()
    }
