from proyect_machine.pipelines.modelo_regresion import create_pipeline as modelo_regresion_pipeline
from proyect_machine.pipelines.analisis_modelos import create_pipeline as analisis_modelos_pipeline

def register_pipelines():
    return {
        "modelo_regresion": modelo_regresion_pipeline(),
        "analisis_modelos": analisis_modelos_pipeline(),
        "__default__": modelo_regresion_pipeline() + analisis_modelos_pipeline(),
    }
