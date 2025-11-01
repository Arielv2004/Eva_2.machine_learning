"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from proyect_machine.pipelines.modelo_regresion.pipeline import create_pipeline as modelo_regresion_pipeline
from proyect_machine.pipelines.modelo_clasificacion.pipeline import create_pipeline as modelo_clasificacion_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    modelo_regresion = modelo_regresion_pipeline()
    modelo_clasificacion = modelo_clasificacion_pipeline()

    return {
        "__default__": modelo_regresion + modelo_clasificacion,  # Ejecuta ambos al correr `kedro run`
        "modelo_regresion": modelo_regresion,
        "modelo_clasificacion": modelo_clasificacion,
    }
