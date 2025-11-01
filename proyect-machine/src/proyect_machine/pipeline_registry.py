"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from proyect_machine.pipelines.modelo_regresion.pipeline import create_pipeline as modelo_regresion_pipeline
from proyect_machine.pipelines.modelo_clasificacion.pipeline import create_pipeline as modelo_clasificacion_pipeline
from proyect_machine.pipelines.analisis_modelos.pipeline import create_pipeline as analisis_modelos_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    # Crear cada pipeline por separado
    pipeline_regresion = modelo_regresion_pipeline()
    pipeline_clasificacion = modelo_clasificacion_pipeline()
    pipeline_analisis = analisis_modelos_pipeline()

    # Retornar todos los pipelines
    return {
        "modelo_regresion": pipeline_regresion,
        "modelo_clasificacion": pipeline_clasificacion,
        "analisis_modelos": pipeline_analisis,
        "__default__": pipeline_regresion + pipeline_analisis,
    }
