"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

# Pipelines existentes
from proyect_machine.pipelines.modelo_regresion.pipeline import create_pipeline as modelo_regresion_pipeline
from proyect_machine.pipelines.modelo_clasificacion.pipeline import create_pipeline as modelo_clasificacion_pipeline

# 🚀 Nuevo pipeline de clustering (IMPORTANTE)
from proyect_machine.pipelines.unsupervised_learning.pipeline import create_pipeline as unsupervised_learning_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""

    pipeline_regresion = modelo_regresion_pipeline()
    pipeline_clasificacion = modelo_clasificacion_pipeline()
    pipeline_unsupervised = unsupervised_learning_pipeline()  # NUEVO

    return {
        "modelo_regresion": pipeline_regresion,
        "modelo_clasificacion": pipeline_clasificacion,
        "unsupervised_learning": pipeline_unsupervised,  # NUEVO
        "__default__": pipeline_regresion
    }
