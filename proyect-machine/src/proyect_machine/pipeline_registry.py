"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

# Pipelines existentes
from proyect_machine.pipelines.modelo_regresion.pipeline import create_pipeline as modelo_regresion_pipeline
from proyect_machine.pipelines.modelo_clasificacion.pipeline import create_pipeline as modelo_clasificacion_pipeline

# 🚀 Pipelines nuevos de aprendizaje no supervisado
from proyect_machine.pipelines.unsupervised_learning.clustering.pipeline import create_pipeline as clustering_pipeline
from proyect_machine.pipelines.unsupervised_learning.dimensionality_reduction.pipeline import create_pipeline as dr_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""

    pipeline_regresion = modelo_regresion_pipeline()
    pipeline_clasificacion = modelo_clasificacion_pipeline()
    pipeline_clustering = clustering_pipeline()
    pipeline_dr = dr_pipeline()  # dimensionality reduction

    return {
        "modelo_regresion": pipeline_regresion,
        "modelo_clasificacion": pipeline_clasificacion,
        "unsupervised_learning_clustering": pipeline_clustering,
        "dimensionality_reduction": pipeline_dr,
        "__default__": pipeline_regresion + pipeline_clustering + pipeline_dr
    }
