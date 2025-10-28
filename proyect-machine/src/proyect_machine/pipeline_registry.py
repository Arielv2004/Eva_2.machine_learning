"""Project pipelines."""
from __future__ import annotations


from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from proyect_machine.pipelines.modelo_regresion.pipeline import create_pipeline as modelo_regresion_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines




def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    return {
        "modelo_regresion": modelo_regresion_pipeline(),  # <--- llamar directamente
        "__default__": modelo_regresion_pipeline(),        # <--- llamar directamente
    }
