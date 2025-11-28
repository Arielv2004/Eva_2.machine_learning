"""Paquete principal del proyecto Kedro."""

# Kedro / setuptools usan esta variable cuando leen pyproject.toml
__version__ = "0.1.0"


def register_pipelines():
    """Registra los pipelines del proyecto.

    Usamos un import dentro de la función para evitar errores
    cuando setuptools importa este módulo durante `pip install -e .`.
    """
    from proyect_machine.pipelines.modelo_regresion import (
        create_pipeline as modelo_regresion_pipeline,
    )

    return {
        "modelo_regresion": modelo_regresion_pipeline(),
        "__default__": modelo_regresion_pipeline(),
    }
