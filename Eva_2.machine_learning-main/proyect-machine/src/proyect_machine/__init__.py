from proyect_machine.pipelines.modelo_regresion import create_pipeline as modelo_regresion_pipeline

def register_pipelines():
    return {
        "modelo_regresion": modelo_regresion_pipeline(),

        "__default__": modelo_regresion_pipeline() 
    }
