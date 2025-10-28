"""
This is a boilerplate test file for pipeline 'new_pipeline'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from proyect_machine.pipelines.movies_pipeline.nodes import clean_movies_metadata
import pandas as pd

def test_clean_movies_metadata():
    # Creamos un DataFrame con un valor nulo
    df = pd.DataFrame({
        'title': ['A', None],
        'release_date': ['2020-01-01', None]
    })
    
    # Llamamos a la función que limpia los datos
    cleaned = clean_movies_metadata(df)
    
    # Verificamos que solo quede una fila (la que no tiene nulos)
    assert cleaned.shape[0] == 1
