from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'retries': 0,
}

with DAG(
    dag_id='run_kedro_pipelines',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['kedro', 'machine_learning'],
) as dag:

    run_modelo_regresion = BashOperator(
        task_id='run_modelo_regresion',
        bash_command='cd /opt/airflow/src/proyect-machine && kedro run --pipeline modelo_regresion',
    )

    run_modelo_clasificacion = BashOperator(
        task_id='run_modelo_clasificacion',
        bash_command='cd /opt/airflow/src/proyect-machine && kedro run --pipeline modelo_clasificacion',
    )

    run_modelo_regresion >> run_modelo_clasificacion
