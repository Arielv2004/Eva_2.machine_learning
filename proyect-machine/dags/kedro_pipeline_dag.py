from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuración básica del DAG
default_args = {
    'owner': 'Sergio Vera',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Definición del DAG
with DAG(
    dag_id='kedro_pipeline_dag',
    default_args=default_args,
    description='Ejecuta el pipeline de Kedro del proyecto Machine Learning',
    schedule_interval=None,  # se ejecuta manualmente desde la UI
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=['machine_learning', 'kedro', 'duoc'],
) as dag:

    # Tarea 1: correr pipeline Kedro
    run_kedro = BashOperator(
        task_id='run_kedro_pipeline',
        bash_command='cd /opt/airflow/proyect-machine && kedro run'
    )

    # Tarea 2: ejecutar notebook de evaluación
    run_evaluation = BashOperator(
        task_id='run_evaluation_notebook',
        bash_command='cd /opt/airflow/proyect-machine/notebooks && jupyter nbconvert --to notebook --execute 05_evaluation.ipynb --output results_evaluation.ipynb'
    )

    # Flujo
    run_kedro >> run_evaluation
