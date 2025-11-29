from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 1),
    "retries": 0,
}

with DAG(
    dag_id="ml_pipeline_master",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["kedro", "machine_learning", "final_project"],
) as dag:

    modelo_regresion = BashOperator(
        task_id="modelo_regresion",
        bash_command=(
            "cd /opt/airflow/src/proyect-machine && "
            "kedro run --pipeline=modelo_regresion"
        ),
    )

    modelo_clasificacion = BashOperator(
        task_id="modelo_clasificacion",
        bash_command=(
            "cd /opt/airflow/src/proyect-machine && "
            "kedro run --pipeline=modelo_clasificacion"
        ),
    )

    clustering = BashOperator(
        task_id="clustering",
        bash_command=(
            "cd /opt/airflow/src/proyect-machine && "
            "kedro run --pipeline=unsupervised_learning_clustering"
        ),
    )

    dimensionality_reduction = BashOperator(
        task_id="dimensionality_reduction",
        bash_command=(
            "cd /opt/airflow/src/proyect-machine && "
            "kedro run --pipeline=dimensionality_reduction"
        ),
    )

    # Dependencias en cadena
    modelo_regresion >> modelo_clasificacion >> clustering >> dimensionality_reduction
