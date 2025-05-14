from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2024, 1, 1),
    'catchup': False,
}

with DAG(
    dag_id='pipeline_mlops',
    default_args=default_args,
    schedule_interval=None,
    description='Pipeline MLOps : collecte et entraînement',
    tags=['mlops'],
) as dag:

    collect_data = BashOperator(
        task_id='collect_data',
        bash_command='python3 /opt/airflow/scripts/dag_collect_data_2.py'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /opt/airflow/scripts/model_2.py'
    )

    collect_data >> train_model
