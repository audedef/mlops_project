from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    'train_model_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Entraîne le modèle ML depuis model_2.py',
) as dag:

    run_training = BashOperator(
        task_id='run_model_training',
        bash_command='python /app/model_2.py',
        cwd='/app'
    )
