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
    description='Collecte les données, entraîne le modèle et met à jour Streamlit',
) as dag:

    collect_data = BashOperator(
        task_id='collect_data',
        bash_command='python /app/dag_collect_data_2.py',
        cwd='/app'
    )

    run_training = BashOperator(
        task_id='run_model_training',
        bash_command='python /app/model_2.py',
        cwd='/app'
    )

    update_streamlit_model = BashOperator(
        task_id='update_streamlit_model',
        bash_command='cp /app/temp_model/model_state_dict.pth /streamlit/temp_model/model_state_dict.pth',
        cwd='/app'
    )

    collect_data >> run_training >> update_streamlit_model
