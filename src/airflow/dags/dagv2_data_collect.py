from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import requests
import os

# Configuration du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'download_and_store_images',
    default_args=default_args,
    description='A simple DAG to download and store images',
    schedule_interval=timedelta(days=1),
)


# Fonction pour télécharger une image
def download_image(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)


# Fonction pour uploader une image vers Minio
def upload_to_minio(filename, key, bucket_name='images-bucket'):
    s3 = S3Hook(aws_conn_id='minio_s3')
    s3.load_file(filename=filename, key=key,
                 bucket_name=bucket_name, replace=True)


# Liste des URLs d'images
image_urls = [
    "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000000.jpg",
    "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000000.jpg",
    # Ajoutez d'autres URLs ici
]

# Tâche pour télécharger les images
download_images = PythonOperator(
    task_id='download_images',
    python_callable=lambda: [download_image(
        url, f"/tmp/{os.path.basename(url)}") for url in image_urls],
    dag=dag,
)

# Tâche pour uploader les images vers Minio
upload_images = PythonOperator(
    task_id='upload_images',
    python_callable=lambda: [upload_to_minio(
        f"/tmp/{os.path.basename(url)}", os.path.basename(url)) for url in image_urls],
    dag=dag,
)

# Définir l'ordre des tâches
download_images >> upload_images
