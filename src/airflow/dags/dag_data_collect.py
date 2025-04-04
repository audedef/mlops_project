from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import requests
import os
import csv

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
    description='A DAG to download and store images from GitHub repositories',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=32
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


# Fonction pour lister les images dans un répertoire GitHub
def list_github_images(repo_url):
    response = requests.get(repo_url)
    if response.status_code == 200:
        return [item['download_url'] for item in response.json() if item['name'].endswith('.jpg')]
    else:
        return []


# Fonction pour vérifier si une image existe dans le bucket S3
def check_image_in_s3(key, bucket_name='images-bucket'):
    s3 = S3Hook(aws_conn_id='minio_s3')
    return s3.check_for_key(key, bucket_name)


# Fonction pour générer le fichier CSV
def generate_csv(image_data, output_path='../../data/dataset.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        writer.writerows(image_data)


# Tâche pour télécharger, uploader les images et générer le fichier CSV
def download_and_upload_images():
    # Répertoires GitHub contenant les images
    repos = [
        ("https://api.github.com/repos/btphan95/greenr-airflow/contents/data/dandelion", "dandelion"),
        ("https://api.github.com/repos/btphan95/greenr-airflow/contents/data/grass", "grass")
    ]

    image_data = []

    for repo, label in repos:
        image_urls = list_github_images(repo)
        for url in image_urls:
            image_name = os.path.basename(url)
            new_image_name = f"{label}_{image_name}"
            local_path = f"/tmp/{new_image_name}"

            # Vérifier si l'image existe déjà dans le bucket S3
            if not check_image_in_s3(new_image_name):
                # Télécharger l'image
                download_image(url, local_path)
                # Uploader l'image vers Minio
                upload_to_minio(local_path, new_image_name)

            # Ajouter l'image et son label au fichier CSV
            image_data.append((f"s3://images-bucket/{new_image_name}", label))

    # Générer le fichier CSV
    generate_csv(image_data)


# Tâche pour télécharger, uploader les images et générer le fichier CSV
download_and_upload_task = PythonOperator(
    task_id='download_and_upload_images',
    python_callable=download_and_upload_images,
    dag=dag,
)

# Définir l'ordre des tâches
download_and_upload_task
