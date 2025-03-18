from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.hooks.S3_hook import S3Hook
import requests
from datetime import datetime

# Configuration du DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG('extract_plant_images', default_args=default_args,
          schedule_interval='@daily')


def extract_and_upload_images():
    # Connexion à PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    connection = pg_hook.get_conn()
    cursor = connection.cursor()

    # Récupération des données
    cursor.execute(
        "SELECT id, url_source, label FROM plants_data WHERE url_s3 IS NULL")
    rows = cursor.fetchall()

    # Connexion à S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = 'your-s3-bucket-name'

    for row in rows:
        image_id, url_source, label = row
        response = requests.get(url_source)

        if response.status_code == 200:
            s3_key = f"plants/{label}/{image_id}.jpg"
            s3_hook.load_bytes(response.content, key=s3_key,
                               bucket_name=bucket_name)

            # Mise à jour de l'URL S3 dans la base de données
            cursor.execute("UPDATE plants_data SET url_s3 = %s WHERE id = %s",
                           (f"s3://{bucket_name}/{s3_key}", image_id))
            connection.commit()

    cursor.close()
    connection.close()


# Tâche d'extraction et de téléchargement des images
extract_and_upload = PythonOperator(
    task_id='extract_and_upload_images',
    python_callable=extract_and_upload_images,
    dag=dag,
)

extract_and_upload
