import os
import requests
import csv
import logging
from PIL import Image
import boto3
from botocore.client import Config

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("📦 Début de la collecte des données...")

# --- Étape 1 : configuration ---
repos = {
    "dandelion": "https://api.github.com/repos/btphan95/greenr-airflow/contents/data/dandelion",
    "grass": "https://api.github.com/repos/btphan95/greenr-airflow/contents/data/grass"
}
bucket_name = "images-bucket"
prefix = "pipeline_test/"

endpoint_url = os.environ.get("AWS_S3_ENDPOINT", "http://minio:9000")
access_key = "minioadmin"
secret_key = "minioadmin"

# --- Connexion MinIO ---
try:
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    logger.info(f"✅ Connexion à MinIO établie via {endpoint_url}")
except Exception as e:
    logger.error(f"❌ Erreur de connexion MinIO : {e}")
    raise

# --- Préparer les dossiers ---
os.makedirs("temp_images", exist_ok=True)
os.makedirs("data", exist_ok=True)
dataset_entries = []

# --- Télécharger, vérifier, uploader ---
for label, api_url in repos.items():
    response = requests.get(api_url)
    if response.status_code == 200:
        images = [f for f in response.json() if f['name'].endswith('.jpg')]
        for f in images[:150]:
            filename = f"{label}_{f['name']}"
            local_path = os.path.join("temp_images", filename)
            s3_key = f"{prefix}{filename}"
            image_url = f['download_url']

            try:
                img_data = requests.get(image_url).content
                with open(local_path, 'wb') as out:
                    out.write(img_data)

                # Vérification d'intégrité
                with Image.open(local_path) as img:
                    img.verify()

                # Upload vers MinIO
                s3.upload_file(local_path, bucket_name, s3_key)

                # Ajouter à dataset.csv
                dataset_entries.append([f"s3://{bucket_name}/{s3_key}", label])
                logger.info(f"📤 Uploadé : {filename}")

            except Exception as e:
                logger.error(f"❌ Erreur sur {filename} : {e}")

# --- Écrire le fichier CSV ---
if dataset_entries:
    with open('./data/dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(dataset_entries)
    logger.info(f"✅ Dataset prêt avec {len(dataset_entries)} images.")
else:
    logger.error("❌ Aucun fichier valide n'a été traité.")
    raise RuntimeError("Aucune image n’a pu être ajoutée au dataset.")
