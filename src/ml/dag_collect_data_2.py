import os
import requests
import csv
from PIL import Image
import boto3
from botocore.client import Config

# --- Étape 1 : configuration ---
repos = {
    "dandelion": "https://api.github.com/repos/btphan95/greenr-airflow/contents/data/dandelion",
    "grass": "https://api.github.com/repos/btphan95/greenr-airflow/contents/data/grass"
}
bucket_name = "images-bucket"
prefix = "pipeline_test/"
endpoint_url = "http://localhost:9000"
access_key = "minioadmin"
secret_key = "minioadmin"

# --- Connexion MinIO ---
s3 = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# --- Préparer les dossiers ---
os.makedirs("temp_images", exist_ok=True)
os.makedirs("data", exist_ok=True)
dataset_entries = []

# --- Télécharger, vérifier, uploader ---
for label, api_url in repos.items():
    response = requests.get(api_url)
    if response.status_code == 200:
        images = [f for f in response.json() if f['name'].endswith('.jpg')]
        for f in images[:50]:
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

            except Exception as e:
                print(f"❌ Erreur sur {filename} : {e}")

# --- Écrire le fichier CSV ---
with open('./data/dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    writer.writerows(dataset_entries)

print(f"✅ Dataset prêt avec {len(dataset_entries)} images.")
