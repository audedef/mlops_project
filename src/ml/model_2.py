import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from urllib.parse import urlparse
import boto3
from botocore.client import Config
import io
import logging
import mlflow
import mlflow.pytorch

# ----- Setup logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🔧 Initialisation de l'entraînement du modèle...")

# ----- CONFIG -----
CSV_PATH = './data/dataset.csv'
BUCKET_NAME = 'images-bucket'
MODEL_OUTPUT_PATH = 'model_state_dict.pth'
MINIO_ENDPOINT = os.environ.get('AWS_S3_ENDPOINT', 'http://minio:9000')
ACCESS_KEY = 'minioadmin'
SECRET_KEY = 'minioadmin'
BATCH_SIZE = 4
EPOCHS = 3

# ----- MLflow Setup -----
mlflow.set_tracking_uri("http://mlflow:5001")  # côté hôte
mlflow.set_experiment("classification-dandelion-grass")

# ----- Connexion MinIO -----
try:
    s3 = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    logger.info(f"✅ Connexion à MinIO via {MINIO_ENDPOINT}")
except Exception as e:
    logger.error(f"❌ Échec de la connexion à MinIO : {e}")
    raise

# ----- Dataset depuis MinIO -----
class S3ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
        self.reverse_map = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parsed = urlparse(row['image_path'])
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        label = self.label_map[row['label']]

        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            img_data = obj['Body'].read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"[SKIP] Erreur sur {key} : {e}")
            return None

# ----- Lecture du dataset -----
logger.info(f"📄 Lecture du fichier CSV : {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
logger.info(f"🔍 {len(df)} images listées dans le CSV.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

raw_dataset = S3ImageDataset(df, transform=transform)

def custom_collate(batch):
    return [b for b in batch if b is not None]

loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

# Vérification
sample_batch = next(iter(loader), [])
if len(sample_batch) < 2:
    logger.error("❌ Pas assez d’images valides pour entraîner.")
    raise RuntimeError("Échec : pas assez d’images valides.")

logger.info(f"✅ DataLoader prêt avec batch_size={BATCH_SIZE}")

# ----- Modèle -----
logger.info("🧠 Chargement de ResNet34...")
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(raw_dataset.label_map))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----- Entraînement avec MLflow -----
with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in loader:
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs)
            labels = torch.tensor(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
        mlflow.log_metric("loss", total_loss, step=epoch)

    # ----- Sauvegarde du modèle -----
    logger.info("💾 Sauvegarde du modèle localement...")
    OUTPUT_DIR = os.path.abspath("temp_model")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, MODEL_OUTPUT_PATH)
    torch.save(model.state_dict(), model_path)

    logger.info("☁️ Upload du modèle dans MinIO...")
    s3.upload_file(model_path, BUCKET_NAME, f"models/{MODEL_OUTPUT_PATH}")
    logger.info(f"✅ Modèle sauvegardé dans MinIO : models/{MODEL_OUTPUT_PATH}")

    # ----- Log du modèle dans MLflow -----
    mlflow.pytorch.log_model(model, artifact_path="model")
    logger.info("✅ Modèle loggé dans MLflow.")
