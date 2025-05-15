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

# ----- Logging -----
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
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- MLflow -----
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("classification-dandelion-grass")

# ----- MinIO -----
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

# ----- Dataset -----
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

# ----- Data -----
logger.info(f"📄 Lecture du CSV : {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
logger.info(f"🔍 {len(df)} images détectées.")

# ----- Transformations (avec augmentation) -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

raw_dataset = S3ImageDataset(df, transform=transform)

def custom_collate(batch):
    return [b for b in batch if b is not None]

loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

sample_batch = next(iter(loader), [])
if len(sample_batch) < 2:
    logger.warning("⚠️ Pas assez d’images valides pour entraîner. Entraînement annulé proprement.")
    exit(0)

logger.info(f"✅ DataLoader prêt - batch_size={BATCH_SIZE}")

# ----- Modèle -----
logger.info("🧠 Chargement de ResNet34...")
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(raw_dataset.label_map))
model = model.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----- Entraînement -----
with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("lr", LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs).to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        logger.info(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # ----- Sauvegarde -----
    logger.info("💾 Sauvegarde du modèle...")
    OUTPUT_DIR = os.path.abspath("temp_model")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, MODEL_OUTPUT_PATH)
    torch.save(model.state_dict(), model_path)

    logger.info("☁️ Upload vers MinIO...")
    s3.upload_file(model_path, BUCKET_NAME, f"models/{MODEL_OUTPUT_PATH}")
    logger.info(f"✅ Modèle sauvegardé dans MinIO : models/{MODEL_OUTPUT_PATH}")

    mlflow.pytorch.log_model(model, artifact_path="model")
    logger.info("✅ Modèle loggé dans MLflow.")
