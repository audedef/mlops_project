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

# ----- CONFIG -----
CSV_PATH = './data/dataset.csv'
BUCKET_NAME = 'images-bucket'
MODEL_OUTPUT_PATH = 'model_state_dict.pth'
MINIO_ENDPOINT = 'http://localhost:9000'
ACCESS_KEY = 'minioadmin'
SECRET_KEY = 'minioadmin'
BATCH_SIZE = 4
EPOCHS = 3

print("🔧 Initialisation du client MinIO...")
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

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
            print(f"[ERREUR] Image ignorée : {key} → {e}")
            return None  # DataLoader filtrera avec collate_fn

# ----- Lecture du dataset -----
print(f"📄 Lecture du fichier CSV : {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"🔍 {len(df)} entrées trouvées dans le CSV.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

print("📥 Chargement des images depuis MinIO...")
raw_dataset = S3ImageDataset(df, transform=transform)

# ----- DataLoader avec filtrage à la volée -----
def custom_collate(batch):
    return [b for b in batch if b is not None]

loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

# Vérifier que le DataLoader contient des données
sample_batch = next(iter(loader), [])
if len(sample_batch) < 2:
    print("❌ Pas assez d’images valides pour entraîner.")
    exit()
print(f"✅ DataLoader prêt avec batch_size={BATCH_SIZE}")

# ----- Modèle -----
print("🧠 Chargement du modèle ResNet34...")
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(raw_dataset.label_map))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----- Entraînement -----
print("🚀 Début de l'entraînement...")
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
    print(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ----- Sauvegarde du modèle -----
print("💾 Sauvegarde du modèle localement...")
os.makedirs("temp_model", exist_ok=True)
torch.save(model.state_dict(), f"temp_model/{MODEL_OUTPUT_PATH}")

print("☁️ Upload du modèle dans MinIO...")
s3.upload_file(f"temp_model/{MODEL_OUTPUT_PATH}", BUCKET_NAME, f"models/{MODEL_OUTPUT_PATH}")
print(f"✅ Modèle sauvegardé dans MinIO : models/{MODEL_OUTPUT_PATH}")
