from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import boto3
from botocore.client import Config

app = FastAPI()

# ----- Configuration -----
BUCKET_NAME = 'images-bucket'
MODEL_KEY = 'models/model_state_dict.pth'
MINIO_ENDPOINT = 'http://minio:9000'  # ou localhost si lancé sans Docker
ACCESS_KEY = 'minioadmin'
SECRET_KEY = 'minioadmin'

# ----- Initialisation MinIO -----
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# ----- Charger le modèle depuis minio, on l'a enregistré avec le script model_2 dans minio-----
print("📦 Téléchargement du modèle depuis MinIO...")
obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
model_data = obj['Body'].read()

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes
model.load_state_dict(torch.load(io.BytesIO(model_data), map_location=torch.device('cpu')))
model.eval()
print("✅ Modèle chargé et prêt.")

# Mapping label → nom
id_to_label = {0: 'dandelion', 1: 'grass'}

# ----- Prétraitement de l'image déposée sur la webapp -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----- Endpoint de test -----
@app.get("/")
def read_root():
    return {"message": "API modèle opérationnelle 🎯"}

# ----- Endpoint de prédiction -----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = id_to_label[pred]
        return {"prediction": label}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
