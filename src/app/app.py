# src/webapp/app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import io
import os
from torchvision.models import resnet34


MODEL_PATH = os.path.join(os.path.dirname(__file__), "temp_model/model_state_dict.pth")

# Chargement du modèle
model = resnet34(weights=None)

model.fc = torch.nn.Linear(model.fc.in_features, 2)  # adapter au nb de classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

st.title("🌿 Classification Plantes : Dandelion ou Grass")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Image chargée', use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()

    label_map = {0: "Dandelion", 1: "Grass"}
    st.markdown(f"### 🧠 Prédiction : **{label_map[prediction]}**")
