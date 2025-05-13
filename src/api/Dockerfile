# Utilise une image légère avec Python
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires à Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python si tu en as une liste
# Sinon, installe-les directement :
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    boto3 \
    pillow \
    torch \
    torchvision \
    python-multipart


# Copier le script FastAPI
COPY main.py .

# Lancer le serveur FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
