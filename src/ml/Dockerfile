FROM python:3.12-slim

WORKDIR /app

# Installer les outils nécessaires pour Pillow et Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    torch torchvision \
    pandas boto3 pillow requests \
    mlflow

# Copier le code
COPY . .

# Lancer par défaut model_2.py (modifie dans docker-compose si besoin)
CMD ["python", "model_2.py"]
