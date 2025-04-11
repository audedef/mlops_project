FROM python:3.12-slim

WORKDIR /app

# Installer les outils de build essentiels et les dépendances pour Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances 
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier tout le code source de votre projet
COPY ./ /app/src/ml/

# Définir la commande à exécuter lorsque le conteneur démarre
CMD ["python", "/app/src/ml/model.py"]