FROM python:3.9-slim

ENV AIRFLOW_HOME=/opt/airflow

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    git \
    curl \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Installer Apache Airflow + dépendances ML + base de données
RUN pip install apache-airflow==2.6.3 \
    torch==2.0.1 torchvision==0.15.2 \
    boto3 pandas pillow mlflow psycopg2-binary \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.3/constraints-3.9.txt"

# Créer les dossiers Airflow
RUN mkdir -p $AIRFLOW_HOME/dags

WORKDIR $AIRFLOW_HOME

CMD ["airflow", "webserver"]
