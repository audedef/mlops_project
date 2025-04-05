# mlops_project

**Objectifs du projet**

1. Extraire et prétraiter les données : télécharger les images depuis les URLs, les nettoyer et les préparer pour l'entraînement.

2. Construire un modèle de classification : développer un modèle de deep learning permettant de classer les images en deux catégories (dandelion vs grass). Utilisez la techno la plus adaptée à vos besoins : FastAI, PyTorch, Tensorflow, etc.

3. Stocker le modèle sur AWS S3 (Minio) : une fois entraîné, le modèle devra être sauvegardé dans un bucket S3.

4. Créer un pipeline de réentraînement avec Apache Airflow : mettre en place un
pipeline qui récupère les nouvelles données, met à jour le modèle et le redéploie.

5. Suivre les modèles et les expériences avec MLFlow : utiliser MLFlow pour enregistrer les performances et suivre les versions du modèle.

6. Développer une API : construire une API permettant d'envoyer une image et de
recevoir une prédiction. Utilisez la techno la plus adaptée à vos besoins : FastAPI, Flask, KServe, Torch Serve, etc.

7. Créer une WebApp pour interagir avec le modèle : développer une interface
utilisateur simple pour visualiser les résultats des prédictions. Utilisez la techno la plus adaptée à vos besoins : Gradio, Streamlit, Voila, etc.

8. Dockeriser votre API et déployer là sur un cluster Kubernetes en local (Docker Desktop ou MiniKube). Utilisez la CI/CD de GitHub Actions pour celà.

9. Versionner et documenter le projet sur GitHub : tout le projet devra être hébergé sur GitHub, avec une structure de fichiers propre et une documentation claire.

10. Ajouter du monitoring pour visualiser l’ensemble des métriques que vous souhaitez monitorer (ex: airflow, API, performances du modèle, etc.). Utilisez la techno la plus adaptée à vos besoins : Elasticsearch + Kibana, Prometheus + Grafana, etc.

11. Vous pouvez ajouter des tests de montée en charge (ex: avec Locust)

12. On souhaite maintenant faire de l'Entraînement Continue (CT). Ajouter un ou plusieurs DAG Airflow avec des triggers que vous définirez (nouvelle données, entrainement hebdomadaire, performances du modèle en baisse, etc.) pour réentraîner et déployer automatiquement un nouveau modèle.


**Pré-requis côté base de données PostgreSQL**

PostgreSQL doit être installé et en cours d'exécution sur le serveur ou la machine locale.
Il faut créer au préalable une base de données dans laquelle la table plants_data sera créée.


**Pré-requis côté Airflow**

Installation d'Airflow :
Apache Airflow doit être installé et configuré sur la machine ou serveur.
Installer avec pip : pip install apache-airflow.

Connexions Airflow :
Il faut configurer une connexion PostgreSQL dans Airflow via l'interface utilisateur d'Airflow ou en modifiant le fichier de configuration airflow.cfg.
Ensuite il faut configurer une connexion AWS S3 dans Airflow pour permettre le téléchargement des images vers le bucket S3.

Dépendances Python :
Il faut s'assurer que les bibliothèques Python nécessaires sont installées, notamment psycopg2 pour la connexion à PostgreSQL et boto3 pour l'interaction avec AWS S3.
Les installer avec pip : pip install psycopg2-binary boto3.

Configuration des DAGs :
Placer le fichier DAG dans le répertoire dags d'Airflow pour qu'il soit détecté et exécuté par le planificateur Airflow.

Variables d'environnement et permissions AWS :

Les informations d'identification AWS (clé d'accès et clé secrète) doivent être configurées dans Airflow ou dans les variables d'environnement de votre système. Et vérifier que le compte AWS a les permissions nécessaires pour écrire dans le bucket S3 spécifié.
