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


**Pré-requis Airflow**

Installation d'Airflow :
Apache Airflow est installé via une image Docker : docker-compose up --build
Les dépendances Python sont intégrées à l'image et seront téléchargées avec le conteneur ainsi construit.

Connexions Airflow :
Il faut configurer une connexion Minio S3 dans Airflow via l'UI d'Airflow, onglet Connections, pour permettre le téléchargement des images vers le bucket S3.

WebUI Airflow : http://127.0.0.1:8080


**Pré-requis Minio S3**

Création bucket :
Se connecter au service S3 avec les login configurés dans Airflow. Et créer un bucket pour y télécharger les images et les paramètres du modèle ML.
Le nom du bucket doit être "images-bucket"

WebUI Minio : http://127.0.0.1:9001  


**Modèle ML de classification binaire**

Pour tester le modèle dans le conteneur :
`docker-compose restart ml-train`

Pour vérifier les logs du modèle :
`docker-compose logs -f ml-train`