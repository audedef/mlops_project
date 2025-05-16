# mlops_project

**Recontextualisation des objectifs du projet**

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

**Lancement du projet**

1. Build du projet avec docker compose qui lance nos services principaux de: airflow, minio, mlflow, redis, api du modèle, et l'interface frontend avec streamlit.
   <pre> docker-compose -f docker-compose.yaml up -d   </pre>
   ![docker build 1](https://github.com/user-attachments/assets/3228efa9-5cd5-4811-85b2-fc4d12a19c49)
2. Construction du bucket "images-bucket" sur minio
   ![Minio1 1](https://github.com/user-attachments/assets/4948ad64-6a9a-41de-8113-78a06ff802f9)

   ![Minio2 1](https://github.com/user-attachments/assets/18f060fa-2265-49a5-857f-e841245fd36f)
3. Lancement du dag qui automatise le workflow
   ![dag 1](https://github.com/user-attachments/assets/87ff28a9-f7bb-42dd-a12e-d399128b33c2)

**Explication et prise en main du projet**
1. utilisation de mlflow pour le suivi des performances de notre modèle
   ![ml_flow 1](https://github.com/user-attachments/assets/90164a7f-cd0c-4a76-8b5b-0cbd8a6fc197)

2. l'interface utilisateur qui permet d'utiliser notre modèle
   ![streamlit2 1](https://github.com/user-attachments/assets/dfa1ecbb-3b5c-4da1-8e68-5ac67f6e3ee0)
![streamlit1 1](https://github.com/user-attachments/assets/09af827b-fd2b-4385-81f2-6fe416c046ef)





   
