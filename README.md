# mlops_project

## Objectifs du projet

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

## 🔧 Lancement du projet

1. Lancez l'ensemble des services avec Docker Compose :
   <pre> docker-compose -f docker-compose.yaml up -d   </pre>
   Services inclus :

   - Airflow : Orchestration des workflows

   - MinIO : Stockage d'objets pour les données et du modèle entraîné 

   - MLflow : Suivi des expériences ML 📊

   - Redis : Cache et système de message

   - API : Service d'inférence du modèle (FastAPI)

   - Streamlit : Interface utilisateur 🖥️
   ![docker build 1](https://github.com/user-attachments/assets/3228efa9-5cd5-4811-85b2-fc4d12a19c49)

   NB : Nous avons fait le choix de ne pas utiliser de service MySQL pour stocker les données tabulaires mais plutôt d'automatiser avec Airflow la génération et mise à jour d'un fichier csv dans le dossier data du projet Git. Compte tenu de la taille du dataset (400 lignes), cette méthode nous paraissait plus directe et efficace.

2. Construction du bucket "images-bucket" sur minio
   - Accédez à l'interface MinIO (http://localhost:9001)

   - Créez un bucket nommé images-bucket

   - Configurez les permissions d'accès
   #ajouter le screen du dossier ou ce trouve le modèle


   ![Minio1 1](https://github.com/user-attachments/assets/4948ad64-6a9a-41de-8113-78a06ff802f9)

   ![Minio2 1](https://github.com/user-attachments/assets/18f060fa-2265-49a5-857f-e841245fd36f)
   
3. Exécution du Workflow Airflow
   - Accédez à l'interface Airflow (http://localhost:8080)

   - Activez le DAG principal

   - Ps: Il faut configurer une connexion Minio S3 dans Airflow via l'UI d'Airflow, onglet Connections, pour permettre le téléchargement des images vers le bucket S3.

   ![dag 1](https://github.com/user-attachments/assets/87ff28a9-f7bb-42dd-a12e-d399128b33c2)

**📚 Guide d'Utilisation**
1. Accédez à MLflow (http://localhost:5001) pour :
   - Comparer les performances des modèles

   - Visualiser les métriques

   - Gérer les versions des modèles
     #ajouter le screen de differentes etapes du modèles (plusieurs train)
   ![ml_flow 1](https://github.com/user-attachments/assets/90164a7f-cd0c-4a76-8b5b-0cbd8a6fc197)

2. l'interface utilisateur qui permet d'utiliser notre modèle 🖥️
   - Accédez à l'application (http://localhost:8501)

   - Chargez vos données

   - Visualisez les prédictions
     
![streamlit2 1](https://github.com/user-attachments/assets/dfa1ecbb-3b5c-4da1-8e68-5ac67f6e3ee0)
![streamlit1 1](https://github.com/user-attachments/assets/09af827b-fd2b-4385-81f2-6fe416c046ef)

## 🌿 Gestion des Branches Git

### Stratégie de Branches
Nous utilisons un workflow Git avec deux branches principales (main et dev) et des sous branches :

main → Branche stable (production)

└── dev → Branche d'intégration (développement)

├── sous branche 1

├── sous branche 2

├── sous branche 3

└── sous branche 4

## 🔄 Intégration Continue avec GitHub Actions
### Workflow d'Exécution
Notre pipeline CI s'exécute automatiquement à chaque push via GitHub Actions :
![image](https://github.com/user-attachments/assets/c671cdd5-4adc-462e-ae5c-4dc399aea8af)

## Choix du modèle
L'objectif du projet était d'utiliser du deep learning, bien qu'une simple régression logistique aurait également eu de très bonnes performances pour cette tâche.
Nous avons choisi de le développer avec Pytorch un petit réseaux de neurone à 34 couches (resnet34).


## Difficultés
1/ Nous avons commencé par construire les services Docker sur mac os mais avons eu de nombreux problèmes de comptabilité qui nous ont obligé de revoir toute une partie du code pour les rendre compatible sur windows / linux. Cependant une fois fonctionnel sur windows/linux, ils ne l'étaient plus sur mac os. Nous n'avons pas réussi à aboutir à une version fonctionnelle sur les deux os. De plus la branche dev a trop dévié de la branche main donc nous avons réalisé un git reset --hard dev, perdant ainsi les insights de la branche main. Ci-dessous une sauvegarde des commits. 
<img width="1261" alt="image" src="https://github.com/user-attachments/assets/8a3ad146-0de2-46aa-ac44-744941662fe4" />


2/ Le déploiement Kubernetes (en local) ne fonctionne que pour 3 services : minio, mlflow et postgres.




   
