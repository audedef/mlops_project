# mlops_project



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
