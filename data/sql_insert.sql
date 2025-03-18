-- Création de la table plants_data

CREATE TABLE IF NOT EXISTS plants_data (
    id SERIAL PRIMARY KEY,
    url_source VARCHAR(255) NOT NULL,
    url_s3 VARCHAR(255),
    label VARCHAR(50) NOT NULL
);

-- Insertion des métadonnées pour les images de pissenlits
INSERT INTO plants_data (url_source, label)
SELECT
    CONCAT('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/', LPAD(seq::text, 8, '0'), '.jpg'),
    'dandelion'
FROM generate_series(0, 199) AS seq;

-- Insertion des métadonnées pour les images d'herbe
INSERT INTO plants_data (url_source, label)
SELECT
    CONCAT('https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/', LPAD(seq::text, 8, '0'), '.jpg'),
    'grass'
FROM generate_series(0, 199) AS seq;
