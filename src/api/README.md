# Air Paradis - API Analyse de Sentiment

Bienvenue dans le projet **API FastAPI - Analyse de sentiment pour Air Paradis** !  
Cette API permet de prédire le sentiment (positif/négatif) d’un tweet en utilisant un modèle FastText + Random Forest, avec un déploiement orienté **MLOps** via MLflow.

---

## Fonctionnalités

- Prédiction du sentiment d’un tweet (positif/négatif).
- Modèle RandomForest chargé dynamiquement depuis le **MLflow Model Registry**.
- Modèle FastText chargé localement ou via un lien distant.
- Logging automatique des tweets mal classifiés.
- Variables d'environnement centralisées via `.env`.
- Déploiement prêt pour Cloud (Railway, AWS EC2...) via Docker.

## ⚙️ Configuration

**Variables d'environnement :**

Créer un fichier `.env` à la racine avec les variables suivantes :

MLFLOW_TRACKING_URI=http://localhost:5000 
MLFLOW_MODEL_NAME=random_forest_sentiment 
MLFLOW_MODEL_STAGE=Production 
FASTTEXT_MODEL_URL=https://lien_vers_fasttext_model.ftz 
FASTTEXT_LOCAL_PATH=../models_saved/fasttext_model.ftz 
LOG_PATH=../logs/misclassified_tweets.log

## Lancement local :

pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload