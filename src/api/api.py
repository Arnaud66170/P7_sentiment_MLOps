from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import fasttext
import re
import emoji
import uvicorn
import os
import mlflow
from mlflow import sklearn
from dotenv import load_dotenv
import requests


# 1 - Initialisation

print(f"ENV VAR FASTTEXT_MODEL_URL: {os.getenv('FASTTEXT_MODEL_URL')}")
print(f"ENV VAR MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
print(f"ENV VAR LOG_PATH: {os.getenv('LOG_PATH')}")


# 1.1 - Chargement des variables d'environnement
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "random_forest_sentiment")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
FASTTEXT_MODEL_URL = os.getenv("FASTTEXT_MODEL_URL")
FASTTEXT_LOCAL_PATH = os.getenv("FASTTEXT_LOCAL_PATH", "./models_saved/fasttext_model.ftz")
LOG_PATH = os.getenv("LOG_PATH", "../logs/misclassified_tweets.log")

print("DEBUG - Variables env :")
print(f"FASTTEXT_MODEL_URL: {os.getenv('FASTTEXT_MODEL_URL')}")

app = FastAPI(title="Air Paradis - Sentiment Analysis API (FastText + RF)")


#  2 - Classe d'entrée

class TweetRequest(BaseModel):
    text: str


# 3 - Chargement modèles


# 3.1 - Chargement FastText supervisé (distant ou local)
if not os.path.exists(FASTTEXT_LOCAL_PATH):
    print(f"📥 Téléchargement du modèle FastText depuis {FASTTEXT_MODEL_URL}...")
    r = requests.get(FASTTEXT_MODEL_URL)
    with open(FASTTEXT_LOCAL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Modèle FastText téléchargé.")

fasttext_model = fasttext.load_model(FASTTEXT_LOCAL_PATH)
print("✅ Modèle FastText chargé.")

# 3.2 - Chargement Random Forest depuis MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"📡 Connexion MLflow URI : {MLFLOW_TRACKING_URI}")
rf_model = mlflow.sklearn.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}")
print(f"✅ Modèle RandomForest chargé depuis MLflow ({MLFLOW_MODEL_NAME} - {MLFLOW_MODEL_STAGE}).")



# 4 - Nettoyage texte

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text


# 5 - Endpoint prédiction

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    tweet_cleaned = clean_text(request.text)

    # Vectorisation
    vector = fasttext_model.get_sentence_vector(tweet_cleaned).reshape(1, -1)

    # Prédiction
    prediction = rf_model.predict(vector)[0]

    # Logging erreurs
    if prediction == 0:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"Tweet mal prédit : {request.text}\n")

    return {
        "sentiment": "positif" if prediction == 1 else "négatif",
        "prediction_label": int(prediction)
    }


# 6 - Lancement local

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)