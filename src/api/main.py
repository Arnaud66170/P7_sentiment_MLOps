from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import fasttext
import re
import emoji
from sklearn.ensemble import RandomForestClassifier
import uvicorn
import os


# Initialisation FastAPI
app = FastAPI(title="Air Paradis - Sentiment Analysis API (FastText + RF)")


# Modèle Pydantic pour recevoir un tweet
class TweetRequest(BaseModel):
    text: str

# Chargement des modèles fastText / RandomForestClassifier
FASTTEXT_MODEL_PATH = "../models_saved/fasttext_model.ftz"
RF_MODEL_PATH = "../models_saved/rf_model.pkl"

# Chargement de FastText supervisé
if not os.path.exists(FASTTEXT_MODEL_PATH):
    raise FileNotFoundError(f"Modèle FastText introuvable à {FASTTEXT_MODEL_PATH}")
fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)


# Chargemant de Random Forest
rf_model = joblib.load(RF_MODEL_PATH)
print("✅ Modèles FastText + RandomForest chargés.")


# Fonction nettoyage (même logique que le notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = emoji.replace_emoji(text, replace = '')
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text

# Endpoint principal
@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    tweet_cleaned = clean_text(request.text)

    # Vectorisation avec FastText
    vector = fasttext_model.get_sentence_vector(tweet_cleaned).reshape(1, -1)

    # Prédiction RF
    prediction = rf_model.predict(vector)[0]

    # Logging si négatif (par ex)
    if prediction == 0:
        log_path = "../logs/misclassified_tweets.log"
        with open(log_path, "a") as f:
            f.write(f"Tweet mal prédit : {request.text}\n")

    return {
        "sentiment": "positif" if prediction == 1 else "négatif",
        "prediction_label": int(prediction)
    }

# Lancement local
if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000, reload = True)
