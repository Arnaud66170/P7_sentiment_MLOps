from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import re
import emoji
import uvicorn
import os

# 1 - Initialisation

LOG_PATH = os.getenv("LOG_PATH", "../logs/misclassified_tweets.log")

app = FastAPI(title="Air Paradis - Sentiment Analysis API (TF-IDF + LogReg)")


# 2 - Classe d'entrée

class TweetRequest(BaseModel):
    text: str


# 3 - Chargement modèles

# 3.1 - Chargement TF-IDF Vectorizer et Logistic Regression
print("📥 Chargement du vectorizer TF-IDF...")
vectorizer = joblib.load("models_saved/tfidf_vectorizer.pkl")
print("✅ Vectorizer chargé.")

print("📥 Chargement du modèle Logistic Regression...")
logreg_model = joblib.load("models_saved/log_reg_model.pkl")
print("✅ Modèle LogReg chargé.")


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
    vector = vectorizer.transform([tweet_cleaned])

    # Prédiction
    prediction = logreg_model.predict(vector)[0]

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
