from fastapi import FastAPI
from pydantic import BaseModel
from requirements import *

app = FastAPI()

# Charger modèle avancé (placeholder - à ajuster selon modèle final)
model = joblib.load("lgbm_model.txt")  # Exemple : LightGBM

class TweetRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    tweet_text = request.text
    # Ici : preprocessing minimal + vectorisation simple pour l'exemple
    vectorizer = TfidfVectorizer(max_features=2000)
    transformed = vectorizer.fit_transform([tweet_text])
    prediction = model.predict(transformed)
    return {"tweet": tweet_text, "prediction": int(prediction[0])}

@app.post("/report_error")
def report_error(request: TweetRequest):
    # Log ou système d'alerte ici (à intégrer : monitoring futur)
    print(f"⚠️ Erreur signalée sur tweet : {request.text}")
    return {"status": "error reported"}
