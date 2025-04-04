# 🧠 Projet P7 - Analyse de Sentiments & Pipeline MLOps

## 🎯 Objectif
Développer un prototype d’**IA de prédiction de sentiment** pour anticiper les bad buzz sur **Twitter**, dans le cadre d’une mission pour la compagnie **Air Paradis**.

---

## 🗂️ Approches Modélisées

| Catégorie             | Modèle(s) utilisé(s)                              |
|----------------------|----------------------------------------------------|
| 🔹 Classique          | TF-IDF + Logistic Regression                      |
| 🔸 Intermédiaire      | FastText + Random Forest / LSTM                   |
| 🔸 Embedding avancé   | USE + LightGBM                                    |
| 🧠 Deep Learning      | LSTM (FastText)                                    |
| 🧠 Transformers       | DistilBERT fine-tuné sur 100k tweets              |

---

## ⚙️ Pipeline MLOps Intégré

- **🧪 Tracking des expérimentations** : `MLflow` (runs automatiques via décorateurs)
- **📦 Model Registry MLflow** : backend local `SQLite`, support REST API
- **🚀 Déploiement API** :
  - `FastAPI` (Railway / AWS EC2)
  - `Gradio UI` (Hugging Face Spaces)
- **🔁 CI/CD** : `GitHub Actions` (push = déploiement automatisé)
- **📉 Monitoring** :
  - Feedback utilisateur intégré
  - Logs des prédictions erronées
  - Déclenchement d’alerte mail si ≥ 3 erreurs en < 5 min

---

## 🧱 Arborescence du Projet

```
P7_sentiment_MLOps/
├── notebooks/
│   └── P7_main_notebook.ipynb
│   └── mlflow_registry_management.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   ├── utils.py
│   └── api/
│       ├── api.py
│       ├── app.py (interface Gradio)
│       └── alert_email.py
├── models_saved/
│   ├── log_reg_model.pkl
│   ├── rf_model.pkl
│   ├── lstm_model.h5
│   ├── distilbert_model/
│   └── comparaison_resultats.csv
├── huggingface_space/
│   ├── app.py
│   └── utils/
├── .github/workflows/
│   └── deploy_railway.yml
│   └── deploy_huggingface.yml
├── tests/
│   └── test_api.py
├── requirements.txt
├── requirements.py
├── README.md
└── .gitignore
```

---

## 📦 Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Lancement de l’API en local

```bash
uvicorn src.api.api:app --reload
```

---

## 📊 Suivi des expérimentations avec MLflow

### Démarrage du serveur MLflow local

```bash
mlflow ui --backend-store-uri ./mlruns
```

Accédez à l’interface sur : http://127.0.0.1:5000

### Contenu visible dans MLflow

| Onglet           | Détails                                                                 |
|------------------|-------------------------------------------------------------------------|
| **Experiments**  | Expérience nommée `P7_sentiment_analysis`                               |
| **Runs**         | Chaque exécution (LogReg, LSTM, etc.) génère un run distinct            |
| **Metrics**      | Accuracy & F1-score (ex : `lstm_f1`, `distilbert_accuracy`, etc.)       |
| **Artifacts**    | Fichiers comme `comparaison_resultats.csv`, matrices, modèles `.pkl/.h5`|

---

## 🌐 Interfaces Utilisateurs

### Gradio (Hugging Face Spaces)

> URL publique : https://huggingface.co/spaces/arnaud66170/P7-airparadis-sentiment

Fonctionnalités principales :
- Test interactif de tweets (copier-coller ou upload CSV)
- Prédiction + Emoji + Coloration dynamique
- Feedback utilisateur ✅ / ❌
- Historique + Graphiques en temps réel
- Export CSV des prédictions
- Alerte e-mail automatisée via microservice FastAPI

---

## ✉️ Alerte e-mail automatisée

> En cas de **≥ 3 erreurs en < 5 minutes**, une alerte est envoyée à l'équipe.

- Service SMTP hébergé sur Railway (`alert_mail_api`)
- Appelé par l'interface Gradio via requête POST sécurisée

---

## 📈 Comparaison des modèles (dernière mise à jour)

| Modèle                    | Accuracy | F1-score | Temps       | Ressources |
|---------------------------|----------|----------|-------------|------------|
| TF-IDF + LogisticRegression | 0.76     | 0.76     | 💨 Rapide    | CPU        |
| FastText + Random Forest   | 0.83     | 0.83     | ⚡ Moyen     | CPU        |
| FastText + LSTM            | 0.835    | 0.835    | ⏱️ 3 min     | CPU        |
| USE + LightGBM             | 0.82     | 0.82     | ⚡ Moyen     | CPU        |
| DistilBERT fine-tuné       | 0.85     | 0.85     | ⏳ Lent      | CPU (RAM++)|

---

## ✍️ Auteur

👨‍💻 **Arnaud Caille**  
Parcours Data Scientist & AI Engineer - OpenClassrooms  
Avril 2025

---

## 📬 Contact

> Pour toute question technique : [via GitHub Issues ou Discussions](https://github.com/Arnaud66170/P7_sentiment_MLOps)