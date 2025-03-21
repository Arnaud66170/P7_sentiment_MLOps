# Projet P7 - Analyse de Sentiments & MLOps

## Objectif
Prototype IA pour anticiper les bad buzz sur Twitter pour la compagnie aérienne **Air Paradis**.

## Prérequis :
Python 3.10.11 recomandé.

## Approches Modélisées
1. **Modèle Classique :** Logistic Regression + TF-IDF
2. **Modèle Avancé :** Embeddings (FastText, USE) + LSTM / LightGBM
3. **Modèle BERT :** DistilBERT fine-tuné

## Pipeline MLOps Intégré
- **Tracking expérimentations :** via MLFlow
- **Déploiement API :** FastAPI exposé sur Cloud (Railway / AWS EC2 / HF Spaces)
- **CI/CD :** GitHub Actions (automatisé)
- **Monitoring :** Suivi des erreurs utilisateur, déclenchement d'alertes

## Modèles & Checkpoints
Tous les modèles et résultats intermédiaires sont stockés dans le dossier "models_saved"

## Arborescence

project_root/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   ├── utils.py
│   └── api/
│       └── main.py
├── notebooks/
│   └── P7_main_notebook.ipynb
├── requirements.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
pip install -r requirements.txt


## Lancement API (local)
uvicorn src.api.main:app --reload
