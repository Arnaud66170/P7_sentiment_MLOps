# Projet P7 - Analyse de Sentiments & MLOps

## Objectif
Prototype IA pour anticiper les bad buzz sur Twitter pour la compagnie aérienne **Air Paradis**.

## Prérequis :
- Python 3.10.11 recommandé.
- Espace disque suffisant pour stocker embeddings et modèles (~3 Go).

## Approches Modélisées
1. **Modèle Classique :** Logistic Regression + TF-IDF
2. **Modèle Avancé :** Embeddings (FastText, USE) + LSTM / LightGBM
3. **Modèle BERT :** DistilBERT fine-tuné

## Pipeline MLOps Intégré
- **Tracking expérimentations :** via MLFlow (lancé directement depuis le notebook)
- **Gestion du Model Registry :** via MLflow (avec backend SQLite local)
- **Déploiement API :** FastAPI exposé sur Cloud (Railway / AWS EC2 / Hugging Face Spaces)
- **CI/CD :** GitHub Actions (automatisé)
- **Monitoring :** Suivi des erreurs utilisateur, déclenchement d'alertes

## Modèles & Checkpoints
Tous les modèles, embeddings et résultats intermédiaires sont stockés dans le dossier **`models_saved/`**.

## Arborescence

project_root/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   ├── utils.py
│   └── api/
│       └── api.py
├── notebooks/
│   └── P7_main_notebook.ipynb
├── requirements.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation des dépendances
pip install -r requirements.txt


## Lancement API (local)
uvicorn src.api.api:app --reload

#  Suivi des Expérimentations avec MLflow

##  Lancement du serveur MLflow local :
- Placez-vous à la racine du projet dans votre terminal :
mlflow ui --backend-store-uri ./mlruns

## Lancement du MLFlow Tracking Server
scripts\launch_mlflow_server.bat

- Ouvrez votre navigateur et accédez à :
http://127.0.0.1:5000

## Contenu visible dans MLflow :

| Onglet          | Contenu                                                                                         |
|----------------|--------------------------------------------------------------------------------------------------|
| **Experiments** | Expérience nommée `Comparaison finale`                                                            |
| **Runs**        | Chaque exécution du notebook génère un run distinct                                               |
| **Metrics**     | Accuracy & F1-score de chaque modèle (par ex : `logreg_accuracy`, `distilbert_f1`, etc.)           |
| **Artifacts**   | Tableau comparatif final sauvegardé : `models_saved/comparaison_resultats.csv`                    |

Vous pourrez ainsi visualiser les performances des modèles, télécharger les résultats et assurer le suivi complet de vos entraînements.