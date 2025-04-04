import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import Trainer
from datasets import ClassLabel
from utils import mlflow_run_safety
import mlflow


# Évaluation classification
@mlflow_run_safety(experiment_name = "P7_sentiment_analysis")
def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize = (6,6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot = True, fmt = "d", cmap = "Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()


@mlflow_run_safety(experiment_name = "P7_sentiment_analysis")
def display_misclassified_tweets(X_test_text, y_true, y_pred, model_name, max_display = 10):
    misclassified_idx = np.where(y_pred != y_true)[0]
    print(f"\n🔍 Tweets mal classifiés par {model_name} :")
    for idx in misclassified_idx[:max_display]:
        print(f"❌ {X_test_text.iloc[idx]} \n   Réel: {y_true[idx]} - Prédit: {y_pred[idx]}\n")
    if len(misclassified_idx) == 0:
        print("✅ Aucune erreur détectée !")


@mlflow_run_safety(experiment_name = "P7_sentiment_analysis")
def classification_report_metrics(y_true, y_pred):
    print("\n📊 Rapport de classification :")
    print(classification_report(y_true, y_pred))


# Évaluation spécifique DistilBERT
@mlflow_run_safety(experiment_name = "P7_sentiment_analysis")
def evaluate_distilbert_model(model, tokenized_dataset, results_path = "../models_saved/distilbert_eval_results.pkl"):
    if os.path.exists(results_path):
        print(f"✅ Résultats d'évaluation déjà disponibles. Chargement...")
        eval_results = joblib.load(results_path)
        print("\n📊 Rapport DistilBERT :")
        print(eval_results['classification_report'])
        return eval_results['accuracy'], eval_results['f1']

    print("📊 Évaluation DistilBERT...")
    from datasets import ClassLabel
    if tokenized_dataset.features['label'].__class__.__name__ != 'ClassLabel':
        num_classes = len(set(tokenized_dataset['label']))
        tokenized_dataset = tokenized_dataset.cast_column('label', ClassLabel(num_classes = num_classes))

    dataset_split = tokenized_dataset.train_test_split(test_size = 0.2, stratify_by_column = 'label')
    test_dataset = dataset_split['test']

    trainer = Trainer(model = model)
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(axis = -1)
    y_true = test_dataset['label']

    report = classification_report(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    eval_results = {
        'accuracy': acc,
        'f1': f1,
        'classification_report': report
    }
    joblib.dump(eval_results, results_path)
    print("✅ Résultats sauvegardés")
    return acc, f1


# Comparaison finale des modèles
@mlflow_run_safety(experiment_name = "P7_sentiment_analysis")
def get_all_model_scores(models_dict, datasets_dict, training_stats=None):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    results = {
        'Modèle': [],
        'Accuracy': [],
        'F1-score': [],
        'Temps d\'entraînement (s)': [],
        'CPU (%)': [],
        'RAM (%)': []
    }

    for model_name, model_obj in models_dict.items():
        # Cas particulier DistilBERT et LSTM (meilleures métriques déjà calculées)
        if model_name in ['distilbert_metrics', 'lstm_metrics']:
            label = 'DistilBERT fine-tuné' if model_name == 'distilbert_metrics' else 'LSTM (config_2)'
            results['Modèle'].append(label)
            results['Accuracy'].append(round(model_obj['accuracy'], 4))
            results['F1-score'].append(round(model_obj['f1'], 4))
        elif model_name == 'vader_metrics':
            results['Modèle'].append('VADER baseline')
            results['Accuracy'].append(round(model_obj['accuracy'], 4))
            results['F1-score'].append(round(model_obj['f1'], 4))
        else:
            # Choix dataset associé
            if model_name == 'logreg':
                data = datasets_dict['tfidf']
                X_test, y_test = data['X_test'], data['y_test']
                y_pred = model_obj.predict(X_test)
            elif model_name == 'rf':
                data = datasets_dict['fasttext']
                X_test, y_test = data['X_test'], data['y_test']
                y_pred = model_obj.predict(X_test)
            elif model_name == 'lstm':
                X_test, y_test = datasets_dict['lstm']
                y_pred = (model_obj.predict(X_test) > 0.5).astype(int).flatten()
            elif model_name == 'lgbm':
                data = datasets_dict['use']
                X_test, y_test = data['X_test'], data['y_test']
                y_pred = (model_obj.predict(X_test) > 0.5).astype(int)
            else:
                continue

            # Sécuriser les types
            if isinstance(y_pred[0], str):
                y_pred = np.array([int(p) for p in y_pred])

            acc = round(accuracy_score(y_test, y_pred), 4)
            f1 = round(f1_score(y_test, y_pred), 4)

            results['Modèle'].append(model_name)
            results['Accuracy'].append(acc)
            results['F1-score'].append(f1)

            # Visualisation
            print(f"\n📊 Résultats pour : {model_name}")
            print(classification_report(y_test, y_pred))
            plt.figure(figsize=(6, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title(f"Matrice de confusion - {model_name}")
            plt.xlabel("Prédictions")
            plt.ylabel("Réel")
            plt.show()

        # ➕ Ajout des stats si disponibles
        if training_stats and model_name in training_stats:
            stats = training_stats[model_name]
            results['Temps d\'entraînement (s)'].append(stats.get('time', 'NA'))
            results['CPU (%)'].append(stats.get('cpu', 'NA'))
            results['RAM (%)'].append(stats.get('ram', 'NA'))
        else:
            results['Temps d\'entraînement (s)'].append('NA')
            results['CPU (%)'].append('NA')
            results['RAM (%)'].append('NA')

    # Résultats finaux
    df_results = pd.DataFrame(results)
    print("\n📊 Comparaison finale des modèles :")
    display(df_results)

    # Enregistrement
    df_results.to_csv("../models_saved/comparaison_resultats.csv", index=False)
    mlflow.log_artifact("../models_saved/comparaison_resultats.csv")

    return df_results
