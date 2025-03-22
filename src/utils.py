import os
import time
import pickle
import joblib
import psutil
import mlflow
import subprocess
from functools import wraps


# Décorateur de sécurisation des runs MLflow
def mlflow_run_safety(experiment_name="P7_sentiment_analysis"):
    def decorator(f):
        def wrapped_function(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            if mlflow.active_run() is not None:
                mlflow.end_run()
            with mlflow.start_run():
                print(f"🚀 Nouveau run démarré : {mlflow.active_run().info.run_id}")
                result = f(*args, **kwargs)
            print(f"✅ Run terminé.")  # ✅ Simplifié ici !
            return result
        return wrapped_function
    return decorator



@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def suivi_temps_ressources(start_time, model_name, phase="Fin"):
    end_time = time.time()
    elapsed = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    print(f"\n⏱️ [{model_name}] - {phase} : {elapsed:.2f} sec | CPU: {cpu_usage}% | RAM: {ram_usage}%")


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def save_model_if_not_exists(model, filename):
    if os.path.exists(filename):
        print(f"✅ Modèle déjà sauvegardé : {filename}. Chargement...")
        return joblib.load(filename)
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé sous {filename}")
    return model


def afficher_structure_dossier(chemin_base, niveau = 0, max_niveaux = 3):
    if niveau > max_niveaux:
        return
    indent = "│   " * (niveau - 1) + ("├── " if niveau > 0 else "")
    try:
        elements = sorted(os.listdir(chemin_base))
    except PermissionError:
        print(indent + "🔒 [Accès refusé]")
        return
    for i, element in enumerate(elements):
        chemin_complet = os.path.join(chemin_base, element)
        is_last = (i == len(elements) - 1)
        prefix = "└── " if is_last else "├── "
        print(indent + prefix + element)
        if os.path.isdir(chemin_complet):
            afficher_structure_dossier(chemin_complet, niveau + 1, max_niveaux)


@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"✅ {filename} sauvegardé avec succès.")

@mlflow_run_safety(experiment_name="P7_sentiment_analysis")
def load_matrix(filename):
    with open(filename, 'rb') as f:
        print(f"✅ Chargement de {filename}...")
        return pickle.load(f)


def launch_mlflow_ui(port = 5000):
    try:
        subprocess.Popen(["mlflow", "ui", "--port", str(port)], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        print(f"🚀 MLFlow UI lancé sur http://127.0.0.1:{port}")
    except Exception as e:
        print(f"❌ Impossible de démarrer MLFlow UI : {e}")

