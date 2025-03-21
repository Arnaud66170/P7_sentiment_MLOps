from requirements import *

def suivi_temps_ressources(start_time, model_name, phase="Fin"):
    end_time = time.time()
    elapsed = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    print(f"\n⏱️ [{model_name}] - {phase} : {elapsed:.2f} sec | CPU: {cpu_usage}% | RAM: {ram_usage}%")

def save_model_if_not_exists(model, filename):
    if os.path.exists(filename):
        print(f"✅ Modèle déjà sauvegardé : {filename}. Chargement...")
        return joblib.load(filename)
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé sous {filename}")
    return model

def afficher_structure_dossier(chemin_base, niveau=0, max_niveaux=3):
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

def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"✅ {filename} sauvegardé avec succès.")

def load_matrix(filename):
    with open(filename, 'rb') as f:
        print(f"✅ Chargement de {filename}...")
        return pickle.load(f)