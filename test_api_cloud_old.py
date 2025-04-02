import sys
import os
import requests

# Ajout du path projet pour pouvoir importer config depuis n’importe où
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ✅ On importe l'URL Cloud Hugging Face
from huggingface_api.config import API_URL_HF

def test_cloud_prediction():
    payload = {
        "data": ["This flight was absolutely horrible!"]
    }

    response = requests.post(API_URL_HF, json=payload)

    # === Vérification du statut HTTP ===
    assert response.status_code == 200

    # === Vérification de la structure du retour ===
    result = response.json()
    assert "data" in result
    assert isinstance(result["data"], list)
    assert len(result["data"]) > 0

    print("✅ Résultat reçu depuis Hugging Face Space:", result["data"])



# commande gitbash test cloud :
# pytest tests/test_api_cloud.py
