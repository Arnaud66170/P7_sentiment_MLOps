# Image avec FastText précompilé
FROM ghcr.io/facebookresearch/fasttext:latest

# Installer Python packages nécessaires
RUN apt-get update && apt-get install -y \
    python3-pip \
    && pip3 install \
    fastapi uvicorn scikit-learn mlflow python-dotenv gdown emoji pydantic requests

# Créer dossier de travail
WORKDIR /app

# Copier le code source API
COPY ./src /app/src

# Exposer le port
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]




# Option FastTaxt lente
# Étape 1 : Utiliser une image légère Python
# FROM python:3.9-slim

# # Étape 2 : Installer les outils système nécessaires (compilateurs C++ pour fasttext)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     g++ \
#     gcc \
#     libffi-dev \
#     libssl-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Étape 3 : Créer le dossier de travail
# WORKDIR /app

# # Étape 4 : Copier le requirements.txt et installer les dépendances
# COPY ./requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # Étape 5 : Copier le dossier src contenant l'API
# COPY ./src /app/src

# # Étape 6 : Exposer le port utilisé par l'API FastAPI
# EXPOSE 8000

# # Étape 7 : Lancer l'API FastAPI
# CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]





# # 1 - Base image
# FROM python:3.10-slim

# # 2 - Set work directory
# WORKDIR /app

# # 3 - Copy requirements & install
# COPY requirements.txt .

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # 4 - Copy rest of the app
# COPY . .

# # 5 - Expose port
# EXPOSE 8000

# # 6 - Launch command
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
