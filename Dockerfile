# Dockerfile
# 1 - Utilisation d'une image Python légère
FROM python:3.9-slim

# 2 - Définition du dossier de travail
WORKDIR /app

# 3 - Copie des fichiers nécessaires
COPY ./src /app/src
COPY ./models_saved /app/models_saved
COPY ./requirements.txt /app/requirements.txt

# 4 - Installation des dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5 - Exposition du port API
EXPOSE 8000

# 6 - Commande de lancement API
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]






# PARAMETRES RAILWAYS
# # 1 - Image de base
# FROM python:3.10-slim

# # 2 - Fichier de travail
# WORKDIR /app

# # 3 - Installation dépendances système
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # 4 - Copie des insattl et requirements
# COPY requirements.txt .

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# RUN pip install gdown

# # 5 - Copie du reste de l'app
# COPY . .

# # 6 - Exposition du port
# EXPOSE 8000

# # 7 - Lancement de commande
# CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]

