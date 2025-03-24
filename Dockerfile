# Dockerfile
# parametres AWS EC2

# 1. Choix de l'image de base
FROM python:3.9-slim

# 2. Définir le dossier de travail
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY ./requirements.txt /app/requirements.txt
COPY ./api.py /app/api.py
COPY ./models_saved /app/models_saved
COPY ./src /app/src

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5. Exposer le port de l'API
EXPOSE 8000

# 6. Lancer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]





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

