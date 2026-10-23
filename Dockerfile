# 1. Image de base officielle Python légère
FROM python:3.12-slim

# 2. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 3. Empêche Python d'écrire des fichiers .pyc et force l'affichage direct des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. Copie et installation des dépendances en premier (optimisation du cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du code applicatif et de l'artefact de modèle
COPY app/ ./app
COPY models/ ./models
COPY src/ ./src

# 6. Exposition du port de l'API
EXPOSE 8000

# 7. Commande d'exécution du serveur Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]