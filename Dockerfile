# syntax=docker/dockerfile:1
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# URI de tracking (non secrète) passée au build
ARG MLFLOW_TRACKING_URI

# Téléchargement du dernier modèle PENDANT LE BUILD.
# Le token vient d'un secret BuildKit : il n'est écrit dans aucune couche de l'image.
COPY download_model.py .
RUN --mount=type=secret,id=dagshub_token \
    DAGSHUB_TOKEN="$(cat /run/secrets/dagshub_token)" python download_model.py

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]