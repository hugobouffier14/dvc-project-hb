import os
import shutil
import sys
import tempfile

MODEL_NAME = "digits-rf"
OUTPUT_DIR = "model"

token = os.environ.get("DAGSHUB_TOKEN")
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

if not token or not tracking_uri:
    sys.exit("ERREUR : DAGSHUB_TOKEN et MLFLOW_TRACKING_URI doivent être définis.")

# DagsHub accepte le token comme identifiant ET comme mot de passe
os.environ["MLFLOW_TRACKING_USERNAME"] = token
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

import mlflow  # noqa: E402  (import après la config des variables d'environnement)
from mlflow.tracking import MlflowClient  # noqa: E402

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    sys.exit(f"ERREUR : aucune version trouvée pour le modèle '{MODEL_NAME}'.")

latest = max(versions, key=lambda v: int(v.version))
print(f"Dernière version : {latest.version} (run_id={latest.run_id})")

candidates = [
    f"runs:/{latest.run_id}/model",
    f"models:/{MODEL_NAME}/{latest.version}",
]

local_path = None
with tempfile.TemporaryDirectory() as tmp:
    for uri in candidates:
        try:
            print(f"Téléchargement depuis {uri} ...")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)
            break
        except Exception as exc:  # on tente l'URI suivante
            print(f"Échec avec {uri} : {exc}")

    if local_path is None:
        sys.exit("ERREUR : impossible de télécharger le modèle.")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(local_path, OUTPUT_DIR)

print(f"Modèle téléchargé dans ./{OUTPUT_DIR}")
print(os.listdir(OUTPUT_DIR))