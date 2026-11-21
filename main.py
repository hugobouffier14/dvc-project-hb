from typing import List

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Un seul endpoint exposé : on désactive la doc auto (/docs, /redoc, /openapi.json)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

model = mlflow.pyfunc.load_model("model")


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=64, max_length=64)


@app.post("/predict")
def predict(request: PredictRequest):
    x = np.array([request.features], dtype=np.float64)
    prediction = model.predict(x)
    return {"prediction": int(np.asarray(prediction).ravel()[0])}