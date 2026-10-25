import pandas as pd
from fastapi import FastAPI, HTTPException

from production_app.logging_config import logger
from production_app.model_service import load_model
from production_app.prediction import predict
from production_app.schemas import PaitentData

try:
    model = load_model()
except Exception:  # noqa: BLE001
    logger.exception("Failed to load the model")
    model = None

app = FastAPI(title="Breast Cancer Prediction API")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "message": "Model is unavailable"},
        )

    return {"status": "healthy"}


@app.post("/predict")
def predict_one(patient: PaitentData):
    if model is None:
        raise HTTPException(status_code=503, detail="Train the model first")

    features = pd.DataFrame([patient.model_dump(by_alias=True)])
    probabilities = predict(features)

    return probabilities[0].tolist()
