import pandas as pd
from fastapi import FastAPI, HTTPException

from production_app.model_service import load_model
from production_app.prediction import predict
from production_app.schemas import PaitentData

model = load_model()

app = FastAPI(title="Breast Cancer Prediction API")


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "not_ready"}


@app.post("/predict")
def predict_one(patient: PaitentData):
    if model is None:
        raise HTTPException(status_code=503, detail="Train the model first")

    features = pd.DataFrame([patient.model_dump(by_alias=True)])
    probabilities = predict(features)

    return probabilities[0].tolist()
