import pandas as pd

from production_app.config import FEATURES, config
from production_app.logging_config import logger
from production_app.model_service import load_model


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the same non-feature columns removed during training
    df = df.drop(columns=["Unnamed: 32", "id", "diagnosis"], errors="ignore")

    # Keep only the features selected during training
    df = df[FEATURES]

    return df


def predict(df: pd.DataFrame):
    # Load the saved model
    model = load_model(config.model_path)
    if model is None:
        raise FileNotFoundError("Train the model first")

    # Apply the same preprocessing steps
    df = preprocess_data(df)

    # Make predictions
    probabilities = model.predict_proba(df)

    logger.info("Prediction completed")

    return probabilities
