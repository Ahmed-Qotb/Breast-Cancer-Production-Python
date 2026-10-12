from pathlib import Path

import joblib

from production_app.config import MODEL_PATH
from production_app.logging_config import logger


def save_model(model, model_path=MODEL_PATH) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    logger.info("Model saved to %s", path)


def load_model(model_path=MODEL_PATH):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)

    logger.info("Model loaded from %s", path)

    return model
