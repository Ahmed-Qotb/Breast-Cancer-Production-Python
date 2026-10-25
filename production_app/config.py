import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the project root
load_dotenv(ROOT / ".env")

# RFECV was run once on the full 30-feature training set using 5-fold
# cross-validation and accuracy scoring. These 17 features were selected.
# Rerun RFECV and update this list if the dataset changes.
FEATURES = [
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave points_mean",
    "radius_se",
    "area_se",
    "compactness_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
]


@dataclass
class Config:
    data_path: Path = ROOT / os.getenv("DATA_PATH", "data/data.csv")
    model_path: Path = ROOT / os.getenv("MODEL_PATH", "models/model.joblib")
    metrics_path: Path = ROOT / os.getenv("METRICS_PATH", "artifacts/metrics.json")
    api_url: str = os.getenv("API_URL", "http://127.0.0.1:8000")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    test_size: float = 0.2
    random_state: int = 42
    max_iterations: int = 1000
    target: str = "diagnosis"
    cv_folds: int = 5


config = Config()
