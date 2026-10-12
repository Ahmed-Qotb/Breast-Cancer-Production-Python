import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project root
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Application settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Dataset and model paths
DATA_PATH = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "data.csv")))
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Machine learning settings
TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_ITERATIONS = 1000
