from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from production_app.config import FEATURES, config
from production_app.logging_config import logger


# Load dataset function
def get_dataset(path: Path) -> pd.DataFrame:
    dataset_path = Path(path)

    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    logger.info("Loading dataset")

    df = pd.read_csv(dataset_path)

    # Keep only complete rows for the selected features and target
    df = df.drop_duplicates().dropna(subset=FEATURES + [config.target])

    # Drop unnecessary columns
    df.drop(columns=["Unnamed: 32", "id"], errors="ignore", inplace=True)

    logger.info("Dataset loaded successfully: %s", df.shape)

    return df


# Split the selected features into train and test sets
def split_dataset(df):
    X = df[FEATURES]
    y = df[config.target].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    logger.info("Dataset split completed")

    return X_train, X_test, y_train, y_test
