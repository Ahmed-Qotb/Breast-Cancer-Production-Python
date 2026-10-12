from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from production_app.config import RANDOM_SEED, TEST_SIZE
from production_app.logging_config import logger

load_dotenv()


# Load dataset function
def get_dataset(path: Path) -> pd.DataFrame:
    dataset_path = Path(path)

    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    logger.info("Loading dataset")

    df = pd.read_csv(dataset_path)

    # Drop unnecessary columns
    df.drop(columns=["Unnamed: 32", "id"], errors="ignore", inplace=True)

    logger.info("Dataset loaded successfully: %s", df.shape)

    return df


# Spliting the data and separating numirical from catigorical
def split_dataset(df):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info("Dataset split completed")

    return X_train, X_test, y_train, y_test


# if __name__ == "__main__":
#     df = get_dataset(DATA_PATH)

#     X_train, X_test, y_train, y_test = split_dataset(df)

#     print("Dataset loaded:", df.shape)
#     print("Training features:", X_train.shape)
#     print("Testing features:", X_test.shape)
#     print("Training target:", y_train.shape)
#     print("Testing target:", y_test.shape)

#     assert "diagnosis" not in X_train.columns
#     assert len(X_train) == len(y_train)
#     assert len(X_test) == len(y_test)
#     assert set(y_train.unique()).issubset({0, 1})
#     assert set(y_test.unique()).issubset({0, 1})

#     print("All data tests passed!")
