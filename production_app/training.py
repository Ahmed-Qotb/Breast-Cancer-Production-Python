from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from production_app.config import MAX_ITERATIONS
from production_app.logging_config import logger
from production_app.preprocessing import (
    create_preprocessing_pipeline,
    remove_outliers,
)


def train_model(X_train, X_test, y_train, y_test):
    # Handle outliers
    X_train, X_test = remove_outliers(X_train, X_test)

    # Create preprocessing pipeline
    preprocessing = create_preprocessing_pipeline()

    # Combine preprocessing and model
    model = Pipeline(
        [
            ("scaler", preprocessing.named_steps["scaler"]),
            ("smote", preprocessing.named_steps["smote"]),
            ("classifier", LogisticRegression(max_iter=MAX_ITERATIONS)),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model, X_test, y_test
