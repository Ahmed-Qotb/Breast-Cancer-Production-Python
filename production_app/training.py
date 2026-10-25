from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from production_app.config import config
from production_app.logging_config import logger
from production_app.preprocessing import create_preprocessing_pipeline

# from sklearn.feature_selection import RFECV


def train_model(X_train, X_test, y_train, y_test):
    # Feature selection was done once with RFECV; the FEATURES list in
    # config.py holds the selected features, so training uses only those.

    # feature_selector = RFECV(
    #     estimator=LogisticRegression(max_iter=config.max_iterations),
    #     step=1,
    #     cv=config.cv_folds,
    #     scoring="accuracy",
    # )
    preprocessing = create_preprocessing_pipeline()

    # Combine preprocessing and model
    model = Pipeline(
        [
            ("outlier_clipper", preprocessing.named_steps["outlier_clipper"]),
            ("scaler", preprocessing.named_steps["scaler"]),
            # ("feature_selector", feature_selector),
            ("smote", preprocessing.named_steps["smote"]),
            ("classifier", LogisticRegression(max_iter=config.max_iterations)),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model, X_test, y_test
