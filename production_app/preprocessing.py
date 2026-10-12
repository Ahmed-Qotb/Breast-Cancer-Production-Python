from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from production_app.config import RANDOM_SEED
from production_app.logging_config import logger


def remove_outliers(X_train, X_test):
    # will use iqr clipping like the notebook

    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X_train = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test = X_test.clip(lower=lower, upper=upper, axis=1)

    logger.info("Outlier clipping completed")

    return X_train, X_test


# grouping preprocessing steps into 1 reuseable flow
def create_preprocessing_pipeline():
    logger.info("Creating preprocessing pipeline")

    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
        ]
    )
