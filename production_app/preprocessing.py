from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

from production_app.config import config
from production_app.logging_config import logger


class OutlierClipper(BaseEstimator, TransformerMixin):
    # will use iqr clipping like the notebook
    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        self.lower_ = Q1 - 1.5 * IQR
        self.upper_ = Q3 + 1.5 * IQR

        logger.info("Outlier clipping fitted")

        return self

    def transform(self, X):
        X = X.clip(lower=self.lower_, upper=self.upper_, axis=1)

        return X


# grouping preprocessing steps into 1 reuseable flow
def create_preprocessing_pipeline():
    logger.info("Creating preprocessing pipeline")

    return Pipeline(
        [
            ("outlier_clipper", OutlierClipper()),
            ("scaler", RobustScaler()),
            # Feature selection was run once with RFECV (5-fold CV, accuracy).
            # The selected features are fixed in config.FEATURES, so this step
            # ("feature_selector", RFECV(
            #     estimator=LogisticRegression(max_iter=config.max_iterations),
            #     step=1,
            #     cv=config.cv_folds,
            #     scoring="accuracy",
            # )),
            ("smote", SMOTE(random_state=config.random_state)),
        ]
    )
