import numpy as np
import pandas as pd

from production_app.config import FEATURES, config
from production_app.model_service import load_model
from production_app.prediction import predict


def feature_frame(rows: int = 5) -> pd.DataFrame:
    data = pd.read_csv(config.data_path)

    return data[FEATURES].head(rows)


def test_selected_feature_count():
    assert len(FEATURES) == 17


def test_saved_model_loads_and_predicts_selected_features():
    model = load_model()
    features = feature_frame()

    assert features.shape == (5, len(FEATURES))

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(features))).issubset({0, 1})


def test_prediction_function_uses_the_saved_artifact():
    probabilities = predict(feature_frame(1))

    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
