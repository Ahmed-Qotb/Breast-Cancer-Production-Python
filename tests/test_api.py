import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from production_app import api
from production_app.config import FEATURES, config


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_payload() -> dict[str, float]:
    row = pd.read_csv(config.data_path).iloc[0]

    return {feature: float(row[feature]) for feature in FEATURES}


def test_health_returns_healthy(client):
    assert api.model is not None

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_probabilities(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200, response.text

    probabilities = response.json()

    assert isinstance(probabilities, list)
    assert len(probabilities) == 2
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert np.isclose(sum(probabilities), 1.0)


def test_predict_rejects_a_missing_feature(client, valid_payload):
    payload = dict(valid_payload)
    payload.pop(FEATURES[0])

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_a_malformed_feature(client, valid_payload):
    payload = dict(valid_payload)
    payload[FEATURES[0]] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_health_reports_an_unavailable_model(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "message": "Model is unavailable",
    }


def test_predict_reports_an_unavailable_model(client, valid_payload, monkeypatch):
    monkeypatch.setattr(api, "model", None)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Train the model first"
