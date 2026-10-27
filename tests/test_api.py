import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_predict_endpoint_valid():
    payload = {
        "features": {
            "Gr Liv Area": 1500,
            "Overall Qual": 7,
            "Garage Area": 400,
            "Neighborhood": "NAmes",
            "Full Bath": 2,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_usd" in data
    assert isinstance(data["predicted_price_usd"], (int, float))
    assert data["predicted_price_usd"] > 0


def test_predict_invalid_payload():
    response = client.post("/predict", json={"features": "bad_input"})
    assert response.status_code == 422
