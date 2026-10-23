from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_prediction_endpoint():
    payload = {
        "features": {
            "Gr Liv Area": 1500,
            "Overall Qual": 7,
            "Garage Area": 400,
            "Neighborhood": "NAmes",
            "Full Bath": 2
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_usd" in data
    assert isinstance(data["predicted_price_usd"], (int, float))
    assert data["predicted_price_usd"] > 0