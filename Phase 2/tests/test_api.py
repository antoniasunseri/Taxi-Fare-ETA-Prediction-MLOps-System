from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert "status" in res.json()

def test_predict():
    payload = {
        "pickup_lat": 40.7,
        "pickup_lon": -73.9,
        "dropoff_lat": 40.8,
        "dropoff_lon": -73.95,
        "passenger_count": 1,
        "trip_distance": 2.5,
        "user_id": "testuser"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
