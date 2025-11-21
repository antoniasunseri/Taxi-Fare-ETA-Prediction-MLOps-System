from fastapi.testclient import TestClient
import importlib
import json

# import modules from backend package
app_mod = importlib.import_module("app.main")
model_loader = importlib.import_module("app.model_loader")
db_mod = importlib.import_module("app.db")

app = getattr(app_mod, "app")
client = TestClient(app)

def test_predict_monkeypatched(monkeypatch):
    # 1) Provide a fake model object
    class FakeModel:
        def predict(self, X):
            return [sum(map(float, X[0])) * 0.1]

    # monkeypatch the model in model_loader
    monkeypatch.setattr(model_loader, "model", FakeModel())

    recorded = {}
    def fake_log_prediction(inputs, prediction):
        recorded["inputs"] = inputs
        recorded["prediction"] = prediction
        return True

    monkeypatch.setattr(db_mod, "log_prediction", fake_log_prediction)

    # mock cache lookup/write to simulate cache miss then write
    def fake_cache_lookup(req):
        return None
    def fake_cache_write(req, pred):
        recorded["cache_written"] = True

    monkeypatch.setattr(db_mod, "dynamodb_cache_lookup", fake_cache_lookup)
    monkeypatch.setattr(db_mod, "dynamodb_cache_write", fake_cache_write)

    payload = {
        "pickup_longitude": -73.99,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.98,
        "dropoff_latitude": 40.74,
        "passenger_count": 1,
        "trip_distance": 1.2
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "fare" in body
    assert isinstance(body["fare"], float) or isinstance(body["fare"], int)
    assert body.get("cached", False) in (True, False)
    # check that logging happened
    assert "prediction" in recorded
    assert recorded.get("cache_written") is True
