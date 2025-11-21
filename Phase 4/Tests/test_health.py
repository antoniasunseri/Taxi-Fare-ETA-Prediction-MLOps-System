from fastapi.testclient import TestClient
import importlib

# import app from backend package
app_mod = importlib.import_module("app.main")
app = getattr(app_mod, "app")
client = TestClient(app)

def test_health_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert body["status"] == "ok"
