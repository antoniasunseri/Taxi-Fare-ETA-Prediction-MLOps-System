# tests/test_db_dynamodb_mock.py
import importlib
import types

# import the db module to test
db_mod = importlib.import_module("app.db")

class DummyTable:
    def __init__(self):
        self.store = {}

    def put_item(self, Item):
        # simple store using a key field if present; otherwise generate
        key = Item.get("request_hash") or Item.get("id") or str(len(self.store))
        self.store[key] = Item
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_item(self, Key):
        k = Key.get("request_hash")
        it = self.store.get(k)
        if it is None:
            return {}
        return {"Item": it}

class DummyDynamo:
    def __init__(self):
        self._tables = {}
    def Table(self, name):
        if name not in self._tables:
            self._tables[name] = DummyTable()
        return self._tables[name]

def test_dynamodb_cache_lookup_and_write(monkeypatch):
    dummy = DummyDynamo()
    monkeypatch.setattr(db_mod, "dynamodb", dummy)
    monkeypatch.setattr(db_mod, "logs_table", dummy.Table("prediction_logs"))
    monkeypatch.setattr(db_mod, "cache_table", dummy.Table("fare_cache"))

    # sample request object
    sample_req = {
        "pickup_longitude": -73.99,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.98,
        "dropoff_latitude": 40.74,
        "passenger_count": 1,
        "trip_distance": 1.2
    }

    miss = db_mod.dynamodb_cache_lookup(types.SimpleNamespace(**sample_req))
    assert miss is None

    db_mod.dynamodb_cache_write(types.SimpleNamespace(**sample_req), 9.99)
    found = db_mod.dynamodb_cache_lookup(types.SimpleNamespace(**sample_req))
    assert found == 9.99
