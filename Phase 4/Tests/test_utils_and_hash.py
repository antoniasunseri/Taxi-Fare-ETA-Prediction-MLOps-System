import importlib
import json

db_mod = importlib.import_module("app.db")
utils_mod = importlib.import_module("app.utils")

def test_hash_request_consistent():
    req = {"a": 1, "list": [1,2,3], "nested": {"b": "c"}}
    h1 = db_mod.hash_request(req)
    h2 = db_mod.hash_request(req)
    assert isinstance(h1, str)
    assert h1 == h2

def test_featurize_output_shape_and_values():
    # featurize should return an array-like of numbers
    features = utils_mod.featurize(
        pickup_datetime="2023-01-05T08:23:00",
        pickup_lat=40.7589,
        pickup_lon=-73.9851,
        dropoff_lat=40.7614,
        dropoff_lon=-73.9776,
        passenger_count=1
    )
    assert hasattr(features, "__len__")
    assert len(features) >= 1
    # numeric elements
    for v in features:
        assert isinstance(float(v), float)
