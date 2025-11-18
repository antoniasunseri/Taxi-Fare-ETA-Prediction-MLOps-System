from schema import PredictionRequest

def test_schema_creation():
    req = PredictionRequest(
        pickup_lat=40.7,
        pickup_lon=-73.9,
        dropoff_lat=40.8,
        dropoff_lon=-73.95,
        passenger_count=1,
        trip_distance=2.5
    )
    assert req.passenger_count == 1
    assert req.pickup_lat == 40.7
