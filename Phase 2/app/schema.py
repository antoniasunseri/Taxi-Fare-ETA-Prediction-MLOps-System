# schema.py
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    passenger_count: int
    trip_distance: float
    user_id: str = "anonymous"
