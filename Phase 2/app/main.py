# main.py
import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import wandb
import boto3
from dotenv import load_dotenv

AWS_ACCESS_KEY_ID = ASIAZDNUGXE6RXEL5PGN
AWS_SECRET_ACCESS_KEY = PBPdoAlG1x7SNcjWV8ri/dPZ1A9tK48EKOa6lbAV
AWS_SESSION_TOKEN = IQoJb3JpZ2luX2VjELD//////////wEaCXVzLXdlc3QtMiJHMEUCIQC6FGAqRu0HXGrb3hVM3aJAaigwr04OwbxAfTXUrneKwQIgIbJN/TOY8LAEXub0spvmsUizjfqY0v5wFVHS6ancsloqoQIIeRAAGgw2MjU4MzI1NDA0NzciDFcFOkqCi9BUEoseySr+AYHH1uCH2bwoPgNCvsVRkOstmdCY5SUQbJ/Dm9h+bxAQbyohA2h8VNpmsXW8ii+/Tmxm/Irp3uBqRq9OhXIROwBz2zHCtL5scN3YIzBlKcBQTCwgQbcrvbEUQwkEZoF2UKxfIwcmVyRKjWQq1ktT46ss4Y9kiQQkFm+/fNmIe7lEGEO83VyvVO16OlLt7p2IqDPHjr0MXledOKddhSB3vsDl8xELv5HxhYhANOE8vbKrf3sCujvc4qH9GhQguhWYMBZjRrnF79HNUI6toygTS27vS537KlkaDrbRAZi4ay2o6WRotNgFy+hrbB5e1DpG1qw0WHdjCuYphS7pRhZlMKKMmckGOp0BuF8wg+lpyrsGjdfP9M3lZ+2B1ichg2hzaQE0f4Kp0NjYX/TcqEteOxjyxNz4V0HNHyk7UQvXUZ9Fo6xQIG6a2kYht7JuQsQxUP/Pg4hMW0YLzKqrE0vVSQbqsZY73H0cnGCbuY07LzQelywIU7M5KhppesxKtAIxCAk+rDWkEykOt6r4eEXWSEXHULBBls/eLpAviqONAg2GO0m70A==


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "taxi-fare-eta")
WAND_MODEL_NAME = os.getenv("WAND_MODEL_NAME", "taxi_model")
WAND_MODEL_ALIAS = os.getenv("WAND_MODEL_ALIAS", "production")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE", "taxi-predictions")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Taxi Fare / ETA Prediction API")

# -----------------------------
# Pydantic schema
# -----------------------------
class PredictionRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    passenger_count: int
    trip_distance: float
    user_id: str = "anonymous"

# -----------------------------
# Global variables
# -----------------------------
MODEL = None
dynamo = None

# -----------------------------
# Load model from Weights & Biases
# -----------------------------
def load_model_from_wandb():
    global MODEL
    wandb.login(key=WANDB_API_KEY)
    api = wandb.Api()
    artifact_ref = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WAND_MODEL_NAME}:{WAND_MODEL_ALIAS}"
    artifact = api.artifact(artifact_ref)
    model_dir = artifact.download()
    MODEL = joblib.load(f"{model_dir}/model.joblib")
    print(f"Model loaded from {model_dir}/model.joblib")

# -----------------------------
# Log prediction to DynamoDB
# -----------------------------
def log_prediction(req: PredictionRequest, prediction: float):
    global dynamo
    if dynamo is None:
        dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamo.Table(DYNAMO_TABLE)
    
    item = {
        "request_id": f"{req.user_id}-{datetime.utcnow().timestamp()}",
        "timestamp": datetime.utcnow().isoformat(),
        "input": json.loads(req.json()),
        "prediction": prediction,
        "model_alias": WAND_MODEL_ALIAS
    }
    
    table.put_item(Item=item)

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
def startup_event():
    load_model_from_wandb()
    print("Model loaded and FastAPI ready")

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Prepare features in same order as model training
    features = [
        req.pickup_lat,
        req.pickup_lon,
        req.dropoff_lat,
        req.dropoff_lon,
        req.passenger_count,
        req.trip_distance
    ]
    
    # Make prediction
    prediction = MODEL.predict([features])[0]
    
    # Log prediction to DynamoDB
    log_prediction(req, float(prediction))
    
    return {
        "prediction": float(prediction),
        "timestamp": datetime.utcnow().isoformat()
    }
