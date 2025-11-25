# Taxi Fare / ETA Prediction – End-to-End MLOps System  
# Partners: Antonia Sunseri and Jacob Brooks 
Model: **Linear Regression (`model.joblib`)**  
Backend: **FastAPI**  
Experiment Tracking: **Weights & Biases**  
Database: **DynamoDB**  
CI/CD: **GitHub Actions (pytest)**  

---

## Project Overview
This project implements a full MLOps pipeline for NYC taxi fare/ETA prediction using AWS +  ML tooling.  
The final system consists of:

1. **Experiment Tracking & Model Registry (W&B)**
2. **Model Backend (FastAPI)**
3. **Persistent Data Store (AWS DynamoDB)**
4. **Frontend Interface (Streamlit)**
5. **CI/CD Pipeline (GitHub Actions)**

# Phase 1 — Experimentation & Model Management

## 1.1 Model Development  
Dataset: **NYC TLC Trip Records (AWS Open Data)**  
Model: **Linear Regression** trained on features:
- pickup_lat  
- pickup_lon  
- dropoff_lat  
- dropoff_lon  
- passenger_count  
- trip_distance  

Model exported as:  
```
model.joblib
```

---

## 1.2 Experiment Tracking (Weights & Biases)
Training run logs:
- parameters  
- metrics (MAE, RMSE, R², etc.)  
- dataset version  
- git commit hash  
- generated model artifact  

---

## 1.3 Model Versioning & Registry
The final model is uploaded to W&B as an artifact:

```
taxi_model:production
```

The FastAPI backend downloads this exact file on startup.

---


# Phase 2 — FastAPI Backend & DynamoDB Integration 

## Backend Summary

### Load model from W&B
```python
artifact = api.artifact(f"{ENTITY}/{PROJECT}/{MODEL_NAME}:{ALIAS}")
MODEL = joblib.load(model_dir + "/model.joblib")
```

### Exposes `/predict` endpoint
- Reads JSON input
- Converts into model feature order
- Returns prediction

### Logs every prediction to DynamoDB  
Item contains:
- `request_id`
- `timestamp`
- request payload
- `prediction`

---

## DynamoDB Table  

**Table name:** `taxi-predictions`  
**Partition key:** `request_id` (string)  

---


# Phase 3 — Frontend & Monitoring

### Frontend (Streamlit)
Frontend should:
- Accept user inputs  
- Call the FastAPI `/predict` endpoint  
- Display result  


# Phase 4 — Testing & CI/CD

## 4.1 Tests
Use mocks for:
- W&B
- DynamoDB
- Model

Test examples:
- API loads model  
- /health returns ok  
- /predict returns a float  

Run tests:
```bash
pytest -q
```

---

## 4.2 CI/CD Pipeline

Add this file:
### `.github/workflows/ci.yml`


# Running the Backend 


### Set up env variables
Create `.env` inside `Phase 2/app/`:

```
WANDB_API_KEY=xxxxx
WANDB_ENTITY=your-wandb-user
WANDB_PROJECT=taxi-fare-eta
WAND_MODEL_NAME=taxi_model
WAND_MODEL_ALIAS=production
DYNAMO_TABLE=taxi-predictions
AWS_REGION=us-east-1
```

---

### Install dependencies
```bash
cd "Phase 2/app"
pip install -r requirements.txt
```

---

### Run FastAPI
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit docs at:  
http://localhost:8000/docs

---


# Docker Deployment (EC2)

### Build container
```bash
docker build -t taxi-api .
```

### Run container with env vars
```bash
docker run -p 8000:8000 --env-file .env taxi-api
```

To deploy on EC2:
- Install Docker  
- Pull or build image  
- Run with `.env`  
- Ensure IAM role grants DynamoDB access  

---






