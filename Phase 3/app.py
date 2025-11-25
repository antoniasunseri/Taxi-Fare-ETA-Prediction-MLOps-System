
#!pip install streamlit
import streamlit as st
import requests

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8000" # Replace with your FastAPI app's URL

# --- Streamlit App ---
st.set_page_config(page_title="FastAPI-Streamlit Interactor", layout="centered")
st.title("FastAPI Interaction with Streamlit")

st.header("1. Health Check (GET /health)")
if st.button("Check FastAPI Health"):
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health")
        if response.status_code == 200:
            st.success(f"FastAPI Health: {response.json().get('status', 'OK')}")
        else:
            st.error(f"Health check failed: Status code {response.status_code}, Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI at {FASTAPI_BASE_URL}. Is it running?")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.header("2. Prediction (POST /predict)")

pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.4f")
pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.4f")
dropoff_lat = st.number_input("Dropoff Latitude", value=40.7580, format="%.4f")
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9855, format="%.4f")
passenger_count = st.number_input("Passenger Count", value=1, min_value=1, max_value=8)
trip_distance = st.number_input("Trip Distance", value=2.5, format="%.2f")

if st.button("Get Prediction"):
    # Construct the data payload matching the FastAPI's expected request body
    # Assuming your FastAPI expects a JSON object with these fields.
    # If it expects a list directly, you'll need to adjust this.
    data = {
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "passenger_count": passenger_count,
        "trip_distance": trip_distance
    }

    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/predict", json=data)

        if response.status_code == 200:
            st.success("Prediction successful!")
            st.json(response.json())
        else:
            st.error(f"Prediction failed: Status code {response.status_code}, Response: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI at {FASTAPI_BASE_URL}. Is it running?")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("--- \n _To run this Streamlit app, save the code as a `.py` file (e.g., `app.py`) and then run `streamlit run app.py` in your terminal. Ensure your FastAPI app is running in the background._")