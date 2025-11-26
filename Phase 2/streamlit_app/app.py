import streamlit as st
import pandas as pd
import json

# Path to shared logs directory
log_file = "/home/ubuntu/logs/prediction_logs.json"

st.title("Taxi Fare / ETA Prediction Logs")

try:
    with open(log_file, "r") as f:
        logs = json.load(f)
    df = pd.json_normalize(logs)
    
    st.subheader("All Predictions")
    st.dataframe(df)

    st.subheader("Prediction Statistics")
    st.write(df['prediction'].describe())

    # filter by user_id
    user_ids = df['user_id'].unique()
    selected_user = st.selectbox("Filter by User ID", options=user_ids)
    st.dataframe(df[df['user_id'] == selected_user])

except FileNotFoundError:
    st.warning("No prediction logs found yet.")
