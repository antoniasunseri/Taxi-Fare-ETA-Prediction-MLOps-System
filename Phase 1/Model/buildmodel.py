import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Load Data (Adjusted for direct upload) ---
# NOTE: Assuming the files 'train.csv' and 'test.csv' are uploaded to the root.
try:
    df_train = pd.read_csv('nyc-taxi-trip-duration/train.csv')
    
except FileNotFoundError:
    print("Error: Files not found. Please upload 'train.csv' and 'test.csv'.")
    # Exiting the setup if files are missing
    raise

# --- 2. Data Preparation & Feature Engineering ---

# Convert datetime columns
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
# df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime']) # Not needed for simple features

# Create Log of the target variable (standard practice for skewed data)
df_train['log_trip_duration'] = np.log1p(df_train['trip_duration'])

# Extract Simple Time Features
df_train['pickup_hour'] = df_train['pickup_datetime'].dt.hour
df_train['pickup_dayofweek'] = df_train['pickup_datetime'].dt.dayofweek # Monday=0, Sunday=6

# Convert store_and_fwd_flag to a numerical feature
df_train['flag_is_Y'] = (df_train['store_and_fwd_flag'] == 'Y').astype(int)

# --- 3. Simple Model Setup ---

# Select features: latitude, longitude, passenger count, and new time features
features = [
    'passenger_count', 
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude', 
    'pickup_hour', 
    'pickup_dayofweek', 
]
target = 'log_trip_duration'

# Handle potential NaNs (though rare in this dataset, it's good practice)
df_train = df_train.dropna(subset=features + [target])

X = df_train[features]
y = df_train[target]



# Train-test split (for internal validation, not the official test set)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
median_log_duration = y_val.median()

print(f"Median duration: {median_log_duration:.3f}")
print(f"Validation Root Mean Squared Error (RMSE) on Log-Transformed Target: {val_rmse:.4f}")

# --- 5. Basic Relationship Visualization: Trip Duration vs. Hour of Day ---
# Group the training data by hour and calculate the mean log duration
hourly_mean = df_train.groupby('pickup_hour')[target].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(hourly_mean['pickup_hour'], hourly_mean[target], marker='o', linestyle='-', color='skyblue')
plt.title('Relationship: Mean Log Trip Duration by Pickup Hour')
plt.xlabel('Pickup Hour (0=Midnight, 23=11PM)')
plt.ylabel('Mean Log Trip Duration ($\log(1+\text{seconds})$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 24, 2)) # Show every other hour
plt.savefig('mean_log_duration_by_hour.png')
plt.close()

import joblib
from sklearn.linear_model import LinearRegression 
# Assuming 'model' is your trained LinearRegression instance

# Define the filename for your saved model
filename = 'taxi_model.joblib'

# 1. Save the model to the specified file
joblib.dump(model, filename)

# Log the evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
df_test = pd.read_csv('nyc-taxi-trip-duration/test.csv')
X_test = df_train[features]
y_test = df_train[target]

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model successfully saved to {filename}")

import wandb

# Install wandb if not already installed (uncomment the line below if needed)
# !pip install wandb

# Login to Weights & Biases (you will be prompted to enter your API key if not already logged in)
wandb.login()

# Initialize a new W&B run
# You can customize the project name and configuration parameters
run = wandb.init(project="nyc-taxi-fare-prediction",
                 config={
                     "model_type": "Linear Regression",
                     "test_size": 0.2,
                     "random_state": 42,
                     "features": X.columns.tolist() # Log the features used
                 })

# Log the evaluation metrics
wandb.log({
    "mean_absolute_error": mae,
    "mean_squared_error": mse,
    "root_mean_squared_error": rmse
})

print("Weights & Biases run initialized and metrics logged successfully.")

# Initialize a new W&B run for tracking the model artifact
# Use a different project name to distinguish from the metrics logging run
run_artifact = wandb.init(project="nyc-taxi-model-predictions", job_type="model-logging")

# Create a wandb.Artifact instance
artifact = wandb.Artifact('linear-regression-model', type='model')

# Add the locally saved model file to the artifact
# The model_filename variable was defined in the previous step where the model was saved.
artifact.add_file(filename)

# Log the created artifact to Weights & Biases
wandb.log_artifact(artifact)

print(f"Weights & Biases artifact '{artifact.name}' (type: {artifact.type}) logged successfully.")

# Finish the W&B run for artifact logging
run_artifact.finish()

wandb.finish()