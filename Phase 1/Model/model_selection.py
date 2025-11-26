import wandb
print("wandb imported successfully.")

wandb.login()

print("Weights & Biases authentication complete. Ready to proceed with W&B logging.")

WANDB_ENTITY = "jacobrbrooks-university-of-denver"
WANDB_PROJECT = "nyc-taxi-fare-prediction"

print(f"WANDB_ENTITY set to: {WANDB_ENTITY}")
print(f"WANDB_PROJECT set to: {WANDB_PROJECT}")

api = wandb.Api()
print("Weights & Biases API client initialized.")

runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
print(f"Fetched {len(runs)} runs from project '{WANDB_PROJECT}'.")

try:
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    print(f"Fetched {len(runs)} runs from project '{WANDB_PROJECT}'.")
except ValueError as e:
    print(f"Error: {e}")
    print(f"Could not find W&B project '{WANDB_PROJECT}' under entity '{WANDB_ENTITY}'.")
    print("Please ensure that 'WANDB_PROJECT' variable contains the correct name of your Weights & Biases project.")
    print("You can find your project name on the W&B website for your entity.")
    runs = [] # Initialize 'runs' as an empty list to prevent further errors if the project is not found.

runs_data = []
for run in runs:
    metrics = dict(run.summary)
    artifact_names = [artifact.name for artifact in run.logged_artifacts()]
    runs_data.append({
        "run_id": run.id,
        "run_name": run.name,
        "metrics": metrics,
        "artifact_names": artifact_names
    })

print(f"Extracted data for {len(runs_data)} runs.")
if runs_data:
    print("Sample of extracted data for the first run:")
    print(runs_data[0])
else:
    print("No runs data extracted.")

runs_data = []
for run in runs:
    metrics = dict(run.summary)
    artifact_names = [artifact.name for artifact in run.logged_artifacts()]
    runs_data.append({
        "run_id": run.id,
        "run_name": run.name,
        "metrics": metrics,
        "artifact_names": artifact_names
    })

print(f"Extracted data for {len(runs_data)} runs.")
if runs_data:
    print("Sample of extracted data for the first run:")
    print(runs_data[0])
else:
    print("No runs data extracted.")

metric_to_optimize = 'root_mean_squared_error'
optimization_direction = 'minimize'

# Initialize variables for tracking the best run
best_run_id = None
best_run_name = None
best_metric_value = float('inf') if optimization_direction == 'minimize' else float('-inf')
best_artifact_name = None

print(f"Optimizing for '{metric_to_optimize}' by trying to {optimization_direction} it.\n")

for run_data in runs_data:
    run_id = run_data['run_id']
    run_name = run_data['run_name']
    metrics = run_data['metrics']
    artifact_names = run_data['artifact_names']

    if metric_to_optimize in metrics:
        current_metric_value = metrics[metric_to_optimize]
        print(f"Run '{run_name}' ({run_id}) has {metric_to_optimize}: {current_metric_value:.4f}")

        if optimization_direction == 'minimize':
            if current_metric_value < best_metric_value:
                best_metric_value = current_metric_value
                best_run_id = run_id
                best_run_name = run_name
                # Assuming the primary artifact is the one to track, or pick the first if multiple
                best_artifact_name = artifact_names[0] if artifact_names else None
                #print(f"  -> New best found: {best_metric_value:.4f}")
        elif optimization_direction == 'maximize':
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_run_id = run_id
                best_run_name = run_name
                best_artifact_name = artifact_names[0] if artifact_names else None
                #print(f"  -> New best found: {best_metric_value:.4f}")
    else:
        print(f"Run '{run_name}' ({run_id}) does not have metric '{metric_to_optimize}'. Skipping.")


print("\n--- Optimization Complete ---")
if best_run_id:
    print(f"Best Run ID: {best_run_id}")
    print(f"Best Run Name: {best_run_name}")
    print(f"Best {metric_to_optimize}: {best_metric_value:.4f}")
    print(f"Associated Best Artifact Name: {best_artifact_name}")
else:
    print("No runs found with the specified metric or no best run could be determined.")

"""# Task
Find and download the best performing model artifact from Weights & Biases, storing it in a new local directory named 'best_current_model'. The best model is defined as the one that minimizes 'root_mean_squared_error'. After downloading, confirm the model's presence in the 'best_current_model' directory and inform the user of its location.

## Identify Best Model Artifact

### Subtask:
Search the runs data to find a run that achieved the same best metric value as previously identified, and logged an artifact containing 'model' in its name. This ensures we download an actual model.

**Reasoning**:
The subtask requires identifying a run that achieved the best metric value and has an artifact name containing 'model'. This code iterates through the runs, checks for the metric, and then searches for a relevant artifact name, storing the details of the first such run found.
"""

best_model_run_id = None
best_model_run_name = None
best_model_metric_value = None
best_model_artifact_name = None

for run_data in runs_data:
    run_id = run_data['run_id']
    run_name = run_data['run_name']
    metrics = run_data['metrics']
    artifact_names = run_data['artifact_names']

    # Check if the metric exists and matches the best metric value
    if metric_to_optimize in metrics and abs(metrics[metric_to_optimize] - best_metric_value) < 1e-6: # Using a small tolerance for float comparison
        for artifact_name in artifact_names:
            if 'model' in artifact_name.lower(): # Case-insensitive search for 'model'
                best_model_run_id = run_id
                best_model_run_name = run_name
                best_model_metric_value = metrics[metric_to_optimize]
                best_model_artifact_name = artifact_name
                break # Found the model artifact for this run, move to the next run
        if best_model_run_id: # If a model artifact was found for this run, we can stop searching
            break

print("\n--- Identified Best Model Artifact ---")
if best_model_run_id:
    print(f"Best Model Run ID: {best_model_run_id}")
    print(f"Best Model Run Name: {best_model_run_name}")
    print(f"Best {metric_to_optimize} Value: {best_model_metric_value:.4f}")
    print(f"Best Model Artifact Name: {best_model_artifact_name}")
else:
    print("No best model artifact found matching the criteria.")

import os

DOWNLOAD_DIR = "best_current_model"

# Create the download directory if it doesn't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(f"Download directory '{DOWNLOAD_DIR}' ensured.")

if best_model_artifact_name and best_model_run_id:
    try:
        # Construct the artifact path using WANDB_ENTITY and WANDB_PROJECT
        # The artifact name already contains the version (e.g., 'linear-regression-model:v2')
        artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{best_model_artifact_name}"

        print(f"Attempting to download artifact: {artifact_path}")

        # Get the artifact object
        artifact = api.artifact(artifact_path)

        # Download the artifact
        download_path = artifact.download(root=DOWNLOAD_DIR)

        print(f"Successfully downloaded best model artifact to: {download_path}")
        print(f"Contents of '{download_path}': {os.listdir(download_path)}")

    except Exception as e:
        print(f"Error downloading artifact '{best_model_artifact_name}' for run '{best_model_run_id}': {e}")
        print("Please ensure the artifact path is correct and accessible.")
else:
    print("No best model artifact found to download. Please review previous steps.")