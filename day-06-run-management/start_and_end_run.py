# Import necessary libraries
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from pathlib import Path 

# Import MLflow and its sklearn module
import mlflow    
import mlflow.sklearn      

# Function to calculate evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Function to print model performance metrics
def print_metrics(alpha, l1_ratio, rmse, mae, r2):
    print("\n" + "="*50)
    print("ELASTICNET MODEL PERFORMANCE")
    print("="*50)
    print(f"Parameters: alpha = {alpha:.4f}, l1_ratio = {l1_ratio:.4f}")
    print("-"*50)
    print(f"{'Metric':<10} {'Value':<12} {'Interpretation':<15}")
    print("-"*50)
    print(f"{'RMSE':<10} {rmse:<12.4f} {'Lower = Better':<15}")
    print(f"{'MAE':<10} {mae:<12.4f} {'Lower = Better':<15}")
    print(f"{'R²':<10} {r2:<12.4f} {'Higher = Better':<15}")
    print("="*50)

# New function to print experiment details
def print_experiment_details(exp): # New function
    print(f"Experiment Name: {exp.name}")
    print(f"Experiment ID: {exp.experiment_id}")
    print(f"Artifact Location: {exp.artifact_location}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle Stage: {exp.lifecycle_stage}")
    print(f"Creation Timestamp: {exp.creation_time}")

# New function to log model parameters and metrics
def log_model_params_and_metrics(lr, alpha, l1_ratio, rmse, mae, r2): # New function
    # Log the model parameters to MLflow
    mlflow.log_param("alpha", alpha) 
    mlflow.log_param("l1_ratio", l1_ratio) 

    # Log the evaluation metrics to MLflow
    mlflow.log_metric("rmse", rmse) 
    mlflow.log_metric("mae", mae) 
    mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")# Ignore warnings
    np.random.seed(42)  # Set random seed for reproducibility

    # Set up logging configuration (only show warnings and above)
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    # Read the wine quality dataset from CSV file
    csv_url = "https://raw.githubusercontent.com/myuser114/wine-dataset/refs/heads/main/winequality.csv"
    data = pd.read_csv(csv_url, sep=";")

    # Split the data into features and target variable
    features = data.drop(["quality"], axis=1)
    target = data[["quality"]]

    # Split the data into training (75%) and testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

    # Set the parameters
    alpha = 0.5
    l1_ratio = 0.5

    # Set the tracking URI to default (local file-based)
    mlflow.set_tracking_uri("")

    # Print the current tracking URI
    print("Current Tracking URI: ", mlflow.get_tracking_uri())  

    # Create a new experiment
    experiment_obj = mlflow.set_experiment(experiment_name="ElasticNet-Wine-Quality-1")

    # Note that now we have an Experiment object, not just an ID
    # A new experiment is created only if it doesn't already exist
    experiment_id = experiment_obj.experiment_id

    # Print experiment details
    print_experiment_details(experiment_obj) # <- New line

    with mlflow.start_run(experiment_id=experiment_id, run_name="ElasticNet-Run-1"): # <- New line
        # Create and train the ElasticNet model with specified parameters
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        # Use the trained model to make predictions on test data
        y_pred = lr.predict(X_test)

        # Calculate evaluation metrics by comparing predictions with actual values
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        # Print the results
        print_metrics(alpha, l1_ratio, rmse, mae, r2)

        # Log the model parameters and metrics to MLflow
        log_model_params_and_metrics(lr, alpha, l1_ratio, rmse, mae, r2) # <- New line

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(lr, name="elasticnet-model", input_example=X_test.iloc[0:])
    # The run is automatically ended when exiting the 'with' block

mlflow.start_run()  # <- New line
# Create and train the ElasticNet model with specified parameters
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(X_train, y_train)

# Use the trained model to make predictions on test data
y_pred = lr.predict(X_test)

# Calculate evaluation metrics by comparing predictions with actual values
(rmse, mae, r2) = eval_metrics(y_test, y_pred)

# Print the results
print_metrics(alpha, l1_ratio, rmse, mae, r2)

# Log the model parameters and metrics to MLflow
log_model_params_and_metrics(lr, alpha, l1_ratio, rmse, mae, r2) # <- New line

# Log the trained model to MLflow
mlflow.sklearn.log_model(lr, name="elasticnet-model", input_example=X_test.iloc[0:])
mlflow.end_run()  # <- New line