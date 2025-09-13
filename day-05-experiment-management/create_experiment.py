# Import necessary libraries
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from pathlib import Path # <- New line

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
    print("Current Tracking URI: ", mlflow.get_tracking_uri())  # <- New line

    # Create a new experiment
    experiment_id = mlflow.create_experiment(
        name="ElasticNet-Wine-Quality-2",
        tags={"version": "v1", "priority": "p1"},
        artifact_location=Path.cwd().joinpath("my_artifacts_location").as_uri()
    )

    # Get the experiment object
    experiment_obj = mlflow.get_experiment(experiment_id) # <- New line

    # Print experiment details
    print(f"Experiment Name: {experiment_obj.name}") # <- New line
    print(f"Experiment ID: {experiment_id}") # <- New line
    print(f"Artifact Location: {experiment_obj.artifact_location}") # <- New line
    print(f"Tags: {experiment_obj.tags}") # <- New line
    print(f"Lifecycle Stage: {experiment_obj.lifecycle_stage}") # <- New line
    print(f"Creation Timestamp: {experiment_obj.creation_time}") # <- New line 

    with mlflow.start_run(experiment_id=experiment_id): 
        # Create and train the ElasticNet model with specified parameters
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        # Use the trained model to make predictions on test data
        y_pred = lr.predict(X_test)

        # Calculate evaluation metrics by comparing predictions with actual values
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        # Print the results
        print_metrics(alpha, l1_ratio, rmse, mae, r2)

        # Log the model parameters to MLflow
        mlflow.log_param("alpha", alpha) 
        mlflow.log_param("l1_ratio", l1_ratio) 

        # Log the evaluation metrics to MLflow
        mlflow.log_metric("rmse", rmse) 
        mlflow.log_metric("mae", mae) 
        mlflow.log_metric("r2", r2) 

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(lr, name="elasticnet-model", input_example=X_test.iloc[0, 1])
