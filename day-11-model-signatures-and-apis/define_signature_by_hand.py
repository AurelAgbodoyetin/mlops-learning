# Import necessary libraries
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# Import MLflow and its sklearn module
import mlflow    
import mlflow.sklearn      
from mlflow.types.schema import Schema,ColSpec # <- New Line
from mlflow.models.signature import ModelSignature # <- New Line

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
    experiment_obj = mlflow.set_experiment(experiment_name="ElasticNet-Wine-Quality-Signature-Manual")

    # Note that now we have an Experiment object, not just an ID
    # A new experiment is created only if it doesn't already exist
    experiment_id = experiment_obj.experiment_id

    # Print experiment details
    print_experiment_details(experiment_obj) 

    mlflow.start_run()  
    mlflow.set_tag("release.version", "0.1")
    mlflow.set_tag("developer", "MasterBee")

    mlflow.sklearn.autolog( # <- New Line Enable autologging for MLflow
        log_input_examples=False,  # <- New Line Disable input example logging
        log_model_signatures=False,  # <- New Line Disable model signature logging
        log_models=False         # <- New Line Disable model logging
    )

    # Create and train the ElasticNet model with specified parameters
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    # Use the trained model to make predictions on test data
    y_pred = lr.predict(X_test)

    # Calculate evaluation metrics by comparing predictions with actual values
    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    # Print the results
    print_metrics(alpha, l1_ratio, rmse, mae, r2)

    input_data = [ # <- New Lines Define input schema manually
        {"name": "fixed acidity", "type": "double"},
        {"name": "volatile acidity", "type": "double"},
        {"name": "citric acid", "type": "double"},
        {"name": "residual sugar", "type": "double"},
        {"name": "chlorides", "type": "double"},
        {"name": "free sulfur dioxide", "type": "double"},
        {"name": "total sulfur dioxide", "type": "double"},
        {"name": "density", "type": "double"},
        {"name": "pH", "type": "double"},
        {"name": "sulphates", "type": "double"},
        {"name": "alcohol", "type": "double"},
    ]
    output_data = [ # <- New Lines Define output schema manually
        {"name": "quality", "type": "long"}
    ]

    # Create input and output schemas and the model signature
    input_schema = Schema([ColSpec(type=col["type"], name=col["name"]) for col in input_data])
    output_schema = Schema([ColSpec(type=col["type"], name=col["name"]) for col in output_data])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Define a sample input example for the model
    input_example = {
        "fixed acidity": np.array([7.2, 7.5, 7.0, 6.8, 6.91]),
        "volatile acidity": np.array([0.35, 0.3, 0.28, 0.38, 0.251]),
        "citric acid": np.array([0.45, 0.5, 0.55, 0.4, 0.42]),
        "residual sugar": np.array([8.5, 9.0, 8.2, 7.8, 8.11]),
        "chlorides": np.array([0.045, 0.04, 0.035, 0.05, 0.042]),
        "free sulfur dioxide": np.array([30, 35, 40, 28, 32]),
        "total sulfur dioxide": np.array([120, 125, 130, 115, 110]),
        "density": np.array([0.997, 0.996, 0.995, 0.998, 0.994]),
        "pH": np.array([3.2, 3.1, 3.0, 3.3, 3.2]),
        "sulphates": np.array([0.65, 0.7, 0.68, 0.72, 0.62]),
        "alcohol": np.array([9.2, 9.5, 9.0, 9.8, 9.4]),
    }

    # Log the trained model to MLflow and register the signature and input example
    mlflow.sklearn.log_model(lr, name="elasticnet-model", signature=signature, input_example=input_example)

    # Get and print the URI of the logged artifacts
    artifact_uri = mlflow.get_artifact_uri()  
    print("Model artifact URI: ", artifact_uri)

    mlflow.end_run()