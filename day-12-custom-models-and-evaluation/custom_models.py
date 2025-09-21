# Import necessary libraries
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import cloudpickle # <- New Line
import joblib # <- New Line
import sklearn # <- New Line
import os # <- New Line

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

def print_experiment_details(exp):
    print(f"Experiment Name: {exp.name}")
    print(f"Experiment ID: {exp.experiment_id}")
    print(f"Artifact Location: {exp.artifact_location}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle Stage: {exp.lifecycle_stage}")
    print(f"Creation Timestamp: {exp.creation_time}")

def log_model_params_and_metrics(alpha, l1_ratio, rmse, mae, r2):
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

    data_dir = 'red-wine-data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data.to_csv(data_dir + '/wine-data.csv')
    X_train.to_csv(data_dir + '/X_train.csv')
    X_test.to_csv(data_dir + '/X_test.csv')

    # Set the parameters
    alpha = 0.5
    l1_ratio = 0.5

    # Set the tracking URI to default (local file-based)
    mlflow.set_tracking_uri("")

    # Print the current tracking URI
    print("Current Tracking URI: ", mlflow.get_tracking_uri())  

    # Create a new experiment
    experiment_obj = mlflow.set_experiment(experiment_name="ElasticNet-Wine-Quality-Custom-Model")

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

    # Log the model parameters and metrics to MLflow
    log_model_params_and_metrics(alpha, l1_ratio, rmse, mae, r2)

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)
    artifacts = {
        "sklearn_model" : sklearn_model_path,
        "data" : data_dir
    }

    class SklearnWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

        def predict(self, context, model_input):
            return self.sklearn_model.predict(model_input.values)


    # Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(3.10),
            "pip",
            {
                "pip": [
                    "mlflow=={}".format(mlflow.__version__),
                    "scikit-learn=={}".format(sklearn.__version__),
                    "cloudpickle=={}".format(cloudpickle.__version__),
                ],
            },
        ],
        "name": "sklearn_env",
    }

    mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_paths=["custom_models.py"],
        conda_env=conda_env
    )

    mlflow.end_run()  