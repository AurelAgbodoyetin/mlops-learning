# Day 3

## Topic: Basic MLflow Implementation

### Overview
- MLflow can be integrated into standard ML workflows by **adding just a few lines of code**.  
- Example used here:
  - **Dataset**: [Wine Quality Dataset](https://github.com/myuser114/wine-dataset)  
  - **Model**: `ElasticNet` regression model from `scikit-learn`  


### Basic Tracking Mechanism
1. **Import MLflow and the sklearn integration**
```python
   import mlflow
   import mlflow.sklearn
```

2. **Create an experiment**
```python
   experiment = mlflow.set_experiment("wine-quality-elasticnet")
```
3. **Start runs**
```python
   with mlflow.start_run(experiment_id=experiment.experiment_id):
       ...
```
4. **Log parameters, metrics, and the trained model**
```python
   mlflow.log_param("alpha", alpha)
   mlflow.log_param("l1_ratio", l1_ratio)

   mlflow.log_metric("rmse", rmse)
   mlflow.log_metric("r2", r2)
   mlflow.log_metric("mae", mae)

   mlflow.sklearn.log_model(model, "model")
```

### Concepts: Experiments & Runs

* **Experiment**
  * Logical grouping of runs.
  * Allows organization and comparison of multiple runs.
* **Run**
  * A single execution of code.
  * Records code version, hyperparameters, metrics, and artifacts.

### MLflow Tracking Folder Structure

When MLflow Tracking code runs, it automatically creates a **`mlruns/`** directory:

```
mlruns/
│
├── .trash/            # Deleted experiments/runs (recoverable)
│
├── 0/                 # Default experiment folder (if no ID is set)
│   └── <run_id>/      # Each run is stored here
│       ├── meta.yaml  # Experiment metadata (id, name, creation time, etc.)
│       ├── artifacts/ # Stored artifacts (empty if none logged)
│       ├── metrics/   # One file per logged metric
│       ├── params/    # One file per logged parameter
│       └── tags/      # Optional run tags
│
├── <experiment_id>/   # Additional experiments by ID
│   └── <run_id>/      # Same structure as above
│
└── models/            # Model registry storage
    └── <model_id>/    
        ├── artifacts/     # Contains model.pkl, code, requirements.txt
        ├── metrics/       # Metrics related to the model
        ├── params/        # Parameters related to the model
        ├── meta.yaml      # Model metadata
        └── tags/          # Tags for model tracking
```

## Key Takeaways

* MLflow makes experiment tracking **easy to set up and lightweight**.
* Experiments organize multiple runs; runs log **code, params, metrics, and artifacts**.
* The `mlruns/` folder is automatically created for storing **all tracking metadata**.
* Models logged with MLflow are **packaged with code, requirements, and metadata** → ready for deployment.
