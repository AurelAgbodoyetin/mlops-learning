# Day 8

## Topic: MLflow Autologging

## 1. Purpose

* Automates logging of parameters, metrics, artifacts, and metadata.
* Captures code version, framework, git commit, and dataset information.
* Reduces need for manual calls (`log_param`, `log_metric`, `log_artifact`).


## 2. Methods

### Generic Autologging

* `mlflow.autolog()`
* Enables autologging for **all supported libraries** detected in the workflow.
* Supports libraries like Keras/TensorFlow, LightGBM, Paddle, PySpark, PyTorch, Scikit-learn, Spark, Statsmodels, XGBoost

### Library-Specific Autologging

* Example: `mlflow.sklearn.autolog()`
* Restricts autologging to a **single library** (e.g., Scikit-learn).


## 3. Parameters – Generic `mlflow.autolog()`

* **log\_models** *(bool, default=True)* → Log trained model artifacts.
* **log\_input\_examples** *(bool)* → Log sample input data (requires `log_models=True`).
* **log\_model\_signatures** *(bool)* → Log model input/output schema (requires `log_models=True`).
* **log\_datasets** *(bool)* → Log training and evaluation datasets.
* **disable** *(bool)* → Disable autologging.
* **exclusive** *(bool)* → Log only in autolog-created runs (ignore user-created runs).
* **disable\_for\_unsupported\_versions** *(bool)* → Disable if library version unsupported.
* **silent** *(bool)* → Suppress event logs and warnings.

⚠️ **Important**: Must be called **before `fit()`** for proper instrumentation.


## 4. Notes

* Logs both **default parameters** and **user-specified ones**.
* Captures **framework-specific metrics** automatically.
* For **custom models** (e.g., custom losses/metrics), manual logging may still be required.


## 5. Library-Specific Example – `mlflow.sklearn.autolog()`

Additional parameters available for Scikit-learn:

* **max\_tuning\_runs** *(int, default=5)* → Limit number of hyperparameter search runs logged.
* **log\_post\_training\_metrics** *(bool, default=True)* → Log metrics such as MAE, RMSE, R².
* **serialization\_format** *(str)* → Choose between `pickle`, `cloudpickle`, or `joblib`.
* **registered\_model\_name** *(str)* → Automatically register the model under the specified name.
* **pos\_label** *(int/str, default=1)* → Positive class label for binary classification tasks.


## Key Takeaways

* Use `mlflow.autolog()` for **end-to-end simplicity** across supported libraries.
* Use library-specific functions (e.g., `mlflow.sklearn.autolog()`) for **fine-grained control**.
* Ideal for **standard model training pipelines**.
* For **custom models or metrics**, complement autologging with manual logging.
