# Day 11

## Topic: MLflow Model API

## 1. Overview

### Purpose

* The **Model API** manages the full lifecycle of ML models in MLflow: packaging, saving, logging, loading, and deployment.
* Works across multiple frameworks: **Scikit-learn, PyTorch, TensorFlow, XGBoost**, etc.
* Framework-specific modules (e.g., `mlflow.sklearn`) share the **same core functions**.


## 2. Core Functions

### (a) `save_model`

* Saves a model **locally** → creates an MLflow model directory.
* Produces **two flavors**: `mlflow.<framework>` and `mlflow.pyfunc`.
* Typical use: persisting models without experiment logging.

**Key Parameters**

* `sk_model` → model object.
* `path` → save location.
* `conda_env` → environment (dict or YAML).
* `code_paths` → Python training files for reproducibility.
* `signature` → input/output schema.
* `input_example` → representative input sample.
* `pip_requirements` / `extra_pip_requirements` → dependencies.
* `metadata` → custom metadata in `MLmodel`.


### (b) `log_model`

* Logs a model as an **artifact in a run** → stored in Tracking Server and visible in UI.
* Same parameters as `save_model` plus:

  * `artifact_path` → relative path in artifact store.
  * `registered_model_name` → auto-registers in Model Registry.
  * `await_registration_for` → wait time (default = 300s).

📌 **Difference**

* `save_model` → local only.
* `log_model` → Tracking Server + Model Registry option.


### (c) `load_model`

* Loads a saved/logged model for inference.

**Key Parameters**

* `model_uri` → location of model.
* `dst_path` → local download directory (optional).

**Examples of `model_uri`**

* `runs:/<run-id>/<artifact-path>` → from a specific run.
* `models:/<model-name>/<version>` → from registry by version.
* `models:/<model-name>/Production` → latest Production stage.
* `file:/local/path/to/model` → local path.

## 3. Model Signatures

### (a) What is a Signature?

* A **schema** describing model’s input and output.
* Inputs → names, types, shapes.
* Outputs → scalar, vector, or dataframe.

**Benefits**

* Prevents mismatches between training and serving.
* Enforces schema validation at inference.
* Required for REST API and Model Registry deployment.

### (b) Defining a Signature

**Using `infer_signature`:**

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
```

**With `log_model`:**

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=infer_signature(X_train, model.predict(X_train)),
    input_example=X_train.iloc[:5]
)
```

### (c) Signature Parameters

* `signature` → explicit schema.
* `input_example` → representative input (small sample).

If omitted:

* MLflow may infer from `input_example`.
* If neither given → model saved, but schema not enforced.

## Key Takeaways

* **`save_model`** → persist model locally.
* **`log_model`** → log to Tracking Server + optional registry.
* **`load_model`** → retrieve for inference.
* **Signatures** (`infer_signature`) ensure schema consistency.
* **Input examples** document expected inputs.