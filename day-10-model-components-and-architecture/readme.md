# Day 10

## Topic: MLflow Models

## 1. Overview

### Purpose

* **MLflow Models** provide a standard for packaging and deploying ML models across environments.
* Solve problems of **manual deployment**, reproducibility, and inconsistent environments.

### Benefits

* **Reproducibility** → identical dev/prod environments.
* **Collaboration** → models + lineage easily shared.
* **Flexibility** → deploy to diverse targets (local, cloud, containers, edge).


## 2. Core Components

### (a) Storage Format

* Defines **how a model is saved**.
* Includes: model object, metadata, hyperparameters, environment files.
* Supported formats:

  * **Directory of files** (default).
  * **Single-file format**.
  * **Python functions**.
  * **Container images** (e.g., Docker).

**Directory Format Example**:

* `MLmodel` → central config file.
* `model.pickle` → serialized model.
* `input_example.json` → sample inputs.
* `conda.yaml`, `python_env.yaml`, `requirements.txt` → reproducible environments.

### (b) Model Signature

* Specifies **input/output schema** (data types, shapes).
* Ensures valid inference and correct deployment.
* Supports simple (int, str) and complex (arrays, DataFrames).
* Used to **auto-generate REST API endpoints**.

### (c) Model API

* Provides a **REST API** from the model’s signature.
* Supports:

  * Synchronous + asynchronous requests.
  * Real-time inference.
  * Batch predictions.
* Works across **cloud, edge, and on-prem deployments**.
* Supports **versioning** for continuous model evolution.


### (d) Flavors

* A **flavor = framework-specific serialization format**.
* Built-in flavors: Scikit-learn, PyTorch, TensorFlow, XGBoost, etc.
* Custom flavors can be defined.

**Example**:

* Scikit-learn model saved with both:

  * `sklearn` flavor.
  * `python_function` flavor.
* Allows flexible model loading in different contexts.


## 3. The `MLmodel` File

### Purpose

* Central **YAML config file** that describes model packaging and metadata.

### Key Fields

* **flavors** → supported frameworks for loading.
* **loader\_module** → module for serving.
* **code** → code path or Git commit ID (traceability).
* **pickled\_model** → serialized object.
* **serialization\_format** → e.g., `cloudpickle`.
* **framework\_version** → training library version.
* **mlflow\_version** → MLflow version used.
* **run\_id** → source run that logged the model.
* **signature** → input/output schema.
* **saved\_input\_example\_info** → reference to input example.
* **uid** → unique identifier.


## 4. Reproducibility via Environment Files

* **`conda.yaml`** → conda-based setup.
* **`python_env.yaml`** → pip-based setup.
* **`requirements.txt`** → minimal dependency list.
* Ensures teams can **replicate exact training environment**.


## 5. Evaluation + Deployment

### Evaluation

* Supports metrics like: Accuracy, Precision, Recall, F1, RMSE, R².

### Deployment Targets

* Docker containers.
* REST API endpoints.
* TensorFlow Serving.
* AWS SageMaker.
* On-prem / private cloud servers.


## Key Takeaways

* **MLflow Models** provide a standard for packaging, sharing, and deploying models.
* Core building blocks:

  * **Storage format**.
  * **Model signature**.
  * **Model API**.
  * **Flavors**.
  * **MLmodel file**.
* Enable reproducibility through **environment files**.
* Support deployment to **local, cloud, and enterprise infrastructures**.
