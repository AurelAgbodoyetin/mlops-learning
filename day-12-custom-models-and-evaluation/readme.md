# Day 12

## Topic: Model Flavors, Customisation, and Evaluation

## 1. MLflow Model Flavors and Customisation

### 1.1 Overview of Flavors

* **Definition**: Flavors standardise how a model and its metadata are stored/served.
* **Types**:
  * **Built-in Flavors** → ready-to-use (scikit-learn, TensorFlow, PyTorch).
  * **Custom Flavors** → define own save/load logic.
  * **Community Flavors** → open-source extensions.

### 1.2 When Customisation is Needed

* ML library not supported by MLflow.
* Built-in utilities insufficient for packaging inference logic.
* Solutions:

  * **Custom Python Models**
  * **Custom Flavors**


### 1.3 Custom Python Models

* **Purpose**: Package unsupported models & custom logic while leveraging MLflow tracking + deployment.
* **Key Module**: `mlflow.pyfunc` (used with `save_model` / `log_model`).
* **Outcome**: Model packaged with the **`pyfunc` flavor** → deployable to SageMaker, Azure ML, etc.

**Steps**:

1. Save model artifact (e.g., pickle with `joblib`).
2. Create wrapper class:
   * Inherit from `mlflow.pyfunc.PythonModel`.
   * Implement `load_context()` (load model/deps).
   * Implement `predict()` (inference logic).
3. Define Conda environment (`conda.yaml`).
4. Define artifacts (paths to pickled model, datasets, etc.).
5. Log with `mlflow.pyfunc.log_model()`.

**Loading & Inference**:

* Load → `mlflow.pyfunc.load_model()`.
* Predict → `.predict()` on loaded model.


### 1.4 Custom Flavors

* **Purpose**: Build your own flavor with custom serialization/deserialization.
* **Distinction**: Custom Python Models → `pyfunc` flavor.
  Custom Flavor → your own flavor format.
* **Complexity**: Advanced; often unnecessary since `pyfunc` is flexible.

**Steps**:

1. Implement save/load logic.
2. Create flavor directory + `MLmodel` file.
3. Register flavor under `mlflow/models`.
4. (Optional) Add custom utilities for serving.

## 2. MLflow Model Evaluation

### 2.1 Introduction

* **Purpose**: Assess performance on unseen data; compare models.
* **API**: `mlflow.evaluate`.
* **Features**:

  * Compute metrics (accuracy, F1, MSE, etc.).
  * Generate plots (confusion matrix, ROC curve).
  * Provide explanations (SHAP, feature importance).
  * Log all results to Tracking Server.
* **Supported Flavor**: `pyfunc`.


### 2.2 `mlflow.evaluate` Parameters

* `model` → model URI / instance.
* `data` → evaluation dataset.
* `targets` → true labels (array or DataFrame column).
* `model_type` → "classifier" / "regressor".
* `evaluators` → e.g., `"default"`.
* `validation_thresholds` → dict for metric acceptance/rejection.
* `baseline_model` → optional for comparisons.


### 2.3 Comparing Runs in MLflow UI

* **Process**: Run multiple experiments → select runs → "Compare".
* **Report Sections**:
  a. **Visualizations**
     * Parallel Coordinates Plot (multi-variate patterns).
     * Scatter Plot (variable relationships).
     * Box Plot (distribution summary).
     * Contour Plot (3D trends in 2D).
  b. **Run Details** → metadata (start time, duration).
  c. **Parameters** → hyperparameters comparison.
  d. **Metrics** → side-by-side metric values.
  e. **Tags** → compare run tags.


## Key Takeaways

* **Flavors** standardise model storage/serving.
  * Built-in, Custom Python Models (`pyfunc`), or Custom Flavors.
* **Custom Python Models** allow packaging unsupported libraries into MLflow.
* **Custom Flavors** → advanced, define own save/load logic.
* **Model Evaluation (`mlflow.evaluate`)** provides metrics, plots, and explanations, fully integrated with MLflow Tracking.
* MLflow UI supports rich run comparisons across metrics, parameters, and visualisations.