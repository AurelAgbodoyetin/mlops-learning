# Day 7

## Topic: MLflow Logging Functions and Batch Operations

## 1. Logging Parameters

### `mlflow.log_param()`

**Purpose**

* Logs a **single parameter** as a key-value pair.

**Parameters**

* `key` *(str)* → Parameter name (supports alphanumeric, `_ - . / space`).
* `value` → Parameter value used in training.

**Return Value**

* Returns the logged value.

### `mlflow.log_params()`

**Purpose**

* Logs **multiple parameters** at once.

**Parameters**

* `params` *(dict)* → Dictionary of key-value pairs.

**Return Value**

* None.


## 2. Logging Metrics

### `mlflow.log_metric()`

**Purpose**

* Logs a **single metric** (e.g., RMSE, MAE, R²).

**Parameters**

* `key` *(str)* → Metric name.
* `value` *(float)* → Metric value.
* `step` *(int, optional)* → Defaults to `0`. Useful for tracking across epochs/steps.


**Return Value**

* None.


### `mlflow.log_metrics()`

**Purpose**

* Logs **multiple metrics** at once.

**Parameters**

* `metrics` *(dict)* → Dictionary of metric key-value pairs.
* `step` *(int, optional)* → Same as above.

**Return Value**

* None.


## 3. Logging Artifacts

### `mlflow.log_artifact()`

**Purpose**

* Logs a **single artifact file** (e.g., dataset, model, plot, script).

**Parameters**

* `local_path` *(str)* → Path to file.
* `artifact_path` *(str, optional)* → Destination path inside artifacts directory.

**Return Value**

* None.


### `mlflow.log_artifacts()`

**Purpose**

* Logs **all artifacts in a directory**.

**Parameters**

* `local_dir` *(str)* → Path to local directory.
* `artifact_path` *(str, optional)* → Destination path inside artifacts directory.

**Return Value**

* None.


## 4. Retrieving Artifact Paths

### `mlflow.get_artifact_uri()`

**Purpose**

* Returns the **absolute URI** of an artifact or artifact directory.

**Parameters**

* `artifact_path` *(str, optional)* → Run-relative artifact path.

**Return Value**

* URI as a string.

**Behavior**

* Without argument → returns artifact root directory.
* With argument → returns URI of specific artifact.


## 5. Setting Tags

### `mlflow.set_tag()`

**Purpose**

* Sets a **single tag** for the current run.

**Parameters**

* `key` *(str)* → Tag name (≤250 characters).
* `value` *(str)* → Tag value (≤5000 characters).

**Return Value**

* None.


### `mlflow.set_tags()`

**Purpose**

* Sets **multiple tags** at once.

**Parameters**

* `tags` *(dict)* → Dictionary of tag key-value pairs.

**Return Value**

* None.

**System Tags (reserved by MLflow)**

* `mlflow.runName` → Run name.
* `mlflow.source.name` → Source file name.
* `mlflow.source.type` → Execution environment.
* `mlflow.user` → Username.
* `mlflow.log.history` → Model registry metadata.


## 6. Multiple Runs in a Single Program

**Concept**

* Useful for incremental training, checkpointing, hyperparameter tuning, cross-validation.

**Implementation**

* Repeat `start_run()` → training block → `end_run()`.
* Use `active_run()` inside block to fetch run ID/name.
* Use `last_active_run()` after block to fetch most recent run.

**Example**

* Run 1 → `alpha=0.7, l1_ratio=0.7`.
* Run 2 → `alpha=0.9, l1_ratio=0.9`.
* Run 3 → `alpha=0.4, l1_ratio=0.4`.
* Compare metrics across runs.


## 7. Multiple Experiments in a Single Program

**Concept**

* Useful for testing different models, datasets, or feature sets.

**Implementation**

* Use `mlflow.set_experiment()` to define new experiment.
* Run training inside each experiment block.

**Example**

* **Experiment 1** → ElasticNet (α, l1\_ratio).
* **Experiment 2** → Ridge (α).
* **Experiment 3** → Lasso (α).

**Benefit**

* Clear separation of related experiments for easier comparison.


## Key Takeaways

* `log_param(s)` → Log training parameters.
* `log_metric(s)` → Log evaluation metrics.
* `log_artifact(s)` → Log datasets, models, or files.
* `get_artifact_uri()` → Retrieve artifact locations.
* `set_tag(s)` → Add metadata for filtering/grouping.
* Multiple runs → Track variations within one experiment.
* Multiple experiments → Organize runs across models/datasets.
