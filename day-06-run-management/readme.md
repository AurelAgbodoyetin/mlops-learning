# Day 6

## Topic: MLflow Run Management

## 1. `mlflow.start_run()`

### Purpose

* Starts a **new run** or **resumes an existing run**.

### Parameters

* **run\_id** *(optional)* → Resume existing run, status set to *RUNNING*, logs to same run.
* **experiment\_id** *(optional)* → Assigns run to an experiment.

  * Only valid if `run_id` is not specified.
  * Order of precedence for experiment selection:

    1. Experiment from `set_experiment()` / `create_experiment()`.
    2. Env var: `MLFLOW_EXPERIMENT_NAME`.
    3. Env var: `MLFLOW_EXPERIMENT_ID`.
    4. Default experiment (tracking server default).
* **run\_name** *(optional)* → Assigns name to new run.
* **nested** *(optional, bool)* → Enables **nested runs** (child runs under parent).
* **tags** *(optional, dict)* → Metadata tags for grouping runs.
* **description** *(optional, str)* → Run description.

### Return Value

* Returns an **`mlflow.active_run` object** (context manager).

### Usage

* If using the **with block** → run auto-closes.
* If not using the **with block** → must close manually using `mlflow.end_run()`.

## 2. `mlflow.end_run()`

### Purpose

* Ends the currently active run.

### Parameters

* **status** *(optional, str)* → Default = `"FINISHED"`.

  * Options: `"RUNNING"`, `"SCHEDULED"`, `"FAILED"`, `"KILLED"`.

### Return Value

* None.

### Usage

* Needed if not using the `with` block.
* Useful to explicitly set final run status.


## 3. `mlflow.active_run()`

### Purpose

* Returns the **currently active run**.

### Return Value

* Run object or `None` (if no run is active).

## 4. `mlflow.last_active_run()`

### Purpose

* Returns the **most recent active run**.


### Return Value

* Run object.

### Behavior

* Inside active run → returns current run.
* After run ended → returns last completed run.


## Key Takeaways

* `start_run()` → Start or resume a run (`run_id`, `experiment_id`, `run_name`, `nested`, `tags`, `description`).
* `end_run()` → End run manually, optionally setting status.
* `active_run()` → Retrieve the currently running run.
* `last_active_run()` → Retrieve last finished run (or current if active).

