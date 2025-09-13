# Day 5

## Topic: MLflow Experiment Management

## 1. `mlflow.create_experiment()`

### Purpose
- Creates a **new experiment** for managing and tracking ML workflows.

### Behavior
- Always creates a new experiment with a **unique name**.
- If **name already exists** → raises an exception.

### Parameters
- **name** *(required)* → Unique, case-sensitive experiment name.  
- **artifact_location** *(optional)* → Path where artifacts are stored.  
  - Default: same location as `set_tracking_uri`.  
  - Can specify a **custom artifact directory**.  
- **tags** *(optional)* → Dictionary of key-value pairs (e.g., version, priority).  

### Return Value
- Returns an **experiment ID**, which can be passed to `mlflow.start_run()`.  

### Additional Function
- `mlflow.get_experiment(experiment_id)` → Retrieves experiment details:  
  - Name, ID, artifact location, tags, lifecycle stage, creation timestamp.  


## 2. `mlflow.set_experiment()`

### Purpose

* Use an **already existing experiment** for new runs.

### Behavior

* If experiment **exists** → sets it active.
* If experiment **name doesn’t exist** → creates a new experiment with that name.
* If experiment **ID doesn’t exist** → raises an exception.

### Parameters

* **experiment\_name** *(string)* → Name of existing experiment.
* **experiment\_id** *(string)* → ID of existing experiment.

### Return Value

* Returns an **`mlflow.Experiment` object** containing:
  * ID, name, artifact location, tags, etc.

## Key Takeaways

* **`create_experiment`** → Always creates a new experiment (returns **experiment ID**).
* **`set_experiment`** → Activates an existing experiment (returns **experiment object**).
* Key distinction:
  * Name not found → Creates new experiment.
  * ID not found → Throws error.
* Artifacts can be stored in the **default `mlruns/` folder** or in **custom paths**.
* Use `mlflow.get_experiment(experiment_id)` to retrieve experiment details.
