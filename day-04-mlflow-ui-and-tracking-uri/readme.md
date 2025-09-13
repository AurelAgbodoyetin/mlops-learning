# Day 4

## Topic: MLflow UI and Logging Functions


## 1. MLflow UI Overview

### Purpose
- Visualize **runs, experiments, and models** directly in a web browser.  

### Accessing the UI
1. Activate the correct **conda environment**.  
2. Run:
```bash
   mlflow ui
```
3. Copy the link from the terminal → open in browser.

### UI Sections

#### Experiments

* Lists experiments (default + custom).
* **Metadata shown**: Experiment ID, artifact location.
* **Runs table**: Displays all executions with customizable columns.
  * Default run names auto-generated (can be customized).
  * Views available:
    * Table
    * Chart
    * Artifact
  * Sort by: Creation date, user, run name, source version, metrics, etc.
  * Columns: Datasets, models, metrics, parameters (customizable).

**Run comparison**:
* **Naive**: Compare metrics/params directly in runs table.
* **Advanced**: Select runs → click **Compare** → visualization options:
  * Parallel coordinate plot
  * Scatter plot
  * Box plot
  * Contour plot

#### Models

* Displays only **registered models** (empty if none are registered).

### Run Details

* Information displayed per run:
  * Run ID, run name, date, source, status, lifecycle stage.

* Options:
  * **Log model** → For experiment tracking only.
  * **Register model** → Promotes model for production in a centralized registry.

## 2. Logging Functions in MLflow

### Tracking URI

* **Functions**:

  ```python
  mlflow.set_tracking_uri(uri)   # Sets storage location for run tracking
  mlflow.get_tracking_uri()      # Retrieves current tracking URI
  ```
* **Default behavior**:
  * Without setting, MLflow logs to the local `mlruns/` folder.
  * Using `""` (empty string) behaves the same.

### `set_tracking_uri` Options

#### Local Storage

* Default: `mlruns/` in current directory.
* Custom folder:

  ```python
  mlflow.set_tracking_uri("./mytracks/")
  ```
* File path:

  ```python
  mlflow.set_tracking_uri("file:/full/path/to/dir")
  ```

  ⚠️ Only works on the **C drive** (not D/E).

#### Remote Storage

* HTTP/HTTPS tracking server:

  ```
  https://<server>:<port>
  ```
* Databricks workspace URI also supported.

## Key Takeaways

* **MLflow UI** provides interactive visualization for experiments, runs, and registered models.
* **Run comparison tools** (parallel coordinates, scatter, box, contour) help analyze results.
* **Tracking URIs** allow flexibility in storing experiment metadata locally or remotely.