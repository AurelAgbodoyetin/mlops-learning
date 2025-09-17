# Day 9

## Topic: MLflow Tracking Server


## 1. Overview

### Purpose

* The **tracking server** is a centralized MLflow service for storing and sharing experiment metadata and artifacts.
* Moves beyond local logging (default: `./mlruns`) to enable **scalable, collaborative, and secure experiment tracking**.

### Benefits

* Enterprise-ready: multiple users, remote access, and centralized storage.
* Supports experiment reproducibility and model comparison.
* Secure integration with artifact stores (local, cloud, hybrid).


## 2. Core Components

### (a) Storage

* **Backend Store** → Stores experiment metadata (experiments, runs, params, metrics, tags).

  * Options:

    * File store (local filesystem, cloud storage).
    * Database store (SQLite, MySQL, PostgreSQL).

* **Artifact Store** → Stores large files (trained models, datasets, visualizations, outputs).

  * Options:

    * Local directories.
    * Cloud storage (S3, Azure Blob, GCP buckets).

### (b) Networking / Communication

* Supports **REST API** (HTTP) and **gRPC** (faster, bi-directional).
* SDKs available in Python, Java, R.
* Can **proxy artifact storage** for secure setups.


## 3. Tracking Server Setup

### Basic Command

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow_artifacts \
  --host 127.0.0.1 \
  --port 5000
```

### Key Flags

* **`--backend-store-uri`** → Metadata DB (SQLite, MySQL, PostgreSQL).
* **`--default-artifact-root`** → Artifact location (local/cloud).
* **`--host`, `--port`** → Server address.
* **UI** → Accessible at `http://host:port`.


## 4. Configuration Scenarios

### Scenario 1: Localhost (default)

* Metadata + artifacts in `./mlruns`.
* Setup: automatic.
* Use case: development, small projects.

### Scenario 2: Localhost with SQLite

* Metadata stored in SQLite DB (`sqlite:///mlflow.db`).
* Artifacts stored locally.
* Use case: lightweight persistence.

### Scenario 3: Localhost with Dedicated Tracking Server

* MLflow server running locally (port 5000).
* Metadata: file store / SQLite DB.
* Artifacts: local filesystem.
* Communication: REST API.

### Scenario 4: Remote Tracking Server (Distributed)

* Remote server hosting:

  * Backend store (e.g., PostgreSQL on AWS RDS).
  * Artifact store (e.g., S3 bucket).
* Clients communicate via REST API.
* Use case: scalable, multi-user collaboration.

### Scenario 5: Tracking Server with Proxied Artifact Access

* Server proxies access to restricted artifact store.
* Benefits:

  * No direct credentials needed for users.
  * Centralized authentication.
* Use case: secure enterprise setups.

### Scenario 6: Tracking Server as Artifact Proxy Only

* Server only handles artifact-related API calls.
* No metadata logging.
* Started with `--artifacts-only` flag.
* Use case: central artifact management.


## 5. Key Benefits

* Centralized experiment tracking and artifact management.
* Easy model comparison across runs and versions.
* Team-wide collaboration with controlled access.
* Works with multiple ML frameworks and languages.
* Flexible deployment: local, remote, on-prem, or cloud.


## Key Takeaways

* **MLflow tracking server** = scalable, centralized service for experiment tracking.
* Splits storage into:

  * **Backend store** (metadata).
  * **Artifact store** (models, outputs, datasets).
* Communication supported via **REST API** and **gRPC**.
* Six deployment scenarios, from **simple localhost** to **enterprise-level with security proxying**.