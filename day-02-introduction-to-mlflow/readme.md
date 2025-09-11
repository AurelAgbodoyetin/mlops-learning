# Day 2

## Topic: Introduction to MLflow

### What is MLflow?
- **MLflow** is an open-source platform for managing the **end-to-end machine learning lifecycle**.  
- It covers:
  - Experimentation  
  - Reproducibility  
  - Deployment  
  - Central model registry  


### The 4 Key Components of MLflow

#### 1. Tracking
- Records parameters, metrics, code versions, and experiment results.  
- Provides a **central place** to log and track experiments.  
- Includes a **UI** for visualization and comparison.  
- Supports both **local and remote tracking servers**.  

#### 2. Projects
- Packages ML code to ensure **reusability and reproducibility**.  
- Simplifies running experiments in different environments.  
- Uses a **YAML configuration file** (`MLproject`) for dependencies and entry points.  
- Enables sharing of reproducible experiments across teams.  

#### 3. Models
- Provides a **standard format** for packaging trained models.  
- Streamlines the **deployment process**.  
- Supports multiple deployment options (local REST API, cloud, edge).  
- Ensures model portability across frameworks.  

#### 4. Registry
- A **centralized repository** for managing ML models.  
- Supports:
  - **Versioning** (track multiple versions of a model)  
  - **Stage transitions** (e.g., Staging → Production → Archived)  
  - **Collaboration** (team access and governance)  
- Allows **model search, filtering, and metadata management**.  


### MLflow Integrations
- Easily integrates into **existing ML projects**.  
- Compatible with major ML libraries and frameworks:  
  - **TensorFlow**  
  - **Keras**  
  - **PyTorch**  
  - **Scikit-learn**  
  - **Apache Spark**  
  - ...and more.


## Key Takeaways
- MLflow provides a **modular and flexible** solution for managing ML projects.  
- Its four components (**Tracking, Projects, Models, Registry**) cover the **full ML lifecycle**.  
- Adoption ensures **reproducibility, collaboration, and scalable deployment**.  