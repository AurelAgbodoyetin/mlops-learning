# Day 1

## Topic: MLOps Fundamentals

### ML Project Lifecycle
1. **Business understanding & requirements gathering**  
2. **Data acquisition**  
3. **Data preparation**  
4. **Modeling** (data science team)  
5. **Deployment** (operations team)  
6. **Monitoring & maintenance**  

### What is MLOps?
- **MLOps = Machine Learning + Operations**  
- A set of **principles and practices** that standardize and streamline the **ML lifecycle management**.  
- Extends **DevOps principles** (CI/CD) with **Continuous Training (CT)**, which is unique to ML systems.


### Why MLOps is Needed
- Traditional ML workflows often break down when moving from **research (modeling)** to **production (deployment & monitoring)**.
- Without MLOps:
  - Communication between **data science** and **operations teams** becomes messy.
  - Reproducibility, scalability, and monitoring are often lacking.
  - Manual processes slow down deployment and retraining.
- MLOps addresses these challenges by providing a structured approach to manage the entire ML lifecycle.


### Activities to Productionize a Model
- **Build & test locally**  
- **Package**: Compile code, resolve dependencies  
- **Performance**: Ensure prediction speed, GPU optimization, scalability  
- **Instrument**: Version code, data, features, environment → ensure reproducibility  
- **Automate**: Enable Continuous Training (reduce manual work)  


### Key MLOps Principles
- 📑 **Documentation**: Notebooks showcase core functionalities  
- 🗂 **Version control**: Code, data, environments, artifacts  
- 🐳 **Containerization & distributed computing**  
- 🔄 **Workflow orchestration**: Build pipelines around MLOps processes  
- 📊 **Monitoring**: Track data drift, feature changes, latency, uptime  
  - Tools: **Grafana, Prometheus**  
- ⚡ **Automation**: Training triggered by events or scheduled retraining  


```mermaid
mindmap
  root((MLOps Fundamentals))
    What is MLOps?
      "ML (model development) + Ops (deployment & monitoring)"
      "Extends DevOps with Continuous Training"
    Why MLOps is Needed
      "Bridges research → production gap"
      "Team communication issues"
      "Reproducibility & scalability challenges"
      "Manual workflows slow deployment"
    ML Project Lifecycle
      "Business understanding & requirements"
      "Data acquisition"
      "Data preparation"
      "Modeling (data science team)"
      "Deployment (operations team)"
      "Monitoring & maintenance"
    Productionizing a Model
      "Build & test locally"
      "Package (resolve dependencies)"
      "Performance: speed, GPU, scalability"
      "Instrumentation: version code, data, env"
      "Automation with Continuous Training"
    Key Principles
      "Documentation (notebooks)"
      "Version control: code, data, env, artifacts"
      "Containerization & distributed computing"
      "Workflow orchestration (pipelines)"
      "Monitoring (Grafana, Prometheus)"
      "Automated training (triggered/scheduled)"



## Key Takeaways
- MLOps ensures **scalable, reproducible, and automated ML workflows**.  
- It bridges the gap between **data science research** and **production deployment**.  
- Core principles: **Versioning, containerization, monitoring, and automation**.  

