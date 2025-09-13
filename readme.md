# MLOps Learning Journey 🚀

My hands-on learning experience mastering MLOps with MLflow.

## 📋 Overview

This repository contains all code, experiments, and projects from my intensive MLOps learning journey. Each day's work is organized into dedicated folders with practical implementations, notes, and key learnings.

## 🌟 Why this mattered to me

Before starting this journey into MLOps, most of my ML projects looked “complete” only on paper, but in practice, they had serious cracks. Here are a few pain points I kept hitting:

* **Lost experiments:** I’d tweak hyperparameters, rerun a model, and then forget which version actually worked best.  My results lived in scattered notebooks, terminal logs, and sometimes in random text files I wrote to keep track. After a few batches of experiments, those logs became a messy jungle I didn’t want to revisit.
* **Fragile code:** My projects often only worked on *my machine*. Sharing them meant a maze of missing dependencies and broken environments.
* **No clear lifecycle:** I would jump straight from dataset → model training → accuracy score, skipping the discipline of versioning, tracking, or deployment planning.
* **One-off mentality:** Models were trained once, “good enough,” and then abandoned—no monitoring, no retraining strategy.
* **Reproducibility gap:** Coming back to an old project, I sometimes couldn’t reproduce my own results, even with the same data.

This repo is my attempt at breaking that cycle—building not just models, but repeatable, shareable, and production-aware workflows.

## 🗂️ Repository Structure

```
📁 mlops-learning/
├── 📁 day-01-mlops-fundamentals/
├── 📁 day-02-introduction-to-mlflow/
├── 📁 day-03-basic-implementation/
├── 📁 day-04-mlflow-ui-and-tracking-uri/
├── 📁 day-05-experiment-management/
├── 📁 day-06-run-management/
├── 📁 day-07-advanced-logging-and-batch-operations/
├── 📁 day-08-autologging-and-automation/
├── 📁 day-09-tracking-server-architecture/
├── 📁 day-10-model-components-and-architecture/
├── 📁 day-11-model-signatures-and-apis/
├── 📁 day-12-custom-models-and-evaluation/
├── 📁 day-13-advanced-evaluation-and-model-registry/
├── 📁 day-14-mlflow-projects-and-client/
└── 📁 day-15-advanced-client-operations-and-production-project/
```

## 🛠️ Technologies Used

- **MLflow** - Experiment tracking, model management, and deployment
- **Python** - Primary programming language
- **Scikit-learn** - Machine learning models and pipelines
- **Pandas & NumPy** - Data manipulation and analysis

## 🚦 Getting Started

### Prerequisites
```bash
python >= 3.7
pip or conda package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/AurelAgbodoyetin/mlops-learning.git
cd mlops-learning

# Install dependencies
pip install -r requirements.txt

# Verify MLflow installation
mlflow --version
```

### Running Examples
```bash
# Navigate to any day's folder
cd day-03-basic-implementation

# Run the example
python mlflow_with_sklearn.py

# Launch MLflow UI to view results
mlflow ui
```

## 🤝 Contributing

This is a personal learning repository, but feel free to:
- Open issues for questions or discussions
- Submit PRs for improvements or corrections
- Share your own MLOps learning experiences

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**"The journey of mastering MLOps, one experiment at a time."** 📊✨