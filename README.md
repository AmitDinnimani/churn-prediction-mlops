# 📊 ChurnFlow: End-to-End MLOps Pipeline for Churn Prediction

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [MLOps Workflow](#mlops-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Performance](#models--performance)
- [Monitoring & Observability](#monitoring--observability)
- [Technical Implementation](#technical-implementation)
- [Results & Validation](#results--validation)
- [Resume Highlights](#resume-highlights)

---

## 🎯 Overview

**ChurnFlow** is a production-ready MLOps system built to predict customer churn with high precision. Unlike standalone models, this project implements a **complete machine learning lifecycle**, including automated data validation, experiment tracking with MLflow, containerized infrastructure, and real-time inference via FastAPI.

The core objective is to ensure **reproducibility and scalability**, allowing data scientists to move from research to production seamlessly while maintaining model quality through automated drift detection.

### 🎓 Why This Project?

- **Real-World Impact**: Customer churn is a critical business metric; this project provides an actionable solution.
- **Full Lifecycle**: Covers everything from data ingestion to model monitoring.
- **Tooling Excellence**: Uses industry-standard tools like MLflow, Docker, and FastAPI.
- **Production Standards**: Follows best practices in CI/CD, logging, and automated testing.

---

## ✨ Features

### Core Capabilities

- ✅ **Automated Training Pipeline**: Scripts to train, evaluate, and register models automatically.
- ✅ **Multi-Model Support**: Compares XGBoost, Random Forest, and Logistic Regression.
- ✅ **Experiment Tracking**: Deep integration with **MLflow** for hyperparameter and metric logging.
- ✅ **Model Registry**: Automated promotion of "best-in-class" models to Production stage.
- ✅ **Robust API**: RESTful endpoints for real-time risk scoring and predictions.
- ✅ **Data Validation**: Automated schema checks using custom validation logic.
- ✅ **Dockerized Environment**: Fully containerized stack for consistent deployment.

### Advanced Features

- 🔬 **Data Drift Detection**: Integrated **Evidently AI** for monitoring feature distributions.
- ⚡ **Unified Pipelines**: Encapsulated preprocessing and scaling within the model artifact.
- 🐳 **Service Orchestration**: Complex multi-service setup (MLflow + Postgres + API).
- 🧪 **Comprehensive Testing**: Unit and integration tests for pipeline reliability.

---

## 🏗 `project/` Architecture

```
churn-prediction-mlops/
│
├── src/                          # Core source code
│   ├── data/                     # Data loading and validation logic
│   ├── models/                   # Evaluation and registry wrappers
│   ├── utils/                    # Shared utilities (logger, config)
│   └── api/                      # FastAPI implementation
│
├── scripts/                      # Operational scripts
│   ├── train_model.py            # Main training and registration entry point
│   └── run_inference.py          # Ad-hoc inference script
│
├── docker/                       # Containerization configurations
│   ├── docker-compose.yml        # Orchestration for MLflow & Postgres
│   └── Dockerfile.api            # API service container
│
├── configs/                      # Pipeline and model configurations
├── notebooks/                    # EDA and experimentation workspace
├── tests/                        # Pytest suite
└── Makefile                      # Automation commands
```

---

## 🏗 MLOps Workflow

1.  **Data Ingestion**: Loads raw data from specified storage paths.
2.  **Validation**: Ensures features meet expected data types and ranges.
3.  **Experimentation**: Trains multiple models, logging parameters to MLflow.
4.  **Evaluation**: Computes ROC-AUC, F1-Score, and Precision-Recall metrics.
5.  **Registration**: Automatically promotes the best-performing model to the Model Registry.
6.  **Serving**: FastAPI serves the "latest" production model artifact.
7.  **Monitoring**: Generates drift reports to signal when retraining is needed.

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.9+
Docker & Docker-Compose
Make (optional but recommended)
```

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/AmitDinnimani/churn-prediction-mlops.git
cd churn-prediction-mlops

# Install dependencies and pre-commit hooks
make install

# Start MLOps Infrastructure (MLflow, Postgres)
docker-compose up -d
```

---

## 💻 Usage

### 1. Execute Training Pipeline

```bash
python scripts/train_model.py
```
*This will validate data, train models, log to MLflow, and register the best one.*

### 2. Run API Locally

```bash
uvicorn src.api.main:app --reload
```

### 3. Automation Commands

```bash
make format    # Auto-format code (Black, Isort)
make lint      # Check for style issues (Flake8)
make test      # Run test suite
```

---

## 🧠 Models & Performance

| Model | Type | Use Case | Key Hyperparameters |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Boosting | High-performance prediction | n_estimators, max_depth, lr |
| **Random Forest** | Bagging | Robustness & interpretation | n_estimators, max_features |
| **Logistic Reg** | Linear | Baseline & probability | solver, max_iter |

### Evaluation Suite
- **ROC-AUC**: Primary metric for churn risk separation.
- **F1-Score**: Balance between Precision and Recall for imbalanced churn data.
- **Log-Loss**: Used during optimization for soft-assignment calibration.

---

## 🔬 Technical Implementation

### 1. MLflow Integration Strategy
- **Tracking**: Every run captures code version (Git hash), parameters, and metrics.
- **Artifacts**: Stores the `joblib` serialized pipeline directly in the MLflow artifact store.
- **Registry API**: Uses `mlflow.register_model()` to programmatically manage model stages (Staging -> Production).

### 2. Unified Sklearn Pipelines
To solve the classic **"Training-Serving Mismatch"**, preprocessing (scaling, imputation) is baked into the model artifact. When the API loads the model, it receives raw JSON and handles preprocessing internally via the loaded pipeline.

### 3. Containerized Infrastructure
Orchestrated a 3-tier MLOps stack:
- **MLflow Tracking Server**: Centralized UI for experiments.
- **PostgreSQL**: Backend store for experiment metadata.
- **S3/Local Artifact Store**: Storage for model weights.

---

## 📈 Results & Validation

### Validation Strategy
- **Cross-Validation**: 5-fold CV used to ensure model stability.
- **Hold-out Test Set**: Final evaluation on unseen 20% split.
- **Consistency Check**: API predictions are verified against local training results to ensure parity.

---

## 🎓 Resume Highlights

### **MLOps Engineer / Data Scientist**

**Churn Prediction MLOps Pipeline | Python, MLflow, FastAPI, Docker**

- Engineered an end-to-end MLOps pipeline for churn prediction, automating data validation, multi-model training (XGBoost, RF), and model registration using **MLflow**.
- Developed a high-performance **FastAPI** inference service, reducing prediction latency by loading unified Sklearn pipelines that encapsulate both preprocessing and inference logic.
- Orchestrated a containerized infrastructure using **Docker-Compose**, integrating PostgreSQL for experiment tracking and automated model versioning.
- Implemented automated monitoring with **Evidently AI**, enabling data drift detection and providing insights into feature distribution shifts.
- Standardized the development workflow using **Makefile** automation for testing (Pytest), linting, and formatting, ensuring high code quality and reproducibility.

**Key Technical Skills Demonstrated:**
- MLOps lifecycle management & Automation
- Experiment tracking & Model Versioning
- API Development & Model Serving
- Containerization & Infrastructure-as-Code
- Data validation & Monitoring

---

**⭐ If you found this project helpful, please star the repository!**

---
*Created by [Amit Dinnimani](https://github.com/AmitDinnimani) — Building reliable ML systems.*
