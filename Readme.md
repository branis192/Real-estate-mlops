# 🏠 Real Estate Price Prediction — MLOps

An end-to-end **Machine Learning and MLOps system** for predicting residential property prices using the **Ames Housing dataset**.

This project demonstrates a production-oriented ML workflow, from **data exploration and preprocessing** to **model training, API serving, containerization, Kubernetes deployment, zero-downtime updates, and real-time monitoring with Prometheus**.

The goal is to showcase how a machine learning model can be transformed from an experimental notebook into a **reproducible, containerized, observable production service**.

---

## 📌 Project Overview

The system predicts residential property prices based on property characteristics such as living area, quality, location, garage information, basement features, and other structural attributes.

The complete workflow includes:

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning & Feature Engineering
     │
     ▼
Scikit-Learn Preprocessing Pipeline
     │
     ▼
Model Training & Evaluation
     │
     ▼
Serialized ML Pipeline
     │
     ▼
FastAPI Inference API
     │
     ▼
Docker Container
     │
     ▼
Kubernetes / Kind
     │
     ├──────────────► Prometheus Monitoring
     │
     ▼
Production-ready ML Service
```

---

# 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| **Language** | Python 3.12+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Model Serialization** | Joblib |
| **API** | FastAPI, Uvicorn |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes, Kind |
| **Monitoring** | Prometheus |
| **API Metrics** | `prometheus-fastapi-instrumentator` |
| **Development** | Jupyter Notebook |

---

# 📁 Project Structure

```text
real-estate-mlops/
│
├── app/
│   └── main.py
│       # FastAPI inference API
│       # Prometheus instrumentation
│
├── data/
│   └── AmesHousing.csv
│       # Raw Ames Housing dataset
│
├── k8s/
│   ├── deployment.yaml
│   │   # Kubernetes deployment for the API
│   │   # 2 replicas + health probes
│   │
│   ├── service.yaml
│   │   # ClusterIP service exposing the API
│   │
│   ├── prometheus-configmap.yaml
│   │   # Prometheus scraping configuration
│   │
│   └── prometheus-deployment.yaml
│       # Prometheus deployment and service
│
├── models/
│   └── model_pipeline.joblib
│       # Serialized Scikit-Learn pipeline
│
├── notebooks/
│   ├── explore.ipynb
│   │   # Exploratory Data Analysis
│   │
│   └── baseline_model.ipynb
│       # Model experimentation and benchmarking
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   │   # Production preprocessing pipeline
│   │
│   └── train.py
│       # Model training and serialization
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🧹 Data Engineering & Preprocessing

A major objective of this project is to ensure that preprocessing is **reproducible and consistent between training and inference**.

All preprocessing operations are encapsulated inside a Scikit-Learn `ColumnTransformer` and serialized together with the model.

This prevents discrepancies between the preprocessing used during training and the preprocessing applied to production predictions.

## Data Cleaning

The dataset was analyzed during the exploratory phase to identify problematic observations and domain-specific issues.

One of the main treatments was the removal of extreme observations where:

```text
Gr Liv Area > 4000 sq ft
```

This treatment reduces the impact of extreme outliers on the regression model.

---

## 🎯 Target Transformation

The target variable `SalePrice` is highly right-skewed.

A logarithmic transformation is therefore applied:

```python
np.log1p(SalePrice)
```

This transformation helps:

- Reduce target skewness
- Stabilize variance
- Reduce the influence of extreme prices
- Improve the behavior of linear regression
- Produce a more suitable target distribution for modeling

During evaluation, predictions are converted back to the original price scale before calculating MAE.

---

## 🔢 Numerical Features

Numerical variables are processed through a dedicated preprocessing pipeline.

The main operations include:

- Missing-value imputation
- Domain-specific handling of structural features
- Median imputation for remaining missing numerical values
- Standardization using `StandardScaler`

Conceptually:

```text
Numerical Features
       │
       ▼
Missing Value Handling
       │
       ▼
Median Imputation
       │
       ▼
StandardScaler
```

Structural variables where the absence of a feature represents a meaningful value are handled accordingly, for example:

```text
No garage → 0
No basement → 0
No pool → 0
```

---

## 🔤 Categorical Features

Categorical variables are processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing includes:

- Replacing structural missing values with `"None"`
- Most-frequent imputation for remaining categorical missing values
- One-hot encoding
- Ignoring unseen categories during inference

Using:

```python
handle_unknown="ignore"
```

is particularly important for production because the API may receive a category that was not present in the training dataset.

---

# 🤖 Model Benchmarking

Several regression approaches were evaluated using a **20% holdout test set**.

The primary evaluation metric is:

### Mean Absolute Error — MAE

MAE is calculated on the **original USD price scale**.

This makes the results easier to interpret from a business perspective.

| Model | Test MAE | Status | Comments |
|---|---:|---|---|
| **Linear Regression** | **$12,161.84** | 🏆 Selected Baseline | Strong performance after preprocessing and target transformation |
| Random Forest Regressor | $14,466.52 | Challenger | Less effective with the high-dimensional sparse feature space |

---

## 🏆 Selected Model

The **Linear Regression** model was selected as the baseline production model.

Despite being relatively simple, it achieved better performance than the Random Forest implementation used during benchmarking.

This result highlights an important ML engineering principle:

> Model complexity does not automatically translate into better predictive performance.

The strong performance of Linear Regression can be attributed to:

- Effective preprocessing
- Logarithmic target transformation
- Feature engineering
- Standardization of numerical features
- One-hot encoding of categorical variables
- Compatibility with the resulting high-dimensional feature space

---

# 🔄 Reproducible Training Pipeline

The complete training process is automated through:

```bash
python3 src/train.py
```

The training script performs the following steps:

```text
1. Load dataset
       ↓
2. Clean data
       ↓
3. Remove identified outliers
       ↓
4. Separate features and target
       ↓
5. Train/test split
       ↓
6. Build preprocessing pipeline
       ↓
7. Transform target
       ↓
8. Train regression model
       ↓
9. Evaluate on holdout set
       ↓
10. Convert predictions to original price scale
       ↓
11. Calculate MAE
       ↓
12. Serialize complete pipeline
```

The final artifact is exported to:

```text
models/model_pipeline.joblib
```

Because preprocessing and the estimator are stored together, the same pipeline can be loaded directly by the production API.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd real-estate-mlops
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run the production training pipeline:

```bash
python3 src/train.py
```

After successful training, the serialized model pipeline should be available at:

```text
models/model_pipeline.joblib
```

---

# ⚡ Run the API Locally

The FastAPI application can be started using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will then be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation is available at:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker

The application is containerized using Docker.

## Build the Image

```bash
docker build -t real-estate-api:v2 .
```

## Run the Container

```bash
docker run -p 8000:8000 real-estate-api:v2
```

The API can then be accessed through:

```text
http://localhost:8000
```

---

# ☸️ Kubernetes Deployment with Kind

The project uses **Kind (Kubernetes in Docker)** to run a local Kubernetes cluster.

Create the cluster if necessary:

```bash
kind create cluster --name mlops-cluster
```

---

## Build the Docker Image

```bash
docker build -t real-estate-api:v2 .
```

---

## Load the Image into Kind

```bash
kind load docker-image real-estate-api:v2 --name mlops-cluster
```

This allows the Kubernetes cluster to use the locally built image without pushing it to a container registry.

---

## Deploy the API

Apply the Kubernetes manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## Deploy Prometheus

Deploy the monitoring configuration:

```bash
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
```

---

# 🔄 Kubernetes Architecture

The production-like local environment consists of:

```text
                    ┌──────────────────────┐
                    │       Client         │
                    └──────────┬───────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Kubernetes Service       │
                 │ real-estate-api-service  │
                 └────────────┬─────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │ API Pod         │       │ API Pod         │
        │ Replica 1       │       │ Replica 2       │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Prometheus    │
                    │    Monitoring    │
                    └──────────────────┘
```

The API deployment runs **2 replicas**, providing basic redundancy and allowing rolling updates without taking the service offline.

---

# 🔁 Zero-Downtime Rolling Updates

The Kubernetes deployment supports rolling updates.

For example, when deploying a new image version:

```text
v1
│
├── Pod 1
└── Pod 2

        ↓ Rolling Update

v1 + v2
│
├── Pod 1 → v2
├── Pod 2 → v1

        ↓

v2
│
├── Pod 1
└── Pod 2
```

Kubernetes gradually replaces old pods with new ones while maintaining service availability.

Health probes help Kubernetes determine whether new pods are ready to receive traffic.

---

# 📡 API Endpoints

The main inference endpoint is:

```http
POST /predict
```

The API receives property characteristics and returns a predicted residential property price.

Prometheus metrics are exposed through:

```http
GET /metrics
```

FastAPI documentation is available through:

```http
GET /docs
```

---

# 📈 Monitoring with Prometheus

The API is instrumented using:

```text
prometheus-fastapi-instrumentator
```

This exposes application metrics that Prometheus can scrape.

The monitoring architecture is:

```text
FastAPI
   │
   │ /metrics
   ▼
Prometheus
   │
   ▼
Time-Series Metrics
```

This provides visibility into API traffic and application behavior.

---

# 🔌 Accessing the Services

## API

Forward the Kubernetes service locally:

```bash
kubectl port-forward svc/real-estate-api-service 8080:80
```

The API is then available at:

```text
http://localhost:8080
```

### Prediction endpoint

```text
POST http://localhost:8080/predict
```

### Metrics

```text
GET http://localhost:8080/metrics
```

### Interactive API documentation

```text
http://localhost:8080/docs
```

---

## Prometheus

Forward the Prometheus service:

```bash
kubectl port-forward svc/prometheus-service 9090:9090
```

Prometheus can then be accessed at:

```text
http://localhost:9090
```

---

# 🧪 Example Workflow

A complete local deployment can be performed with:

```bash
# Train the model
python3 src/train.py

# Build Docker image
docker build -t real-estate-api:v2 .

# Load image into Kind
kind load docker-image real-estate-api:v2 --name mlops-cluster

# Deploy API
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy Prometheus
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml

# Check deployments
kubectl get deployments

# Check pods
kubectl get pods

# Check services
kubectl get services
```

Then expose the API:

```bash
kubectl port-forward svc/real-estate-api-service 8080:80
```

And access:

```text
http://localhost:8080/docs
```

---

# 🔍 Useful Kubernetes Commands

Check running pods:

```bash
kubectl get pods
```

Check deployments:

```bash
kubectl get deployments
```

Check services:

```bash
kubectl get services
```

Inspect API logs:

```bash
kubectl logs deployment/real-estate-api
```

Describe a pod:

```bash
kubectl describe pod <pod-name>
```

Check rollout status:

```bash
kubectl rollout status deployment/real-estate-api
```

View rollout history:

```bash
kubectl rollout history deployment/real-estate-api
```

---

# 📊 MLOps Capabilities

This project demonstrates several core MLOps principles:

| Capability | Implementation |
|---|---|
| **Reproducible preprocessing** | Scikit-Learn `ColumnTransformer` |
| **Feature engineering** | Production preprocessing pipeline |
| **Model training** | Automated Python training script |
| **Model serialization** | Joblib |
| **Model serving** | FastAPI |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **Local Kubernetes** | Kind |
| **High availability** | 2 API replicas |
| **Health checks** | Kubernetes probes |
| **Rolling deployments** | Kubernetes Deployment |
| **Observability** | Prometheus |
| **API metrics** | Prometheus FastAPI Instrumentator |

---

# 🗺️ Project Roadmap

## ✅ Completed

- [x] Exploratory Data Analysis
- [x] Outlier detection and treatment
- [x] Automated preprocessing pipeline
- [x] Feature engineering
- [x] Target transformation
- [x] Model benchmarking
- [x] Linear Regression baseline
- [x] Random Forest challenger model
- [x] Model serialization
- [x] Production-ready FastAPI inference API
- [x] Docker containerization
- [x] Kubernetes deployment
- [x] Kind local Kubernetes cluster
- [x] Multiple API replicas
- [x] Kubernetes health probes
- [x] Zero-downtime rolling updates
- [x] Prometheus instrumentation
- [x] Prometheus in-cluster scraping

---

# 🚧 Next Steps

The following improvements could further extend the project toward a more complete production MLOps platform:

### 📈 Horizontal Pod Autoscaling

Implement Kubernetes HPA based on CPU, memory, or custom application metrics.

```text
Traffic ↑
   │
   ▼
HPA
   │
   ├── Pod 1
   ├── Pod 2
   ├── Pod 3
   └── Pod 4
```

### 📊 Grafana

Add Grafana to build dashboards for:

- Request rate
- Latency
- Error rate
- Pod utilization
- Prediction API traffic
- Model-serving performance

### 🔄 CI/CD

Implement GitHub Actions for:

```text
Git Push
   ↓
Unit Tests
   ↓
Model/API Tests
   ↓
Docker Build
   ↓
Image Validation
   ↓
Container Registry
   ↓
Kubernetes Deployment
```

### 📉 Data & Concept Drift

Introduce automated monitoring of:

- Feature distributions
- Missing-value rates
- Categorical distribution changes
- Prediction distribution
- Model performance degradation
- Data drift
- Concept drift

---

# 🎯 Learning Objectives

This project is designed to demonstrate practical skills across the complete machine learning lifecycle:

### Data Engineering

- Data cleaning
- Missing-value treatment
- Outlier analysis
- Feature engineering

### Machine Learning

- Regression
- Target transformation
- Model comparison
- Evaluation using MAE
- Reproducible preprocessing

### MLOps

- Model serialization
- API-based inference
- Containerization
- Kubernetes orchestration
- Rolling deployments
- Health monitoring
- Application observability

---

# 👨‍💻 Author

Developed as an end-to-end **MLOps portfolio project** demonstrating how to take a machine learning model from exploratory experimentation to a **containerized, Kubernetes-deployed, monitored production-style inference service**.

The project combines:

```text
Data Engineering
       +
Machine Learning
       +
Software Engineering
       +
Docker
       +
Kubernetes
       +
Observability
       =
End-to-End MLOps
```

---

## ⭐ If you find this project useful

Feel free to **star the repository**, explore the implementation, and extend the platform with additional MLOps capabilities such as CI/CD, Grafana dashboards, model registries, automated retraining, and drift detection.