# 📊 End-to-End Telco Churn Pipeline

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://scikit-learn.org)
[![Dashboard](https://shields.io)](https://streamlit.io)
[![License](https://shields.io)](https://opensource.org)


An end-to-end, enterprise-grade machine learning application designed to predict and mitigate customer attrition in the telecommunications sector. Moving away from static notebooks, this project is built entirely on a highly decoupled **Modular Software Architecture**, implementing automated data components, strict data-leakage safeguards, and a live user dashboard console.

---

## 📌 Executive Summary & Key Results

Instead of relying on heavy black-box boosting models, this repository demonstrates that **thorough, domain-driven feature engineering allows a clean, highly interpretable linear model to dominate.** Following a rigorous evaluation across multiple algorithms, an optimized **Logistic Regression Pipeline** emerged as the winning production architecture.

### 📊 Model Performance Matrix
*   **ROC-AUC Score:** `84.25%` (Exceptional threshold separation capability)
*   **Raw Accuracy:** `80.48%`
*   **Weighted F1-Score:** `79.96%`
*   **Precision:** `65.52%` (Minimizes capital wasted on false alarms)
*   **Recall / Core Catch Rate:** `55.88%` (Successfully flags 56 out of 100 churning accounts before they request cancellation)

---

## 🛠️ Custom Domain Feature Engineering
The pipeline's strong performance is driven by 3 custom-synthesized features that capture underlying financial and ecosystem behavior:
1.  **Financial Burden Ratio (`MonthlyToTotalRatio`):** Measures the acceleration of financial stress by checking monthly velocity against total lifetime spend.
2.  **Ecosystem Stickiness Counter (`EcosystemFeaturesCount`):** Aggregates total active secondary utility subscriptions (TechSupport, OnlineSecurity, etc.). Higher counts exponentially decrease attrition.
3.  **High-Risk Persona Flag (`Is_High_Risk_Persona`):** A custom binary structural indicator mapping users stuck on Month-to-Month contracts who cross a **\$70 pricing wall** while lacking active tech support.

---

## 📂 Project Directory Structure

The system is split into independent micro-modules following industry standard code-base design patterns:

```text
End-to-End Telco Churn Pipeline/
│
├──  artifacts/                   # Persisted data artifacts and trained model binaries
│   ├── Churn.csv                # Automated raw backup source data file
│   ├── train.csv                # Stratified training split data file (80%)
│   ├── test.csv                 # Stratified testing split data file (20%)
│   ├── model.pkl                # Serialized trained Logistic Regression model 
│   └── preprocessor.pkl         # Serialized ColumnTransformer preprocessing rules
│
├── src/
│   ├── __init__.py
│   ├── exception.py             # Custom system-wide exception handling tracker
│   ├── logger.py                # Live runtime stream log execution capture script
│   ├── utils.py                 # Common serialization helper utilities (Joblib engine)
│   │
│   ├── components/              # Isolated execution data workers
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Stratified data splits & folder creation
│   │   ├── data_transformation.py # Custom transformers & multi-lane ColumnTransformers
│   │   └── Model_trainer.py     # Final classifier validation, scoring & saving
│   │
│   └── pipeline/                # Production orchestration workflows
│       ├── __init__.py
│       ├── train_pipeline.py    # Training lifecycle automation trigger
│       └── predict_pipeline.py  # Lean live customer payload inference mapping
│
├── app.py                       # Live customer retention web dashboard (Streamlit)
├── requirements.txt             # Strict version-controlled production dependencies
└── .gitignore                   # Excludes environments, logs, and tracking binaries
```

---

## ⚙️ Local Environment Installation Guide

Replicate this exact environment workspace on your local computer by executing this quick command sequence:

### 1. Initialize and Activate the Virtual Environment
```bash
# Create an isolated local virtual environment folder using Python 3.12.7
conda create -p ./venv python=3.12.7 -y

# Activate the local prefix environment space
conda activate ./venv
```

### 2. Install Version-Controlled Dependencies
```bash
# Install all required data manipulation, math, pipeline and web frameworks
pip install -r requirements.txt
```

---

## 🚀 Execution & Operational Workflows

### Running the End-to-End Training Lifecycle
To ingest raw source metrics from scratch, split datasets with stratification, execute transformations, evaluate scores, and lock down your serialized binary files, run the master training workflow module:
```bash
python -m src.pipeline.train_pipeline
```

### Simulating Single-Row Inference (Terminal Test)
To verify your predictive infrastructure can parse completely raw customer strings natively in memory without any system bottlenecks, trigger the pipeline test bench:
```bash
python -m src.pipeline.predict_pipeline
```

### Launching the Live Streamlit Dashboard Console
To spin up a fully interactive enterprise workspace with dropdown selectors and slider arrays for client risk assessments, launch the web framework:
```bash
streamlit run app.py
```
*Your browser will automatically open the application at `http://localhost:8501`*

---

## 📡 Production REST API Architecture Strategy
The decoupled nature of this project means it is fully optimized for containerized deployment (Docker, AWS ECS, or Render). The **`PredictPipeline`** class splits object allocation into a two-tiered system:
*   **The Constructor (`__init__`)** loads model and transformer weights into memory **once** when the server boots.
*   **The `predict_live_input` method** runs entirely in RAM, bypassing disk I/O operations to effortlessly manage high-concurrency API calls with millisecond latency.

---

## 🌐 Live Web Application & Cloud Deployment

The prediction architecture has been successfully compiled and deployed to the cloud as a live, interactive web application. This interface bridges the gap between machine learning backend outputs and non-technical business stakeholders, enabling customer success managers to audit client risk variables instantly in their web browsers.

*   📡 **Live Dashboard URL:** [TelcoPulse: Enterprise Retention Console](https://appuction-ready-telco-churn-pipeline-m5gs2jauvvkkyexk6fjpa8.streamlit.app/)
*   ☁️ **Hosting Infrastructure:** Streamlit Community Cloud (Containerized Microservice)
*   🔋 **Operational State:** Live & Stable (Bypasses disk I/O to perform low-latency memory-cached inferences)

### 🖥️ Interactive Web Console Interface Preview
The dashboard accepts user demographic markers, contract infrastructures, and continuous financial fields via interactive sliders and dropdown selectors. Upon triggering calculation workflows, it securely instantiates the backend `PredictPipeline`, outputs a granular decimal-level risk risk percentage and returns contextual corporate intervention playbooks dynamically tailored to that customer's specific risk bracket.

