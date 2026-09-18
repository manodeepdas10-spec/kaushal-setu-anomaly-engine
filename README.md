# 🚀 Kaushal Setu — AI/ML Anomaly Detection Microservice (SIH PS 135)

> **Problem Statement 135 (SIH 2026):** Difficulties in tracking employment outcomes, skill gaps, and the impact of skilling initiatives (Government of Maharashtra).

This repository contains the core **AI-driven Anomaly Detection Module** for the *Kaushal Setu* platform. Built with **Scikit-learn (Isolation Forest)** and **FastAPI**, this engine identifies anomalous claims in post-skilling placement data—such as fraudulent salary spikes, suspicious retention numbers, or unnatural job-hopping patterns—before claims are stored or submitted for automated verification.

---

## 🛠️ Tech Stack & Architecture

* **Programming Language:** Python 3.10+
* **Machine Learning Framework:** Scikit-learn (Isolation Forest, ColumnTransformer)
* **API Microservice:** FastAPI & Uvicorn
* **Data Handling & Preprocessing:** Pandas, NumPy
* **Model Persistence:** Joblib

### 🔄 Data & Execution Pipeline

```text
[ Incoming Trainee Claim Payload ]
               │
               ▼
   ┌───────────────────────┐
   │  FastAPI Microservice │
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐
   │ Scikit-learn Pipeline │
   │  (Imputer + OneHot)   │
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐
   │    Isolation Forest   │  ──► [ Normal Claims ]  ──► Status: VERIFIED
   │    Anomaly Detector   │  ──► [ Outlier Claims ] ──► Status: FLAGGED_FOR_AUDIT
   └───────────────────────┘
