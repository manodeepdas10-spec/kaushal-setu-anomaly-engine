from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Kaushal Setu - Unified AIML Fraud Detection Suite",
    description="Dual-engine AI architecture for SIH Problem Statement 135. Protects enrollment workflows and post-placement outcome logging.",
    version="2.0.0"
)

# ==========================================
# MODEL PIPELINE INGESTION INITIALIZATION
# ==========================================
try:
    outcome_model_pipeline = joblib.load('kaushal_setu_anomaly_model.pkl')
except Exception:
    outcome_model_pipeline = None

try:
    identity_model_pipeline = joblib.load('kaushal_setu_identity_model.pkl')
except Exception:
    identity_model_pipeline = None


# ==========================================
# PYDANTIC SCHEMAS / PAYLOADS
# ==========================================
class TraineeOutcomePayload(BaseModel):
    trainee_id: str = Field(..., example="TRN_MH_2026_9481")
    salary: Optional[float] = Field(default=None, example=18500.0)
    retention_months: Optional[float] = Field(default=None, example=12.0)
    job_changes: Optional[int] = Field(default=None, example=1)
    district: Optional[str] = Field(default=None, example="Pune")
    course_category: Optional[str] = Field(default=None, example="IT-ITeS")
    qualification: Optional[str] = Field(default=None, example="Graduate")

class TraineeIdentityPayload(BaseModel):
    trainee_id: str = Field(..., example="TRN_MH_2026_1102")
    module_completion_speed: float = Field(..., description="Average time taken per module sub-unit in minutes", example=18.4)
    device_concurrency_count: int = Field(..., description="Count of active profiles sharing this hardware ID signature", example=1)
    face_similarity_index: float = Field(..., description="Webcam similarity comparison confidence against Aadhaar (0.0 to 1.0)", example=0.91)
    liveness_status: int = Field(..., description="1 = Passed deep learning liveness check, 0 = Failed", example=1)
    ip_risk_score: int = Field(..., description="1 = Flagged hosting/datacenter/VPN IP block, 0 = Residential local ISP", example=0)
    training_center_id: str = Field(..., description="Unique government identifier of the operational center", example="TC_PUNE_03")


# ==========================================
# ROUTERS / ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    return {
        "status": "active",
        "outcome_engine_loaded": outcome_model_pipeline is not None,
        "identity_engine_loaded": identity_model_pipeline is not None
    }

@app.post("/predict-outcome-anomaly")
def predict_anomaly(payload: TraineeOutcomePayload):
    """ENGINE 1: Identifies fraudulent salary metrics and post-placement claims."""
    if not outcome_model_pipeline:
        raise HTTPException(status_code=500, detail="Outcome Anomaly AI Model pipeline is uninitialized.")

    try:
        input_data = pd.DataFrame([{
            'salary': payload.salary,
            'retention_months': payload.retention_months,
            'job_changes': payload.job_changes,
            'district': payload.district,
            'course_category': payload.course_category,
            'qualification': payload.qualification
        }])

        prediction = outcome_model_pipeline.predict(input_data)[0]
        raw_decision = outcome_model_pipeline.decision_function(input_data)[0]
        anomaly_score = float(-1 * raw_decision)
        is_anomaly = bool(prediction == -1)

        return {
            "trainee_id": payload.trainee_id,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "status_flag": "FLAGGED_FOR_AUDIT" if is_anomaly else "VERIFIED",
            "action_required": is_anomaly
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outcome Engine Exception: {str(e)}")


@app.post("/predict-fake-trainee")
def predict_fake_trainee(payload: TraineeIdentityPayload):
    """ENGINE 2: Real-time analysis of attendance logs and biometrics to detect fake trainees."""
    if not identity_model_pipeline:
        raise HTTPException(status_code=500, detail="Identity Verification AI Model pipeline is uninitialized.")

    try:
        # --- CRITICAL HACKATHON OVERRIDE SECURITY RULES ---
        # Rule 1: If liveness check fails, bypass the model and flag immediately.
        if payload.liveness_status == 0:
            return {
                "trainee_id": payload.trainee_id,
                "is_fake_trainee": True,
                "fraud_probability_score": 1.0000,
                "status_flag": "IDENTITY_SUSPENDED_GHOST_ALERT",
                "action_required": True,
                "reason_degradation": "CRITICAL_BIOMETRIC_LIVENESS_FAILURE"
            }

        # Rule 2: If the face similarity is drastically low, flag immediately.
        if payload.face_similarity_index < 0.40:
            return {
                "trainee_id": payload.trainee_id,
                "is_fake_trainee": True,
                "fraud_probability_score": 0.9800,
                "status_flag": "IDENTITY_SUSPENDED_GHOST_ALERT",
                "action_required": True,
                "reason_degradation": "SEVERE_FACIAL_MISMATCH_THRESHOLD"
            }
        # --------------------------------------------------

        # If it passes the strict security rules, let the machine learning model run its analysis
        input_data = pd.DataFrame([{
            'module_completion_speed': payload.module_completion_speed,
            'device_concurrency_count': payload.device_concurrency_count,
            'face_similarity_index': payload.face_similarity_index,
            'liveness_status': payload.liveness_status,
            'ip_risk_score': payload.ip_risk_score,
            'training_center_id': payload.training_center_id
        }])

        prediction = int(identity_model_pipeline.predict(input_data))
        probabilities = identity_model_pipeline.predict_proba(input_data)
        # [0][1] extracts the specific probability for Class 1 (Fake Trainee)
        fraud_confidence = float(probabilities[0][1])

        is_fake = bool(prediction == 1)

        return {
            "trainee_id": payload.trainee_id,
            "is_fake_trainee": is_fake,
            "fraud_probability_score": round(fraud_confidence, 4),
            "status_flag": "IDENTITY_SUSPENDED_GHOST_ALERT" if is_fake else "IDENTITY_CLEAN",
            "action_required": is_fake,
            "reason_degradation": "ML_CLASSIFIER_DETERMINATION"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity Engine Exception: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
