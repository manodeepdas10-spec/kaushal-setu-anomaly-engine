from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Kaushal Setu - AIML Anomaly Engine",
    description="Detects anomalous claims in employment status, wage progression, and retention."
)

# Load trained pipeline
try:
    model_pipeline = joblib.load('kaushal_setu_anomaly_model.pkl')
except Exception as e:
    model_pipeline = None


class TraineeOutcomePayload(BaseModel):
    trainee_id: str = Field(..., example="TRN_MH_2026_9481")
    salary: Optional[float] = Field(default=None, example=18500.0)
    retention_months: Optional[float] = Field(default=None, example=12.0)
    job_changes: Optional[int] = Field(default=None, example=1)
    district: Optional[str] = Field(default=None, example="Pune")
    course_category: Optional[str] = Field(default=None, example="IT-ITeS")
    qualification: Optional[str] = Field(default=None, example="Graduate")


@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": model_pipeline is not None}


@app.post("/predict-anomaly")
def predict_anomaly(payload: TraineeOutcomePayload):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="AI Model pipeline is uninitialized.")

    try:
        # Convert payload into DataFrame matching preprocessor feature schema
        input_data = pd.DataFrame([{
            'salary': payload.salary,
            'retention_months': payload.retention_months,
            'job_changes': payload.job_changes,
            'district': payload.district,
            'course_category': payload.course_category,
            'qualification': payload.qualification
        }])

        # Predict: -1 indicates Anomaly, 1 indicates Normal
        prediction = model_pipeline.predict(input_data)[0]

        # Calculate decision score: lower values in decision_function mean higher anomaly risk.
        # Inverting (-1 *) turns high anomaly risk into a positive score for intuitive outputs.
        raw_decision = model_pipeline.decision_function(input_data)[0]
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
        raise HTTPException(status_code=500, detail=f"Prediction execution error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)