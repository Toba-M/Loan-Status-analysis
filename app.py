import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load bundle ──────────────────────────────────────────────
bundle = joblib.load(r"C:\Users\PC\Downloads\loan_model.pkl")
model  = bundle["model"]
scaler = bundle["scaler"]

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts whether a loan will be approved or rejected.",
    version="1.0.0"
)

# ── Schema ───────────────────────────────────────────────────
class LoanRequest(BaseModel):
    Credit_History:    float = Field(..., description="1 = good history, 0 = bad")
    ApplicantIncome:   float = Field(..., description="Monthly income of applicant")
    LoanAmount:        float = Field(..., description="Loan amount in thousands")
    CoapplicantIncome: float = Field(..., description="Monthly income of co-applicant")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Credit_History":    1.0,
                "ApplicantIncome":   5000.0,
                "LoanAmount":        120.0,
                "CoapplicantIncome": 2000.0
            }]
        }
    }

class LoanResponse(BaseModel):
    prediction: str
    confidence: float

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Loan Approval Prediction API is running. Go to /docs to test."}


@app.post("/predict", response_model=LoanResponse)
def predict_loan(data: LoanRequest):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        input_scaled = scaler.transform(input_df)

        pred_encoded = model.predict(input_scaled)[0]

        proba      = model.predict_proba(input_scaled)[0]
        confidence = proba[pred_encoded]

        return LoanResponse(
            prediction="Approved" if pred_encoded == 1 else "Rejected",
            confidence=round(float(confidence), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}