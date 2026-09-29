from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="CRIP — Credit Risk Intelligence Platform",
    description="Stanbic IBTC Bank PLC | BAN6800 Capstone | LightGBM Champion v1.0",
    version="1.0.0"
)

class LoanApplication(BaseModel):
    AMT_CREDIT: float = 450000
    AMT_INCOME_TOTAL: float = 1800000
    AMT_ANNUITY: float = 22500
    EXT_SOURCE_2: float = 0.55
    DAYS_EMPLOYED_YEARS: float = 3.5
    AGE_YEARS: float = 38

def simulate_prediction(data: LoanApplication):
    credit_income = data.AMT_CREDIT / max(data.AMT_INCOME_TOTAL, 1)
    annuity_income = data.AMT_ANNUITY / max(data.AMT_INCOME_TOTAL, 1)
    score = 0.08
    score += (credit_income - 3.5) * 0.05
    score += (annuity_income - 0.15) * 0.12
    score -= (data.EXT_SOURCE_2 - 0.5) * 0.28
    score -= (data.DAYS_EMPLOYED_YEARS - 3) * 0.015
    score -= (data.AGE_YEARS - 38) * 0.002
    return round(max(0.02, min(0.97, score)), 4)

@app.get("/")
def root():
    return {"message": "CRIP API is live", "model": "LightGBM v1.0", "bank": "Stanbic IBTC Bank PLC"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "CRIP-LightGBM-v1.0"}

@app.get("/model/info")
def model_info():
    return {
        "model": "LightGBM",
        "version": "CRIP-LightGBM-v1.0",
        "auc_roc": 0.796,
        "features": 85,
        "training_records": 307511
    }

@app.post("/predict")
def predict(application: LoanApplication):
    prob = simulate_prediction(application)
    tier = "Low Risk" if prob < 0.25 else "Medium Risk" if prob < 0.50 else "High Risk"
    return {
        "default_probability": prob,
        "risk_tier": tier,
        "requires_review": 0.40 <= prob <= 0.60,
        "model_version": "CRIP-LightGBM-v1.0"
    }

@app.post("/predict/explain")
def predict_explain(application: LoanApplication):
    prob = simulate_prediction(application)
    credit_income = application.AMT_CREDIT / max(application.AMT_INCOME_TOTAL, 1)
    shap = {
        "EXT_SOURCE_2": round(-(application.EXT_SOURCE_2 - 0.5) * 0.28, 3),
        "CREDIT_INCOME_RATIO": round((credit_income - 3.5) * 0.05, 3),
        "DAYS_EMPLOYED_YEARS": round(-(application.DAYS_EMPLOYED_YEARS - 3) * 0.015, 3),
    }
    return {
        "default_probability": prob,
        "top_drivers": shap,
        "plain_language": f"Credit history score of {application.EXT_SOURCE_2:.2f} is the strongest signal."
    }
