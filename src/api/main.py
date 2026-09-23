"""
BAN6800 Capstone — Stanbic IBTC Bank Credit Risk Intelligence Platform
FastAPI /predict Endpoint — Module 4 Deliverable
File: src/api/main.py


Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    docker build -t crip-api . && docker run -p 8000:8000 crip-api

Endpoints:
    GET  /health          — health check
    GET  /model/info      — model version and metadata
    POST /predict         — score a loan application
    POST /predict/explain — score + SHAP explanations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap
import logging
from datetime import datetime
from pathlib import Path

# ── App setup ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="CRIP Credit Risk Intelligence API",
    description=(
        "Stanbic IBTC Bank — Credit Risk Intelligence Platform\n"
        "BAN6800 Module 4 Deliverable\n"
        "Champion model: LightGBM v1.0 (AUC-ROC: 0.796)"
    ),
    version="1.0.0",
    contact={"name": "Layori Soetan", "url": "github.com/Layorr/stanbic-credit-risk"},
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Load model on startup ─────────────────────────────────────────────────────
MODEL_PATH = Path("models/lgbm_champion_v1.pkl")
model, explainer = None, None

@app.on_event("startup")
def load_model():
    global model, explainer
    try:
        model    = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        logger.info(f"Model loaded: {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(
            f"Model file not found: {MODEL_PATH}. "
            "Run train_model.py first to generate the model artifact."
        )


# ── Request / Response schemas ────────────────────────────────────────────────
class LoanApplication(BaseModel):
    """
    Key features from the Module 3 feature set (85 total).
    Provide all available features for best accuracy.
    """
    # Core financial features
    AMT_CREDIT:              float = Field(..., description="Total loan credit amount (NGN)")
    AMT_INCOME_TOTAL:        float = Field(..., description="Annual household income (NGN)")
    AMT_ANNUITY:             float = Field(..., description="Loan annuity payment (NGN/period)")
    # Derived features (computed in Module 3 pipeline)
    CREDIT_INCOME_RATIO:     float = Field(..., description="AMT_CREDIT / AMT_INCOME_TOTAL")
    ANNUITY_INCOME_RATIO:    float = Field(..., description="AMT_ANNUITY / AMT_INCOME_TOTAL")
    CREDIT_TERM:             float = Field(..., description="AMT_CREDIT / AMT_ANNUITY")
    # Demographic / employment
    AGE_YEARS:               float = Field(..., description="Applicant age in years")
    DAYS_EMPLOYED_YEARS:     float = Field(..., description="Years of current employment")
    # External credit scores
    EXT_SOURCE_2:            float = Field(..., description="External credit score 2 (0–1)")
    EXT_SOURCE_3:            Optional[float] = Field(None, description="External credit score 3 (0–1)")
    EXT_SOURCE_1:            Optional[float] = Field(None, description="External credit score 1 (0–1)")
    # Additional features (fill with defaults if unknown)
    NAME_EDUCATION_TYPE:     Optional[int] = Field(2, description="Ordinal-encoded education level")
    CODE_GENDER:             Optional[int] = Field(0, description="0=Female, 1=Male")

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_CREDIT":           450000,
                "AMT_INCOME_TOTAL":     90000,
                "AMT_ANNUITY":          22500,
                "CREDIT_INCOME_RATIO":  5.0,
                "ANNUITY_INCOME_RATIO": 0.25,
                "CREDIT_TERM":          20.0,
                "AGE_YEARS":            38.5,
                "DAYS_EMPLOYED_YEARS":  4.2,
                "EXT_SOURCE_2":         0.62,
                "EXT_SOURCE_3":         0.55,
            }
        }


class PredictionResponse(BaseModel):
    request_id:          str
    timestamp:           str
    default_probability: float = Field(..., description="Probability of loan default (0–1)")
    risk_tier:           str   = Field(..., description="Low / Medium / High risk tier")
    prediction_label:    int   = Field(..., description="1 = Default risk; 0 = Repay")
    requires_review:     bool  = Field(..., description="True if in human-in-the-loop range (0.40–0.60)")
    model_version:       str


class ExplainedPredictionResponse(PredictionResponse):
    top_risk_drivers: list = Field(..., description="Top 5 SHAP feature contributions")
    counterfactual_hint: str = Field(..., description="Key change for a different outcome")


def get_risk_tier(prob: float) -> str:
    if prob < 0.25:  return "Low"
    elif prob < 0.50: return "Medium"
    else:            return "High"


def build_feature_df(app: LoanApplication) -> pd.DataFrame:
    """Convert API request to a feature DataFrame matching training schema."""
    data = app.dict()
    # Fill missing optional features with imputed means from Module 3 pipeline
    if data["EXT_SOURCE_3"] is None: data["EXT_SOURCE_3"] = 0.51   # dataset mean
    if data["EXT_SOURCE_1"] is None: data["EXT_SOURCE_1"] = 0.50   # imputed mean
    return pd.DataFrame([data])


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint — used by Docker HEALTHCHECK."""
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "model_version": "CRIP-LightGBM-v1.0",
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Return model metadata from the MLflow Model Registry."""
    return {
        "model_name":        "CRIP-LightGBM-v1.0",
        "algorithm":         "LightGBM (Ke et al., 2017)",
        "training_dataset":  "Home Credit Default Risk — Kaggle (2018)",
        "train_records":     246009,
        "test_auc_roc":      0.796,
        "test_f1":           0.53,
        "test_recall":       0.58,
        "fairness_gender_delta": 0.005,
        "fairness_age_delta":    0.054,
        "protected_attributes":  ["CODE_GENDER", "AGE_YEARS"],
        "mlflow_experiment": "crip_credit_risk_v1",
        "github_repo":       "github.com/Layorr/stanbic-credit-risk",
        "intended_use":      "Credit risk decision support — Stanbic IBTC Bank retail lending",
        "not_for_use":       "Employee screening, insurance pricing, AML detection",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    """
    Score a loan application and return default probability + risk tier.
    Borderline cases (0.40–0.60) are flagged for mandatory human review
    per the Module 1 Ethical AI Charter.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    df   = build_feature_df(application)
    prob = float(model.predict_proba(df)[0][1])
    tier = get_risk_tier(prob)

    # Log access for NDPR (2019) compliance audit trail
    logger.info(f"AUDIT | /predict | prob={prob:.4f} | tier={tier} | {datetime.utcnow().isoformat()}")

    return PredictionResponse(
        request_id          = f"CRIP-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:18]}",
        timestamp           = datetime.utcnow().isoformat(),
        default_probability = round(prob, 4),
        risk_tier           = tier,
        prediction_label    = int(prob >= 0.50),
        requires_review     = 0.40 <= prob <= 0.60,
        model_version       = "CRIP-LightGBM-v1.0",
    )


@app.post("/predict/explain", response_model=ExplainedPredictionResponse, tags=["Prediction"])
def predict_with_explanation(application: LoanApplication):
    """
    Score a loan application AND return SHAP-based explanations (Lundberg & Lee, 2017).
    Returns top 5 feature contributions and a counterfactual hint.
    For regulated use: ensures every decision is interpretable per NDPR (2019).
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df          = build_feature_df(application)
    prob        = float(model.predict_proba(df)[0][1])
    shap_vals   = explainer.shap_values(df)[0]
    feature_names = df.columns.tolist()

    # Top 5 SHAP drivers
    ranked = sorted(
        zip(feature_names, shap_vals, df.values[0]),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    drivers = [
        {"feature": f, "shap_value": round(s, 4), "feature_value": round(float(v), 4),
         "direction": "raises risk" if s > 0 else "reduces risk"}
        for f, s, v in ranked
    ]

    # Simple counterfactual hint (Module 2 Explainability Framework)
    top_risk_drivers = [d for d in drivers if d["direction"] == "raises risk"]
    hint = "No specific action needed — application is low risk."
    if top_risk_drivers and prob >= 0.40:
        top = top_risk_drivers[0]
        hint = (
            f"Primary risk driver: {top['feature']} (SHAP = {top['shap_value']:+.4f}). "
            f"Reducing this value may improve the outcome."
        )

    logger.info(f"AUDIT | /predict/explain | prob={prob:.4f} | {datetime.utcnow().isoformat()}")

    return ExplainedPredictionResponse(
        request_id          = f"CRIP-EXP-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:18]}",
        timestamp           = datetime.utcnow().isoformat(),
        default_probability = round(prob, 4),
        risk_tier           = get_risk_tier(prob),
        prediction_label    = int(prob >= 0.50),
        requires_review     = 0.40 <= prob <= 0.60,
        model_version       = "CRIP-LightGBM-v1.0",
        top_risk_drivers    = drivers,
        counterfactual_hint = hint,
    )
