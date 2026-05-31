"""
main.py
-------
FastAPI application for the Bati Bank Credit Risk Scoring API.

Endpoints:
  GET  /          — Root / welcome
  GET  /health    — Health check
  POST /predict   — Credit risk prediction for a single customer
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import CustomerFeatures, HealthResponse, PredictionResponse
from src.predict import predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description=(
        "Credit risk scoring API for Bati Bank's buy-now-pay-later service. "
        "Returns risk probability, credit score, and risk label for new customers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Bati Bank Credit Risk API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check that the service and model are operational."""
    return HealthResponse(status="ok", model="credit_risk_best_model", version="1.0.0")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_credit_risk(payload: CustomerFeatures):
    """
    Score a new customer's credit risk.

    Accepts customer transaction features and returns:
    - **risk_probability**: P(default) ∈ [0, 1]
    - **credit_score**: integer in [300, 850]
    - **risk_label**: low_risk | medium_risk | high_risk
    """
    logger.info(f"Received prediction request for customer: {payload.customer_id}")

    try:
        features = payload.model_dump(exclude={"customer_id"})
        result = predict(features)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return PredictionResponse(
        customer_id=payload.customer_id,
        risk_probability=result["risk_probability"],
        credit_score=result["credit_score"],
        risk_label=result["risk_label"],
    )
