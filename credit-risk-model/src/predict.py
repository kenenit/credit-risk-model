"""
predict.py
----------
Inference utilities for the Bati Bank Credit Risk Model.

Loads the best model from local MLflow runs and provides:
  - risk_probability: P(is_high_risk=1)
  - credit_score: Scaled score (300-850)
  - risk_label: low_risk | medium_risk | high_risk
"""

import logging
import os
import mlflow
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "credit_risk_best_model"
SCORE_MIN = 300
SCORE_MAX = 850

_model = None


def get_model():
    """Load the registered MLflow model (cached after first call)."""
    global _model
    if _model is None:
        logger.info("Loading model from MLflow registry...")
        try:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/1"
            _model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model


def probability_to_credit_score(risk_probability: float) -> int:
    """Convert risk probability (0-1) to credit score (300-850)."""
    score = SCORE_MAX - (risk_probability * (SCORE_MAX - SCORE_MIN))
    return int(round(score))


def probability_to_label(risk_probability: float) -> str:
    """Map risk probability to human-readable label."""
    if risk_probability < 0.30:
        return "low_risk"
    elif risk_probability < 0.60:
        return "medium_risk"
    else:
        return "high_risk"


def predict(features: dict) -> dict:
    """
    Run inference on a single customer's features.

    Args:
        features: Dict of feature name -> value.

    Returns:
        Dict with risk_probability, credit_score, risk_label.
    """
    model = get_model()
    input_df = pd.DataFrame([features])

    y_prob = model.predict_proba(input_df)[:, 1]
    risk_prob = float(np.clip(y_prob[0], 0.0, 1.0))

    credit_score = probability_to_credit_score(risk_prob)
    risk_label = probability_to_label(risk_prob)

    logger.info(
        f"Prediction - P(default): {risk_prob:.4f} | "
        f"Score: {credit_score} | Label: {risk_label}"
    )

    return {
        "risk_probability": risk_prob,
        "credit_score": credit_score,
        "risk_label": risk_label,
    }
