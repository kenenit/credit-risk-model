"""
predict.py
----------
Inference utilities for the Bati Bank Credit Risk Model.

Loads the registered MLflow model and provides:
  - risk_probability: P(is_high_risk=1) for a given customer
  - credit_score: Scaled score (300–850) derived from risk probability
  - risk_label: "low_risk" | "medium_risk" | "high_risk"
"""

import logging

import mlflow.pyfunc
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGISTERED_MODEL_NAME = "credit_risk_best_model"
MODEL_STAGE = "Production"  # or "Staging" / "latest"
SCORE_MIN = 300
SCORE_MAX = 850


# ---------------------------------------------------------------------------
# Model loader (singleton pattern)
# ---------------------------------------------------------------------------
_model = None


def get_model():
    """Load the registered MLflow model (cached after first call)."""
    global _model
    if _model is None:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        try:
            _model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Fallback: load latest version
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
            logger.warning(f"Falling back to: {model_uri}")
            _model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Credit score conversion
# ---------------------------------------------------------------------------
def probability_to_credit_score(risk_probability: float) -> int:
    """
    Convert a risk probability (0–1) to a credit score (300–850).

    Higher probability of default → lower credit score.
    Uses a simple linear inversion.

    Args:
        risk_probability: P(default) in [0, 1].

    Returns:
        Integer credit score in [300, 850].
    """
    score = SCORE_MAX - (risk_probability * (SCORE_MAX - SCORE_MIN))
    return int(round(score))


def probability_to_label(risk_probability: float) -> str:
    """
    Map risk probability to a human-readable label.

    Thresholds:
        < 0.30  → low_risk
        0.30–0.60 → medium_risk
        > 0.60  → high_risk
    """
    if risk_probability < 0.30:
        return "low_risk"
    elif risk_probability < 0.60:
        return "medium_risk"
    else:
        return "high_risk"


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
def predict(features: dict) -> dict:
    """
    Run inference on a single customer's features.

    Args:
        features: Dict of feature name → value (matching model's training columns).

    Returns:
        Dict with keys: risk_probability, credit_score, risk_label.
    """
    model = get_model()

    input_df = pd.DataFrame([features])
    logger.info(f"Running prediction for input with shape {input_df.shape}")

    # MLflow pyfunc returns a DataFrame or numpy array
    result = model.predict(input_df)

    # Extract probability (assumes model exposes predict_proba-compatible output)
    if hasattr(result, "values"):
        risk_prob = float(result.values[0])
    else:
        risk_prob = float(result[0])

    # Clamp to [0, 1]
    risk_prob = float(np.clip(risk_prob, 0.0, 1.0))

    credit_score = probability_to_credit_score(risk_prob)
    risk_label = probability_to_label(risk_prob)

    logger.info(
        f"Prediction — P(default): {risk_prob:.4f} | "
        f"Score: {credit_score} | Label: {risk_label}"
    )

    return {
        "risk_probability": risk_prob,
        "credit_score": credit_score,
        "risk_label": risk_label,
    }


# ---------------------------------------------------------------------------
# Entry point (quick smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_input = {
        "total_transaction_amount": 12000.0,
        "avg_transaction_amount": 2400.0,
        "std_transaction_amount": 800.0,
        "transaction_count": 5,
        "total_value": 12000.0,
        "avg_value": 2400.0,
        "unique_products": 3,
        "unique_categories": 2,
        "unique_providers": 2,
        "fraud_count": 0,
        "fraud_rate": 0.0,
        "most_common_channel": 1,
        "most_common_category": 2,
        "most_common_pricing": 0,
        "transaction_hour": 14,
        "transaction_day": 15,
        "transaction_month": 3,
        "transaction_year": 2019,
        "recency": 10,
        "frequency": 5,
        "monetary": 12000.0,
    }
    result = predict(sample_input)
    print(result)
