"""
pydantic_models.py
------------------
Request and response schemas for the Bati Bank Credit Risk API.
All fields validated by Pydantic v2.
"""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer credit risk prediction."""

    customer_id: str = Field(..., description="Unique customer identifier")

    # Aggregate features
    total_transaction_amount: float = Field(
        ..., description="Sum of all transaction amounts for the customer"
    )
    avg_transaction_amount: float = Field(
        ..., description="Average transaction amount"
    )
    std_transaction_amount: float = Field(
        default=0.0, description="Standard deviation of transaction amounts"
    )
    transaction_count: int = Field(
        ..., ge=1, description="Total number of transactions"
    )
    total_value: float = Field(..., description="Sum of absolute transaction values")
    avg_value: float = Field(..., description="Average absolute transaction value")
    unique_products: int = Field(
        default=1, ge=1, description="Number of distinct products purchased"
    )
    unique_categories: int = Field(
        default=1, ge=1, description="Number of distinct product categories"
    )
    unique_providers: int = Field(
        default=1, ge=1, description="Number of distinct providers"
    )
    fraud_count: int = Field(
        default=0, ge=0, description="Number of flagged fraudulent transactions"
    )
    fraud_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Proportion of fraudulent transactions"
    )

    # Encoded categoricals (integer-encoded)
    most_common_channel: int = Field(default=0, description="Encoded most-used channel")
    most_common_category: int = Field(
        default=0, description="Encoded most-bought product category"
    )
    most_common_pricing: int = Field(
        default=0, description="Encoded most-common pricing strategy"
    )

    # Datetime features
    transaction_hour: int = Field(
        default=12, ge=0, le=23, description="Hour of most recent transaction"
    )
    transaction_day: int = Field(
        default=15, ge=1, le=31, description="Day of month of most recent transaction"
    )
    transaction_month: int = Field(
        default=6, ge=1, le=12, description="Month of most recent transaction"
    )
    transaction_year: int = Field(
        default=2019, description="Year of most recent transaction"
    )

    # RFM features
    recency: float = Field(
        ..., ge=0, description="Days since the customer's last transaction"
    )
    frequency: int = Field(..., ge=1, description="Number of transactions (= transaction_count)")
    monetary: float = Field(..., description="Total monetary value of transactions")

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "C001",
            "total_transaction_amount": 15000.0,
            "avg_transaction_amount": 3000.0,
            "std_transaction_amount": 500.0,
            "transaction_count": 5,
            "total_value": 15000.0,
            "avg_value": 3000.0,
            "unique_products": 3,
            "unique_categories": 2,
            "unique_providers": 2,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "most_common_channel": 1,
            "most_common_category": 2,
            "most_common_pricing": 0,
            "transaction_hour": 14,
            "transaction_day": 10,
            "transaction_month": 3,
            "transaction_year": 2019,
            "recency": 7,
            "frequency": 5,
            "monetary": 15000.0,
        }
    }}


class PredictionResponse(BaseModel):
    """Credit risk prediction response."""

    customer_id: str = Field(..., description="Customer identifier echoed from input")
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of default (0 = no risk, 1 = certain default)",
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score derived from risk probability (300–850)",
    )
    risk_label: str = Field(
        ...,
        description="Human-readable risk category: low_risk | medium_risk | high_risk",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "C001",
            "risk_probability": 0.12,
            "credit_score": 765,
            "risk_label": "low_risk",
        }
    }}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model: str = "credit_risk_best_model"
    version: str = "1.0.0"
