"""
test_data_processing.py
-----------------------
Unit tests for feature engineering functions in src/data_processing.py.

Run with:
    pytest tests/test_data_processing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    assign_risk_label,
    calculate_rfm,
    create_aggregate_features,
    encode_categoricals,
    extract_datetime_features,
    impute_missing,
    scale_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_transactions():
    """Minimal synthetic transaction DataFrame mimicking Xente structure."""
    return pd.DataFrame(
        {
            "TransactionId": [f"T{i}" for i in range(10)],
            "CustomerId": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3", "C3", "C4"],
            "AccountId": [f"A{i}" for i in range(10)],
            "TransactionStartTime": pd.date_range(
                start="2019-01-01", periods=10, freq="7D"
            ),
            "Amount": [100, 200, 150, 50, 75, 300, 250, 200, 400, 10],
            "Value": [100, 200, 150, 50, 75, 300, 250, 200, 400, 10],
            "ProductId": [f"P{i % 3}" for i in range(10)],
            "ProductCategory": ["airtime", "data", "airtime", "data", "airtime",
                                 "financial_services", "airtime", "data",
                                 "financial_services", "airtime"],
            "ChannelId": ["web", "android", "web", "ios", "web",
                          "android", "web", "ios", "web", "android"],
            "ProviderId": [f"PR{i % 2}" for i in range(10)],
            "PricingStrategy": [0, 1, 0, 2, 1, 0, 1, 2, 0, 1],
            "FraudResult": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        }
    )


@pytest.fixture
def sample_rfm():
    """Synthetic RFM DataFrame for testing cluster assignment."""
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "recency": [5, 100, 200, 3, 7, 150],
            "frequency": [10, 2, 1, 12, 8, 1],
            "monetary": [5000, 100, 50, 6000, 4000, 80],
        }
    )


# ---------------------------------------------------------------------------
# Test: extract_datetime_features
# ---------------------------------------------------------------------------
def test_extract_datetime_features_adds_columns(sample_transactions):
    result = extract_datetime_features(sample_transactions)
    for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]:
        assert col in result.columns, f"Missing column: {col}"


def test_extract_datetime_features_correct_values(sample_transactions):
    result = extract_datetime_features(sample_transactions)
    assert result["transaction_year"].iloc[0] == 2019
    assert result["transaction_month"].iloc[0] == 1
    assert result["transaction_day"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Test: create_aggregate_features
# ---------------------------------------------------------------------------
def test_aggregate_features_shape(sample_transactions):
    """One row per unique customer."""
    result = create_aggregate_features(sample_transactions)
    unique_customers = sample_transactions["CustomerId"].nunique()
    assert result.shape[0] == unique_customers


def test_aggregate_features_columns(sample_transactions):
    result = create_aggregate_features(sample_transactions)
    expected_cols = [
        "CustomerId",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "fraud_rate",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing aggregate column: {col}"


def test_aggregate_transaction_count(sample_transactions):
    result = create_aggregate_features(sample_transactions)
    # C1 has 3 transactions
    c1_row = result[result["CustomerId"] == "C1"]
    assert c1_row["transaction_count"].values[0] == 3


def test_aggregate_total_amount(sample_transactions):
    result = create_aggregate_features(sample_transactions)
    c3_row = result[result["CustomerId"] == "C3"]
    assert c3_row["total_transaction_amount"].values[0] == pytest.approx(1150.0)


# ---------------------------------------------------------------------------
# Test: calculate_rfm
# ---------------------------------------------------------------------------
def test_rfm_columns(sample_transactions):
    result = calculate_rfm(sample_transactions)
    for col in ["CustomerId", "recency", "frequency", "monetary"]:
        assert col in result.columns


def test_rfm_recency_nonnegative(sample_transactions):
    result = calculate_rfm(sample_transactions)
    assert (result["recency"] >= 0).all()


def test_rfm_frequency_positive(sample_transactions):
    result = calculate_rfm(sample_transactions)
    assert (result["frequency"] > 0).all()


def test_rfm_one_row_per_customer(sample_transactions):
    result = calculate_rfm(sample_transactions)
    assert result.shape[0] == sample_transactions["CustomerId"].nunique()


# ---------------------------------------------------------------------------
# Test: assign_risk_label
# ---------------------------------------------------------------------------
def test_assign_risk_label_column_exists(sample_rfm):
    result = assign_risk_label(sample_rfm)
    assert "is_high_risk" in result.columns


def test_assign_risk_label_binary(sample_rfm):
    result = assign_risk_label(sample_rfm)
    assert set(result["is_high_risk"].unique()).issubset({0, 1})


def test_assign_risk_label_has_positives(sample_rfm):
    result = assign_risk_label(sample_rfm)
    assert result["is_high_risk"].sum() > 0, "No high-risk customers identified"


# ---------------------------------------------------------------------------
# Test: encode_categoricals
# ---------------------------------------------------------------------------
def test_encode_categoricals_no_object_columns(sample_transactions):
    df = sample_transactions.copy()
    # Drop datetime for simplicity
    df = df.drop(columns=["TransactionStartTime"])
    result = encode_categoricals(df)
    object_cols = result.select_dtypes(include=["object"]).columns.tolist()
    assert len(object_cols) == 0, f"Object columns remaining: {object_cols}"


# ---------------------------------------------------------------------------
# Test: impute_missing
# ---------------------------------------------------------------------------
def test_impute_missing_no_nans():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    result = impute_missing(df)
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Test: scale_features
# ---------------------------------------------------------------------------
def test_scale_features_mean_approx_zero():
    df = pd.DataFrame(
        {"feature1": [10.0, 20.0, 30.0, 40.0, 50.0], "target": [0, 1, 0, 1, 0]}
    )
    result = scale_features(df, exclude_cols=["target"])
    assert abs(result["feature1"].mean()) < 1e-10


def test_scale_features_std_approx_one():
    df = pd.DataFrame(
        {"feature1": [10.0, 20.0, 30.0, 40.0, 50.0], "target": [0, 1, 0, 1, 0]}
    )
    result = scale_features(df, exclude_cols=["target"])
    assert abs(result["feature1"].std(ddof=0) - 1.0) < 1e-6


def test_scale_features_excludes_target():
    df = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0], "is_high_risk": [0, 1, 0]}
    )
    result = scale_features(df, exclude_cols=["is_high_risk"])
    # Target should be unchanged
    assert list(result["is_high_risk"]) == [0, 1, 0]
