"""
data_processing.py
------------------
Feature engineering pipeline for the Bati Bank Credit Risk Model.

Transforms raw Xente eCommerce transaction data into a model-ready dataset,
including:
  - Aggregate features per customer (RFM + statistical)
  - Datetime feature extraction
  - Categorical encoding
  - Missing value imputation
  - Feature scaling
  - Proxy target variable (is_high_risk) via K-Means clustering on RFM

All steps are wrapped in a reproducible sklearn Pipeline.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
RAW_DATA_PATH = os.path.join("data", "raw", "xente_transactions.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
RANDOM_STATE = 42
N_CLUSTERS = 3


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw transaction CSV from disk."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2: Extract datetime features
# ---------------------------------------------------------------------------
def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, year from TransactionStartTime."""
    logger.info("Extracting datetime features")
    df = df.copy()
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregate features per customer
# ---------------------------------------------------------------------------
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-customer aggregate statistics from transaction history.

    Returns a customer-level DataFrame (one row per CustomerId).
    """
    logger.info("Creating aggregate features per customer")

    agg = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            std_transaction_amount=("Amount", "std"),
            transaction_count=("TransactionId", "count"),
            total_value=("Value", "sum"),
            avg_value=("Value", "mean"),
            unique_products=("ProductId", "nunique"),
            unique_categories=("ProductCategory", "nunique"),
            unique_providers=("ProviderId", "nunique"),
            fraud_count=("FraudResult", "sum"),
            fraud_rate=("FraudResult", "mean"),
            most_common_channel=("ChannelId", lambda x: x.mode()[0]),
            most_common_category=("ProductCategory", lambda x: x.mode()[0]),
            most_common_pricing=("PricingStrategy", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    logger.info(f"Aggregate features shape: {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# Step 4: Calculate RFM metrics
# ---------------------------------------------------------------------------
def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics per customer.

    Args:
        df: Raw transaction DataFrame with TransactionStartTime and Amount.
        snapshot_date: Reference date for recency calculation.
                       Defaults to max transaction date + 1 day.

    Returns:
        Customer-level DataFrame with columns: CustomerId, recency,
        frequency, monetary.
    """
    logger.info("Calculating RFM metrics")

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date for recency: {snapshot_date.date()}")

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    logger.info(f"RFM shape: {rfm.shape}")
    return rfm


# ---------------------------------------------------------------------------
# Step 5: Cluster customers and assign proxy target variable
# ---------------------------------------------------------------------------
def assign_risk_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster customers on RFM features using K-Means and label the
    highest-risk cluster as is_high_risk = 1.

    The high-risk cluster is identified as the one with the highest
    recency (least recent), lowest frequency, and lowest monetary value.

    Args:
        rfm: DataFrame with columns CustomerId, recency, frequency, monetary.

    Returns:
        rfm DataFrame with new column is_high_risk (0 or 1).
    """
    logger.info("Clustering customers on RFM for proxy target variable")

    rfm_features = rfm[["recency", "frequency", "monetary"]].copy()

    # Clip monetary outliers at 1st and 99th percentile before clustering
    # to prevent extreme values (e.g. large negative refunds) from
    # distorting the K-Means distance calculations
    for col in ["recency", "frequency", "monetary"]:
        p01 = rfm_features[col].quantile(0.01)
        p99 = rfm_features[col].quantile(0.99)
        rfm_features[col] = rfm_features[col].clip(p01, p99)
        logger.info(f"  Clipped {col}: [{p01:.1f}, {p99:.1f}]")

    # Scale before clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # Fit K-Means
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify the high-risk cluster:
    # Highest avg recency + lowest avg frequency + lowest avg monetary
    cluster_summary = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    cluster_summary["risk_score"] = (
        cluster_summary["recency"]           # high = bad (not recent)
        - cluster_summary["frequency"]        # low = bad
        - cluster_summary["monetary"]         # low = bad
    )
    high_risk_cluster = cluster_summary["risk_score"].idxmax()

    logger.info(f"Cluster summary:\n{cluster_summary}")
    logger.info(f"High-risk cluster identified: {high_risk_cluster}")

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    high_risk_count = rfm["is_high_risk"].sum()
    logger.info(
        f"High-risk customers: {high_risk_count} / {len(rfm)} "
        f"({high_risk_count / len(rfm) * 100:.1f}%)"
    )

    return rfm[["CustomerId", "recency", "frequency", "monetary", "is_high_risk"]]


# ---------------------------------------------------------------------------
# Step 6: Encode categorical features
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string categorical columns."""
    logger.info("Encoding categorical features")
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.debug(f"  Encoded: {col}")

    logger.info(f"Encoded {len(cat_cols)} categorical columns")
    return df


# ---------------------------------------------------------------------------
# Step 7: Impute missing values
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median; categorical NaNs already encoded as strings."""
    logger.info("Imputing missing values")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df


# ---------------------------------------------------------------------------
# Step 8: Scale numerical features
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Standardize numerical features (mean=0, std=1).

    Args:
        df: Feature DataFrame.
        exclude_cols: Columns to skip (e.g., target, IDs).

    Returns:
        Scaled DataFrame.
    """
    logger.info("Scaling numerical features")
    exclude_cols = exclude_cols or []
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_feature_dataset(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    End-to-end pipeline: raw CSV → model-ready customer-level DataFrame
    with is_high_risk target column.

    Returns:
        Processed DataFrame ready for model training.
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 60)

    # Load
    df = load_raw_data(raw_path)

    # Datetime features (transaction-level)
    df = extract_datetime_features(df)

    # Aggregate to customer level
    agg = create_aggregate_features(df)

    # RFM
    rfm = calculate_rfm(df)

    # Proxy target variable
    rfm_labeled = assign_risk_label(rfm)

    # Merge aggregate features with RFM + labels
    final = agg.merge(rfm_labeled, on="CustomerId", how="left")

    # Encode categoricals
    final = encode_categoricals(final)

    # Impute
    final = impute_missing(final)

    # Scale (excluding ID and target)
    final = scale_features(final, exclude_cols=["CustomerId", "is_high_risk"])

    logger.info(f"Final dataset shape: {final.shape}")
    logger.info(
        f"Target distribution:\n{final['is_high_risk'].value_counts(normalize=True)}"
    )

    return final


def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    """Save the processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = build_feature_dataset()
    save_processed_data(dataset)
    logger.info("✅ Feature engineering complete.")
