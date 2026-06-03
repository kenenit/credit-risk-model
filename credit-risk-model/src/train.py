"""
train.py
--------
Model training, hyperparameter tuning, and MLflow experiment tracking
for the Bati Bank Credit Risk Model.

Models trained:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM

The best model (highest ROC-AUC) is registered in the MLflow Model Registry.
"""

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
TARGET_COL = "is_high_risk"
ID_COL = "CustomerId"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "bati_bank_credit_risk"
REGISTERED_MODEL_NAME = "credit_risk_best_model"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_processed_data(path: str = PROCESSED_DATA_PATH):
    """Load the processed feature dataset and split into X, y."""
    logger.info(f"Loading processed data from {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL]
    logger.info(f"Features: {X.shape[1]} | Samples: {len(y)}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
    return X, y


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict of standard classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"
        ),
        "param_dist": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "n_iter": 5,
    },
    "random_forest": {
        "model": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "param_dist": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "n_iter": 8,
    },
    "xgboost": {
        "model": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "param_dist": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        },
        "n_iter": 10,
    },
    "lightgbm": {
        "model": LGBMClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", verbose=-1
        ),
        "param_dist": {
            "n_estimators": [100, 200, 300],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 100],
        },
        "n_iter": 10,
    },
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_and_track(X_train, X_test, y_train, y_test):
    """
    Train all models with RandomizedSearchCV, log each run to MLflow,
    and return the name + run_id + AUC of the best model.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_auc = -1.0
    best_run_id = None
    best_model_name = None

    for model_name, config in MODEL_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")

        with mlflow.start_run(run_name=model_name):

            # Hyperparameter search
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["param_dist"],
                n_iter=config["n_iter"],
                scoring="roc_auc",
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_

            # Evaluate
            metrics = evaluate_model(best_estimator, X_test, y_test)

            # Log to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

            logger.info(f"Best params: {search.best_params_}")
            logger.info(
                f"Metrics - AUC: {metrics['roc_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"Precision: {metrics['precision']:.4f}"
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run_id
                best_model_name = model_name

    return best_model_name, best_run_id, best_auc


# ---------------------------------------------------------------------------
# Register best model
# ---------------------------------------------------------------------------
def register_best_model(run_id: str, model_name: str) -> None:
    """Register the best model in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from run {run_id} as '{REGISTERED_MODEL_NAME}'")
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info("Model registered successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Load data
    X, y = load_processed_data()

    # Train / test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Train all models and track with MLflow
    best_name, best_run_id, best_auc = train_and_track(
        X_train, X_test, y_train, y_test
    )

    logger.info(f"\nBest model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # Register best model in MLflow Model Registry
    register_best_model(best_run_id, best_name)
    logger.info("Training pipeline complete.")
