# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a Credit Risk Model using alternative eCommerce transaction data for Bati Bank's buy-now-pay-later service.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Training & Tracking](#model-training--tracking)
- [API Deployment](#api-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Team](#team)

---

## Project Overview

Bati Bank is partnering with an eCommerce company to enable a **buy-now-pay-later** service. This project builds a Credit Scoring Model using transaction data from the Xente eCommerce platform to:

1. Define a proxy variable to categorize users as high-risk or low-risk
2. Select observable features with high correlation to the default variable
3. Develop a model that assigns **risk probability** for new customers
4. Develop a model that assigns a **credit score** from risk probability estimates
5. Develop a model that predicts the **optimal loan amount and duration**

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord mandates that financial institutions hold regulatory capital proportional to the credit risk they carry. To satisfy this requirement, banks must not only produce accurate risk estimates — they must be able to **explain, validate, and audit** how those estimates are derived. This has direct consequences for model design:

- **Interpretability is regulatory, not optional.** Basel II's Internal Ratings-Based (IRB) approach requires banks to document the logic behind Probability of Default (PD) estimates. A "black-box" model, however accurate, cannot satisfy examiners who need to trace a decision back to observable borrower characteristics. Regulators expect a clear narrative: which features drive risk, how they are weighted, and why.

- **Documentation supports model governance.** Basel II requires periodic model validation, stress testing, and back-testing against realized defaults. A well-documented pipeline — with versioned data, tracked experiments, and recorded hyperparameters — makes this lifecycle feasible. An undocumented model is, effectively, an unauditable one.

- **Stability and conservatism matter as much as accuracy.** Basel II encourages models that behave predictably across economic cycles. A highly tuned, complex model that overfits recent data may produce unstable capital estimates — a serious regulatory concern. Interpretable models like Logistic Regression with Weight of Evidence (WoE) transformations tend to be more stable and are easier to stress-test.

In this project, these requirements translate to: logging every experiment with MLflow, justifying every feature engineering choice, and documenting the proxy variable design with explicit acknowledgment of its limitations.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The Xente dataset contains transaction records with no ground-truth default label — no record of whether a customer failed to repay a loan. Yet the business objective requires a binary classifier that scores creditworthiness. A **proxy variable** bridges this gap by using observable behavioral signals to approximate the unobserved outcome.

**Why RFM-based proxies make sense:**
- **Recency** captures how recently a customer transacted. A customer who has not engaged in months may be financially stressed or disengaged — behaviors correlated with default risk.
- **Frequency** reflects loyalty and consistency. Infrequent customers have a thinner behavioral history, making risk assessment harder and default more plausible.
- **Monetary** value signals purchasing power and platform engagement. Low monetary value combined with low frequency is a classic pattern of marginal or at-risk customers.

By clustering customers on these three dimensions with K-Means, we identify a segment that is behaviorally disengaged — our **high-risk proxy**.

**Business risks this approach introduces:**

| Risk | Description |
|---|---|
| **Label noise** | The proxy may mislabel genuinely creditworthy customers who simply transact infrequently (e.g., seasonal buyers). This inflates false positives and may lead to unjust credit denials. |
| **Distributional shift** | Behavioral patterns in eCommerce data may not generalize to loan repayment behavior. A customer who buys infrequently online may be a reliable loan repayer offline. |
| **Feedback loops** | If the model systematically denies credit to proxy-labeled high-risk groups, those groups never get the chance to demonstrate creditworthiness — self-fulfilling the proxy's prediction. |
| **Regulatory scrutiny** | Regulators may challenge the validity of a proxy-based PD estimate under Basel II's documentation standards. The proxy must be explicitly justified and its limitations disclosed. |
| **Concept drift** | RFM patterns shift over time (e.g., during economic downturns or promotional periods). The proxy definition must be monitored and potentially recalibrated. |

This project acknowledges these risks explicitly and treats the proxy label as a **modeling assumption**, not ground truth.

---

### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| **Interpretability** | High — coefficients are directly interpretable as log-odds; WoE bins provide intuitive risk bands | Low — ensemble of hundreds of trees; requires SHAP or LIME for post-hoc explanation |
| **Regulatory acceptance** | Widely accepted; Basel II scorecards are traditionally built this way | Increasingly accepted but requires additional explainability tooling and documentation |
| **Performance on complex data** | Moderate — assumes linear decision boundary in WoE space; may underfit non-linear patterns | High — captures non-linear interactions and feature dependencies automatically |
| **Handling class imbalance** | Requires explicit handling (class weights, resampling) | More robust but still benefits from imbalance-aware techniques |
| **Overfitting risk** | Low — few parameters, strong regularization | Higher — requires careful hyperparameter tuning and cross-validation |
| **Feature engineering burden** | High — requires WoE binning, IV selection, and careful monotonicity checks | Lower — can learn complex features automatically |
| **Scorecard conversion** | Straightforward — linear model maps cleanly to a points-based scorecard | Difficult — non-linear structure resists direct scorecard translation |
| **Monitoring & stability** | Easy — coefficients are stable and interpretable across time | Harder — feature importance can shift, requiring drift detection |
| **Speed of scoring** | Very fast at inference | Fast, but slightly heavier |

**Recommendation for this project:** We train both model families and compare them. The Logistic Regression + WoE model serves as the **interpretable baseline** aligned with Basel II expectations. Gradient Boosting models are trained as **performance benchmarks**. The final deployment choice is guided by the regulatory context — if the bank's risk team requires a scorecard they can explain to examiners, interpretability wins even at some performance cost.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline (linting + tests)
├── data/
│   ├── raw/                      # Raw data (gitignored)
│   └── processed/                # Processed features (gitignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Feature engineering pipeline
│   ├── train.py                  # Model training & MLflow tracking
│   ├── predict.py                # Inference utilities
│   └── api/
│       ├── main.py               # FastAPI application
│       └── pydantic_models.py    # Request/response schemas
├── tests/
│   └── test_data_processing.py   # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle (Xente Challenge)
# Place the CSV in data/raw/
```

---

## Usage

### Run Feature Engineering

```bash
python src/data_processing.py
```

### Train Models

```bash
python src/train.py
```

### Launch MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Model Training & Tracking

All experiments are tracked via MLflow. Models trained include:
- Logistic Regression (with WoE features)
- Random Forest
- XGBoost
- LightGBM

The best model is registered in the **MLflow Model Registry** and loaded by the API at runtime.

---

## API Deployment

### Run with Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "total_transaction_amount": 15000.0,
    "avg_transaction_amount": 3000.0,
    "transaction_count": 5,
    "recency_days": 10,
    "frequency": 5,
    "monetary": 15000.0
  }'
```

### Sample Response

```json
{
  "customer_id": "C001",
  "risk_probability": 0.12,
  "credit_score": 720,
  "risk_label": "low_risk"
}
```

---

## CI/CD Pipeline

GitHub Actions runs automatically on every push to `main`:
1. **Lint** — `flake8` checks code style
2. **Test** — `pytest` runs all unit tests

The build fails if either step fails.

---

## Team

- Kerod
- Mahbubah
- Feven

---

## References

- [Basel II Capital Accord](https://www.bis.org/publ/bcbs128.pdf)
- [Alternative Credit Scoring — HKMA](https://www.hkma.gov.hk)
- [Credit Scoring Approaches — World Bank](https://www.worldbank.org)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Xente Challenge — Kaggle](https://www.kaggle.com/c/xente-fraud-detection)
