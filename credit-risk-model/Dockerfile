FROM python:3.10-slim

LABEL maintainer="Bati Bank Analytics Team"
LABEL description="Credit Risk Scoring API"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        pandas numpy scikit-learn xgboost lightgbm \
        mlflow fastapi "uvicorn[standard]" pydantic \
        python-dotenv scipy

COPY src/ ./src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
