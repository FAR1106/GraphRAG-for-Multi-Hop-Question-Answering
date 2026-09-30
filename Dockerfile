# =============================================================================
# Multi-stage Dockerfile for GraphRAG Multi-Hop QA
# =============================================================================
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY results/ ./results/

# Create necessary directories
RUN mkdir -p data/raw data/processed data/cache results logs

# =============================================================================
# API stage
# =============================================================================
FROM base AS api

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Frontend stage
# =============================================================================
FROM base AS frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
