# ============================================================
# BAN6800 Capstone — Stanbic IBTC Credit Risk Intelligence Platform
# Dockerfile — Data Pipeline Container
# Build:  docker build -t crip-pipeline .
# Run:    docker run crip-pipeline
# ============================================================

# ── Base image ───────────────────────────────────────────────
FROM python:3.10-slim

# ── Labels ───────────────────────────────────────────────────
LABEL maintainer="Stanbic IBTC CRIP Team"
LABEL project="BAN6800 Credit Risk Intelligence Platform"
LABEL version="1.0.0"
LABEL description="Data pipeline for credit risk analytics"

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (pinned) ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────
COPY src/    ./src/
COPY dags/   ./dags/
COPY data/   ./data/
COPY tests/  ./tests/
COPY .env.example .env

# ── Environment variables ─────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIPELINE_ENV=production
ENV LOG_LEVEL=INFO

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas; import xgboost; print('OK')" || exit 1

# ── Default entrypoint ────────────────────────────────────────
CMD ["python", "-m", "src.data.ingest"]
