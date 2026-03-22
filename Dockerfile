# ============================================================
# Crypto Oracle — Dockerfile
# Requires NVIDIA GPU with CUDA 12.x support.
# Build:  docker build -t crypto-oracle .
# Run:    docker-compose up
# ============================================================

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-train.txt

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY .env.example .env.example

# Create data directories (mounted as volumes in compose)
RUN mkdir -p data/ohlcv data/backtest_results data/sentiment data/macro \
             data/onchain datasets models logs

# Default: run the API server
# Override with --command in docker-compose for the monitor daemon
EXPOSE 8000
CMD ["python", "src/api/api_server.py"]
