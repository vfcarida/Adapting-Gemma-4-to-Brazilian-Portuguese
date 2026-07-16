# =============================================================================
# Dockerfile — Gemma 4 Portuguese Adaptation
# =============================================================================
# Multi-stage build for training on GCP (A100/H100 GPUs)
# Base: NVIDIA CUDA 12.4 + Python 3.11 + PyTorch 2.4+
#
# Usage:
#   docker build -t gemma4-pt-br .
#   docker run --gpus all -v /mnt/data:/workspace/data gemma4-pt-br gemma4pt preflight
# =============================================================================

# --- Stage 1: Base with CUDA and system deps ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    htop \
    tmux \
    vim \
    build-essential \
    ninja-build \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Stage 2: Python dependencies ---
FROM base AS deps

WORKDIR /workspace

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (requires CUDA build)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- Stage 3: Application ---
FROM deps AS app

WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY Makefile .

# Install project in editable mode
RUN pip install --no-cache-dir -e ".[dev,gpu,eval,monitoring]"

# Create output directories
RUN mkdir -p outputs/cpt_pilot outputs/cpt_main outputs/residual_merge \
    outputs/eval_cache outputs/data_qc outputs/contamination reports

# Default: show help
ENTRYPOINT ["gemma4pt"]
CMD ["--help"]
