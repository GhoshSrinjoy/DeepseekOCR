FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during build (standard Docker best practice)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install exact version required by DeepSeek-OCR
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv (copy from official image - best practice)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure uv to use system Python
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install numpy first to avoid version conflicts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "numpy<2"

# EXACT versions from DeepSeek-OCR GitHub
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install build dependencies for flash-attn
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system packaging wheel setuptools psutil ninja

# Install flash-attn==2.7.3 --no-build-isolation (EXACT from GitHub)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system flash-attn==2.7.3 --no-build-isolation

# Install all required dependencies from requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Create models and cache directories
RUN mkdir -p /data/models /data/models/.cache

# Copy the backend script and .env file
COPY ocr_backend.py /app/
COPY .env* /app/

# Set environment variables
ENV HF_HOME=/data/models/.cache
ENV PYTHONUNBUFFERED=1

# Expose port for Flask API
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask server
CMD ["python", "ocr_backend.py"]
