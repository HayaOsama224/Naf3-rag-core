FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (build tools + git required for some builds/logging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential cmake ninja-build pkg-config \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip build tooling
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install CUDA Torch (matches CUDA 12.1 base image)
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build llama-cpp-python with CUDA
ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUBLAS=on"
ENV FORCE_CMAKE=1
RUN python3 -m pip install llama-cpp-python==0.2.90

# Install the rest of your dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Copy code (includes rag_core.py, handler.py, data/, artifacts/, etc.)
COPY . /app

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
