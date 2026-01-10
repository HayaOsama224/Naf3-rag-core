FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install torch first (helps some builds)
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build llama-cpp-python with CUDA (cuBLAS)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN python3 -m pip install llama-cpp-python==0.2.90

# Install the rest
RUN python3 -m pip install -r /app/requirements.txt

# Copy code
COPY . /app

# RunPod Serverless entrypoint
CMD ["python3", "-u", "handler.py"]
