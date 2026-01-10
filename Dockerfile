# Dockerfile (RunPod Serverless) — CUDA 12.6 + Torch already included
FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    # Build llama-cpp-python with CUDA support
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on"

WORKDIR /workspace

# System deps for building llama-cpp-python (and general utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create expected dirs
RUN mkdir -p /workspace/artifacts /workspace/data /workspace/models /workspace/.cache/huggingface

# Install Python deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /workspace/requirements.txt

# Copy app
COPY rag_core.py /workspace/rag_core.py
COPY handler.py /workspace/handler.py

# Default env (override in RunPod template if you want)
ENV DATA_DIR=/workspace/data \
    INDEX_PATH=/workspace/artifacts/faq.index \
    DOC_STORE_PATH=/workspace/artifacts/faq_docs.pkl \
    USE_FAISS_GPU=1 \
    FAISS_GPU_DEVICE=0 \
    N_GPU_LAYERS=35 \
    N_BATCH=512 \
    N_CTX=4096 \
    TOP_K=5

CMD ["python", "-u", "handler.py"]
