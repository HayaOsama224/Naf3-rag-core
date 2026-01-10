# Dockerfile (RunPod Serverless, CUDA 12.6, Python 3.12, FAISS-GPU, llama-cpp-python with CUDA)
# - Starts RunPod serverless via handler.py
# - Builds llama-cpp-python with GGML_CUDA enabled so n_gpu_layers can offload to GPU
# - Uses faiss-gpu-cu12 for GPU vector search (optional via USE_FAISS_GPU=1)

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace cache (good for RunPod persistent volumes)
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    # Keep logs quieter
    TOKENIZERS_PARALLELISM=false \
    # llama-cpp-python: force CUDA build
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on"

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# (Optional) create folders your app expects
RUN mkdir -p /workspace/artifacts /workspace/data /workspace/models /workspace/.cache/huggingface

# ---- install Python deps ----
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# NOTE: We intentionally do NOT pin torch in requirements.txt.
# But we DO need a CUDA-enabled torch in the image.
# Install a CUDA 12.6 torch wheel (works with your torch 2.9.0+cu126 style setup).
RUN python3.12 -m pip install --upgrade pip setuptools wheel \
 && python3.12 -m pip install --index-url https://download.pytorch.org/whl/cu126 \
      torch torchvision torchaudio \
 && python3.12 -m pip install -r /workspace/requirements.txt

# ---- copy app code ----
# Make sure these exist in your build context:
# - handler.py
# - rag_core.py
COPY handler.py /workspace/handler.py
COPY rag_core.py /workspace/rag_core.py

# ---- runtime env defaults (override in RunPod template if you want) ----
ENV DATA_DIR=/workspace/data \
    INDEX_PATH=/workspace/artifacts/faq.index \
    DOC_STORE_PATH=/workspace/artifacts/faq_docs.pkl \
    USE_FAISS_GPU=1 \
    FAISS_GPU_DEVICE=0 \
    N_GPU_LAYERS=35 \
    N_BATCH=512 \
    N_CTX=4096 \
    TOP_K=5

# RunPod Serverless: run handler
CMD ["python3.12", "-u", "handler.py"]
