# --- Core RAG requirements ---
# Use a newer PyTorch image that supports CUDA 12.8
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Set CUDA paths explicitly
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set build arguments for llama-cpp-python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=${CUDA_HOME}"

WORKDIR /workspace

# Install essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/artifacts /workspace/data /workspace/models /workspace/.cache/huggingface

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /workspace/requirements.txt

# Verify CUDA is visible to the compiler before building
RUN nvcc --version

# Reinstall llama-cpp-python to compile against CUDA 12.8
RUN pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python==0.3.16

COPY rag_core.py /workspace/rag_core.py
COPY handler.py /workspace/handler.py
COPY data/ /workspace/data/

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
