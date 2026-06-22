# Use a runtime image, but we will add build tools
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    ca-certificates \
    cuda-toolkit-12-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set paths so nvcc and libraries are found
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Configure build environment
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

RUN mkdir -p /workspace/artifacts /workspace/data /workspace/models /workspace/.cache/huggingface

# Install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /workspace/requirements.txt

# Verify compiler is present
RUN nvcc --version

# Install llama-cpp-python
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
