FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Build-time environment variables for GPU support
ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CMAKE=1 \
    CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=${CUDA_HOME}"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
      
COPY requirements.txt .

# 1. Install standard dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# 2. Install FAISS GPU (Standard wheel)
RUN pip install --no-cache-dir faiss-gpu
# 3. Force a source build of llama-cpp-python (This one DOES support --no-binary)
RUN pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python 

# Verify GPU availability during build (if the builder has a GPU)
# If this fails, it's okay, but the above builds must succeed
RUN nvcc --version

COPY rag_core.py handler.py /workspace/
COPY data/ /workspace/data/

CMD ["python", "-u", "handler.py"]
