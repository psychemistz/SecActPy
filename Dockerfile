# =============================================================================
# SecActPy + SecAct Unified Docker Image
# 
# Single Dockerfile for both CPU and GPU versions
#
# Build CPU version (default, Python only):
#   docker build -t secactpy:latest .
#
# Build GPU version:
#   docker build -t secactpy:gpu --build-arg USE_GPU=true .
#
# Build with R SecAct package (slower):
#   docker build -t secactpy:with-r --build-arg INSTALL_R=true .
#   docker build -t secactpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
#
# Run CPU:
#   docker run -it --rm -v $(pwd):/workspace secactpy:latest
#
# Run GPU:
#   docker run -it --rm --gpus all -v $(pwd):/workspace secactpy:gpu
#
# Run Jupyter:
#   docker run -it --rm -p 8888:8888 -v $(pwd):/workspace secactpy:latest \
#       jupyter lab --ip=0.0.0.0 --no-browser --allow-root
# =============================================================================

# Build arguments
ARG USE_GPU=false
ARG INSTALL_R=false

# =============================================================================
# Base Image Selection
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false

# Select base image based on USE_GPU argument
FROM base-${USE_GPU} AS base

# Re-declare ARGs after FROM (required by Docker)
ARG USE_GPU=false
ARG INSTALL_R=false

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Build tools
    build-essential \
    gfortran \
    # Libraries
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgsl-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# =============================================================================
# R: Install SecAct (optional)
# =============================================================================

ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "Installing R and SecAct..." && \
        apt-get update && \
        apt-get install -y --no-install-recommends r-base r-base-dev && \
        rm -rf /var/lib/apt/lists/* && \
        R -e "install.packages(c('remotes', 'BiocManager'), repos='https://cloud.r-project.org/')" && \
        R -e "BiocManager::install(c('Biobase'), ask=FALSE, update=FALSE)" && \
        R -e "options(timeout=600); remotes::install_github('data2intelligence/SecAct', dependencies=TRUE, upgrade='never')" && \
        R -e "library(SecAct); cat('SecAct installed successfully\n')"; \
    else \
        echo "Skipping R installation"; \
    fi

# =============================================================================
# Python: Install SecActPy
# =============================================================================

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install base Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    h5py \
    anndata \
    scanpy \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab

# Install CuPy for GPU version only
ARG USE_GPU
RUN if [ "$USE_GPU" = "true" ]; then \
        echo "Installing CuPy for GPU support..." && \
        pip3 install --no-cache-dir cupy-cuda11x; \
    else \
        echo "Skipping CuPy (CPU-only build)"; \
    fi

# Install SecActPy
RUN pip3 install --no-cache-dir git+https://github.com/psychemistz/SecActPy.git

# Verify Python installation
RUN python3 -c "import secactpy; print(f'SecActPy OK, GPU: {secactpy.CUPY_AVAILABLE}')"

# =============================================================================
# Environment
# =============================================================================

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU environment variables (harmless if not using GPU)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Entry Point
# =============================================================================

CMD ["/bin/bash"]

# =============================================================================
# Labels
# =============================================================================

LABEL maintainer="Seongyong Park"
LABEL description="SecActPy + SecAct R package (CPU/GPU unified)"
LABEL version="0.1.1"
