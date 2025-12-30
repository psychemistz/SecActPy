# =============================================================================
# SecActPy + SecAct Unified Docker Image
# 
# Single Dockerfile for both CPU and GPU versions
#
# Build CPU version (default):
#   docker build -t secactpy:latest .
#   docker build -t secactpy:cpu --build-arg USE_GPU=false .
#
# Build GPU version:
#   docker build -t secactpy:gpu --build-arg USE_GPU=true .
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

# Build argument: set to "true" for GPU support
ARG USE_GPU=false

# =============================================================================
# Base Image Selection
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false

# Select base image based on USE_GPU argument
FROM base-${USE_GPU} AS base

# Re-declare ARG after FROM (required by Docker)
ARG USE_GPU=false

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # R
    r-base \
    r-base-dev \
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
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# =============================================================================
# R: Install SecAct
# =============================================================================

RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org/')" && \
    R -e "remotes::install_github('data2intelligence/SecAct', dependencies=TRUE, upgrade='never')"

# Verify R installation
RUN R -e "library(SecAct); cat('SecAct version:', as.character(packageVersion('SecAct')), '\n')"

# =============================================================================
# Python: Install SecActPy
# =============================================================================

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install base Python packages
RUN pip3 install --no-cache-dir \
    numpy>=1.20.0 \
    pandas>=1.3.0 \
    scipy>=1.7.0 \
    h5py>=3.0.0 \
    anndata>=0.8.0 \
    scanpy>=1.9.0 \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab

# Install CuPy for GPU version only
ARG USE_GPU=false
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
