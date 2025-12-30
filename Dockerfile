# =============================================================================
# SecActPy + SecAct/RidgeR Unified Docker Image
# 
# Single Dockerfile for both CPU and GPU versions
#
# Build CPU version (default, Python only):
#   docker build -t secactpy:latest .
#
# Build GPU version:
#   docker build -t secactpy:gpu --build-arg USE_GPU=true .
#
# Build with R SecAct/RidgeR package (slower):
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
    cmake \
    pkg-config \
    # Libraries for R packages
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgsl-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libxt-dev \
    libmagick++-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    # For R Matrix package
    liblapack-dev \
    libblas-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set locale (needed for some R packages)
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# R: Install R and Packages (optional)
# =============================================================================

ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing R..." && \
        echo "========================================" && \
        # Add CRAN repository for latest R
        apt-get update && \
        apt-get install -y --no-install-recommends \
            dirmngr \
            gnupg \
            apt-transport-https \
            ca-certificates && \
        wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
            gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | \
            tee /etc/apt/sources.list.d/r-project.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends r-base r-base-dev && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping R installation"; \
    fi

# Install R packages in stages for better error handling
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing CRAN packages..." && \
        echo "========================================" && \
        R -e "options(repos = c(CRAN = 'https://cloud.r-project.org/')); \
              install.packages(c( \
                  'remotes', \
                  'BiocManager', \
                  'devtools', \
                  'Matrix', \
                  'ggplot2', \
                  'dplyr', \
                  'tidyr', \
                  'data.table', \
                  'Rcpp', \
                  'RcppArmadillo', \
                  'RcppEigen', \
                  'httr', \
                  'jsonlite', \
                  'R6', \
                  'crayon' \
              ), dependencies = TRUE)"; \
    fi

# Install Bioconductor packages
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing Bioconductor packages..." && \
        echo "========================================" && \
        R -e "BiocManager::install(version = '3.18', ask = FALSE, update = FALSE)" && \
        R -e "BiocManager::install(c( \
                  'Biobase', \
                  'S4Vectors', \
                  'IRanges', \
                  'GenomicRanges', \
                  'SummarizedExperiment', \
                  'SingleCellExperiment' \
              ), ask = FALSE, update = FALSE)"; \
    fi

# Install SecAct from GitHub
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing SecAct from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              remotes::install_github('data2intelligence/SecAct', \
                  dependencies = TRUE, \
                  upgrade = 'never', \
                  force = TRUE)" && \
        R -e "library(SecAct); cat('SecAct version:', as.character(packageVersion('SecAct')), '\n')"; \
    fi

# Install RidgeR from GitHub (optional, for cross-validation)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing RidgeR from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('beibeiru/RidgeR', \
                      dependencies = TRUE, \
                      upgrade = 'never', \
                      force = TRUE); \
                  library(RidgeR); \
                  cat('RidgeR version:', as.character(packageVersion('RidgeR')), '\n') \
              }, error = function(e) { \
                  cat('RidgeR installation skipped:', conditionMessage(e), '\n') \
              })"; \
    fi

# Install SpaCET from GitHub (optional, for spatial transcriptomics)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing SpaCET from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('data2intelligence/SpaCET', \
                      dependencies = TRUE, \
                      upgrade = 'never', \
                      force = TRUE); \
                  library(SpaCET); \
                  cat('SpaCET version:', as.character(packageVersion('SpaCET')), '\n') \
              }, error = function(e) { \
                  cat('SpaCET installation skipped:', conditionMessage(e), '\n') \
              })"; \
    fi

# Verify R installation
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Verifying R installation..." && \
        echo "========================================" && \
        R -e "cat('R version:', R.version.string, '\n'); \
              installed <- installed.packages()[, 'Package']; \
              for (pkg in c('SecAct', 'Biobase', 'Matrix')) { \
                  if (pkg %in% installed) { \
                      cat(pkg, 'OK\n') \
                  } else { \
                      cat(pkg, 'NOT FOUND\n') \
                  } \
              }"; \
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

# Install SecActPy from GitHub
RUN pip3 install --no-cache-dir git+https://github.com/psychemistz/SecActPy.git

# Verify Python installation
RUN python3 -c "import secactpy; print(f'SecActPy {secactpy.__version__} OK, GPU: {secactpy.CUPY_AVAILABLE}')"

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
LABEL description="SecActPy + SecAct/RidgeR (CPU/GPU unified)"
LABEL version="0.1.1"
