# Docker Usage Guide

This guide explains how to use SecActPy with Docker, including both CPU and GPU versions.

## Quick Start

### Build Images

```bash
# CPU version (default, Python only - fast build)
docker build -t secactpy:latest .

# CPU version with R SecAct package (slower)
docker build -t secactpy:with-r --build-arg INSTALL_R=true .

# GPU version (Python only)
docker build -t secactpy:gpu --build-arg USE_GPU=true .

# GPU version with R SecAct package
docker build -t secactpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Set to `true` for GPU/CUDA support |
| `INSTALL_R` | `false` | Set to `true` to include R and SecAct package |

### Run Containers

```bash
# Interactive (CPU)
docker run -it --rm -v $(pwd):/workspace secactpy:latest

# Interactive (GPU)
docker run -it --rm --gpus all -v $(pwd):/workspace secactpy:gpu

# Run a Python script
docker run --rm -v $(pwd):/workspace secactpy:latest python your_script.py

# Run an R script (if R installed)
docker run --rm -v $(pwd):/workspace secactpy:latest Rscript your_script.R
```

## Using Docker Compose

Docker Compose simplifies container management:

```bash
# Start CPU container in background
docker-compose up -d secactpy

# Start GPU container in background
docker-compose up -d secactpy-gpu

# Enter the container
docker-compose exec secactpy bash

# Stop all containers
docker-compose down
```

### Jupyter Lab

```bash
# Start Jupyter Lab (CPU)
docker-compose up secactpy-jupyter
# Open http://localhost:8888

# Start Jupyter Lab (GPU)
docker-compose up secactpy-jupyter-gpu
# Open http://localhost:8889
```

## What's Included

### Python Packages (always installed)
- **SecActPy**: Python implementation
- **NumPy, Pandas, SciPy**: Core scientific computing
- **AnnData, Scanpy**: Single-cell analysis
- **h5py**: HDF5 file support
- **Jupyter**: Interactive notebooks
- **CuPy** (GPU version only): GPU acceleration

### R Packages (when INSTALL_R=true)
- **SecAct**: Original R implementation
- **Biobase**: Bioconductor dependency

## Validation Example

Run validation tests inside the container:

```bash
# Enter container
docker run -it --rm -v $(pwd):/workspace secactpy:latest

# Inside container - Python
python -c "
import secactpy
import pandas as pd

# Quick test
print(f'SecActPy imported: {secactpy.__name__}')
print(f'GPU available: {secactpy.CUPY_AVAILABLE}')
"

# Inside container - R
Rscript -e "
library(SecAct)
cat('SecAct version:', as.character(packageVersion('SecAct')), '\n')
"
```

## Cross-Validation (R vs Python)

Compare R and Python outputs:

```bash
# Enter container with your data mounted
docker run -it --rm -v $(pwd):/workspace secactpy:latest

# Run R inference
Rscript -e "
library(SecAct)
data <- read.table('your_data.txt', row.names=1, header=TRUE)
result <- SecAct.inference.gsl(data)
write.table(result\$zscore, 'r_zscore.txt', quote=FALSE)
"

# Run Python inference
python -c "
import pandas as pd
from secactpy import secact_activity_inference

data = pd.read_csv('your_data.txt', sep='\t', index_col=0)
result = secact_activity_inference(data, is_differential=True)
result['zscore'].to_csv('py_zscore.txt', sep='\t')
"

# Compare results
python -c "
import pandas as pd
import numpy as np

r_result = pd.read_csv('r_zscore.txt', sep='\t', index_col=0)
py_result = pd.read_csv('py_zscore.txt', sep='\t', index_col=0)

diff = np.abs(r_result.values - py_result.values)
print(f'Max difference: {diff.max():.2e}')
print(f'Mean difference: {diff.mean():.2e}')
"
```

## GPU Support

### Prerequisites

1. NVIDIA GPU with CUDA support
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Installation (Ubuntu)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Check GPU is accessible
docker run --rm --gpus all secactpy:gpu nvidia-smi

# Check CuPy
docker run --rm --gpus all secactpy:gpu python -c "
import cupy as cp
print(f'CuPy version: {cp.__version__}')
print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')
"
```

## Customization

### Build Arguments

The unified Dockerfile supports the following build arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Set to `true` for GPU support |
| `INSTALL_R` | `true` | Set to `false` to skip R/SecAct (faster build) |

```bash
# Examples
docker build -t secactpy:cpu .                                              # CPU, Python only (fast)
docker build -t secactpy:cpu-with-r --build-arg INSTALL_R=true .            # CPU, with R
docker build -t secactpy:gpu --build-arg USE_GPU=true .                     # GPU, Python only
docker build -t secactpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .  # GPU, with R
```

### Add Your Own Packages

Create a custom Dockerfile:

```dockerfile
FROM secactpy:latest

# Add R packages (if R is installed)
RUN R -e "install.packages('your_package', repos='https://cloud.r-project.org/')" || true

# Add Python packages
RUN pip3 install your_package
```

### Mount Additional Data

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/data:/data:ro \
  -v /path/to/results:/results \
  secactpy:latest
```

## Troubleshooting

### Permission Issues

```bash
# Run as current user
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/workspace secactpy:latest
```

### Memory Issues

```bash
# Increase memory limit
docker run -it --rm -m 16g -v $(pwd):/workspace secactpy:latest
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu

# Verify nvidia-container-toolkit
which nvidia-container-toolkit
```

## Image Sizes

| Build | Base Image | Approximate Size |
|-------|------------|-----------------|
| CPU, no R | ubuntu:22.04 | ~2 GB |
| CPU, with R | ubuntu:22.04 | ~3.5 GB |
| GPU, no R | nvidia/cuda:11.8.0 | ~6 GB |
| GPU, with R | nvidia/cuda:11.8.0 | ~8 GB |

> **Note**: CI/CD builds use `INSTALL_R=false` for faster builds. For local builds with R validation, use `--build-arg INSTALL_R=true`.

## Pushing to Docker Hub

```bash
# Tag
docker tag secactpy:latest yourusername/secactpy:latest
docker tag secactpy:gpu yourusername/secactpy:gpu

# Login
docker login

# Push
docker push yourusername/secactpy:latest
docker push yourusername/secactpy:gpu
```
