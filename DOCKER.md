# Docker Usage Guide

This guide explains how to use SecActPy with Docker, including both CPU and GPU versions.

## Quick Start

### Build Images

```bash
# CPU version (default)
docker build -t secactpy:latest .

# GPU version
docker build -t secactpy:gpu --build-arg USE_GPU=true .
```

### Run Containers

```bash
# Interactive (CPU)
docker run -it --rm -v $(pwd):/workspace secactpy:latest

# Interactive (GPU)
docker run -it --rm --gpus all -v $(pwd):/workspace secactpy:gpu

# Run a Python script
docker run --rm -v $(pwd):/workspace secactpy:latest python your_script.py

# Run an R script
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

### R Packages
- **SecAct**: Original R implementation
- Dependencies: Biobase, preprocessCore, etc.

### Python Packages
- **SecActPy**: Python implementation
- **NumPy, Pandas, SciPy**: Core scientific computing
- **AnnData, Scanpy**: Single-cell analysis
- **h5py**: HDF5 file support
- **Jupyter**: Interactive notebooks
- **CuPy** (GPU version only): GPU acceleration

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

The unified Dockerfile supports the following build argument:

| Argument | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Set to `true` for GPU support |

```bash
# Examples
docker build -t secactpy:cpu --build-arg USE_GPU=false .
docker build -t secactpy:gpu --build-arg USE_GPU=true .
```

### Add Your Own Packages

Create a custom Dockerfile:

```dockerfile
FROM secactpy:latest

# Add R packages
RUN R -e "install.packages('your_package', repos='https://cloud.r-project.org/')"

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
| CPU (`USE_GPU=false`) | ubuntu:22.04 | ~3 GB |
| GPU (`USE_GPU=true`) | nvidia/cuda:11.8.0 | ~8 GB |

## Pushing to Docker Hub

```bash
# Tag
docker tag secactpy:latest psychemistz/secactpy:latest
docker tag secactpy:gpu psychemistz/secactpy:gpu

# Login
docker login

# Push
docker push psychemistz/secactpy:latest
docker push psychemistz/secactpy:gpu
```
