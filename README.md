# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SecActPy is a Python package for inferring secreted protein (e.g. cytokine/chemokine) activity from gene expression data using ridge regression with permutation-based significance testing.

**Key Features:**
- 🎯 **SecAct Compatible**: Produces identical results to the R SecAct package
- 🚀 **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- 📊 **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- 🔬 **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- 🧬 **Scanpy Integration**: Direct conversion to AnnData format

## Installation

### Basic Installation

```bash
pip install git+https://github.com/psychemistz/SecActPy.git
```

### With Optional Dependencies

```bash
# With h5py and anndata for I/O
pip install "secactpy[io] @ git+https://github.com/psychemistz/SecActPy.git"

# With GPU support (requires CUDA)
pip install "secactpy[gpu] @ git+https://github.com/psychemistz/SecActPy.git"

# All optional dependencies
pip install "secactpy[all] @ git+https://github.com/psychemistz/SecActPy.git"
```

### Development Installation

```bash
git clone https://github.com/psychemistz/SecActPy.git
cd SecActPy
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from secactpy import secact_activity, load_signature

# Load the built-in SecAct signature
signature = load_signature()  # Default: 'secact'
# Or use CytoSig: signature = load_signature('cytosig')

# Load your expression data (genes × samples)
expression = pd.read_csv("expression.csv", index_col=0)

# Run inference
result = secact_activity(expression, signature)

# Access results (all are DataFrames with proper labels)
activity = result['zscore']    # Activity z-scores
pvalues = result['pvalue']     # P-values
coefficients = result['beta']  # Regression coefficients

# Find significant activities
significant = result['pvalue'] < 0.05
print(f"Significant: {significant.sum().sum()} / {significant.size}")
```

### Differential Expression Analysis

```python
from secactpy import secact_activity, load_signature, compute_differential

# Load treatment and control expression data
treatment = pd.read_csv("treatment.csv", index_col=0)
control = pd.read_csv("control.csv", index_col=0)

# Compute differential expression
diff_expr = compute_differential(treatment, control)

# Run inference on differential expression
signature = load_signature()
result = secact_activity(diff_expr, signature)
```

### Large-Scale Analysis (>100k samples)

```python
from secactpy import ridge_batch, estimate_batch_size, load_results

# Estimate optimal batch size for available memory
batch_size = estimate_batch_size(
    n_genes=20000, 
    n_features=50, 
    available_gb=8.0  # Available RAM in GB
)

# Prepare data as numpy arrays
X = signature.values  # (n_genes, n_features)
Y = expression.values  # (n_genes, n_samples)

# Stream results directly to disk
ridge_batch(
    X, Y,
    batch_size=batch_size,
    output_path="results.h5ad",
    feature_names=signature.columns.tolist(),
    sample_names=expression.columns.tolist()
)

# Load results later
results = load_results("results.h5ad")
```

### Integration with Scanpy

```python
from secactpy import load_results, results_to_anndata

# Load results and convert to AnnData
results = load_results("results.h5ad")
adata = results_to_anndata(results)

# Use with scanpy
import scanpy as sc
sc.pl.heatmap(adata, var_names=adata.var_names[:20])
```

## API Reference

### High-Level Functions

| Function | Description |
|----------|-------------|
| `secact_activity(expression, signature)` | Main inference function |
| `load_signature(name='secact')` | Load built-in signature matrix |
| `compute_differential(treatment, control)` | Compute differential expression |

### Signature Loading

```python
from secactpy import load_signature, load_secact, load_cytosig, list_signatures

# List available signatures
print(list_signatures())  # ['secact', 'cytosig']

# Load signatures
secact_sig = load_signature('secact')  # or load_secact()
cytosig_sig = load_signature('cytosig')  # or load_cytosig()

# Load subset of features
sig = load_signature('secact', features=['IL6', 'TNF', 'IFNG'])
```

### Batch Processing

```python
from secactpy import ridge_batch, estimate_batch_size, estimate_memory

# Estimate memory requirements
mem = estimate_memory(n_genes=20000, n_features=50, n_samples=100000)
print(f"Estimated memory: {mem['total']:.2f} GB")

# Estimate batch size
batch_size = estimate_batch_size(n_genes=20000, n_features=50, available_gb=8.0)

# Run batch processing
result = ridge_batch(X, Y, batch_size=batch_size, n_rand=1000)

# Or stream to disk
ridge_batch(X, Y, batch_size=batch_size, output_path="results.h5ad")
```

### I/O Functions

```python
from secactpy import (
    load_results, save_results, 
    results_to_anndata, load_as_anndata,
    get_file_info, concatenate_results
)

# Load results
results = load_results("output.h5ad")

# Lazy loading for large files
results = load_results("output.h5ad", load_arrays=False)
batch = results['beta'][:, :1000]  # Load only what you need

# File information
info = get_file_info("output.h5ad")
print(f"Shape: {info['shape']}, Size: {info['file_size_mb']:.1f} MB")

# Concatenate multiple result files
concatenated = concatenate_results(["part1.h5", "part2.h5", "part3.h5"])
```

### Low-Level Ridge Functions

```python
from secactpy import ridge, compute_projection_matrix

# Direct ridge regression
result = ridge(X, Y, lambda_=5e5, n_rand=1000, seed=0)

# Precompute projection matrix for repeated use
T = compute_projection_matrix(X, lambda_=5e5)
```

## Parameters

### `secact_activity()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expression` | - | Gene expression DataFrame (genes × samples) |
| `signature` | - | Signature matrix DataFrame (genes × features) |
| `lambda_` | `5e5` | Ridge regularization parameter |
| `n_rand` | `1000` | Number of permutations (0 for t-test) |
| `seed` | `0` | Random seed for reproducibility |
| `scale` | `'zscore'` | Scaling method: 'zscore', 'center', 'none' |
| `backend` | `'auto'` | Computation backend: 'auto', 'numpy', 'cupy' |

### `ridge_batch()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `X` | - | Signature matrix (n_genes × n_features) |
| `Y` | - | Expression matrix (n_genes × n_samples) |
| `batch_size` | `5000` | Samples per batch |
| `output_path` | `None` | Stream to file instead of memory |
| `progress_callback` | `None` | Function for progress tracking |

## Reproducibility

SecActPy produces **identical results** to the R SecAct package when using the same parameters:

```python
# For exact SecAct compatibility, use these defaults:
result = secact_activity(
    expression, signature,
    lambda_=5e5,
    n_rand=1000,
    seed=0,
    scale='zscore'
)
```

The package uses a GSL-compatible MT19937 random number generator to ensure cross-platform reproducibility.

## GPU Acceleration

For large datasets, enable GPU acceleration with CuPy:

```bash
# Install CuPy (choose version matching your CUDA)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

```python
from secactpy import secact_activity, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")

# GPU will be used automatically when available
result = secact_activity(expression, signature, backend='auto')

# Force GPU usage
result = secact_activity(expression, signature, backend='cupy')
```

## File Formats

### Input

- **Expression data**: CSV, TSV, or any pandas-readable format
  - Rows: genes (gene symbols)
  - Columns: samples

- **Signature matrices**: Bundled as TSV.GZ
  - Rows: genes
  - Columns: proteins/cytokines

### Output

- **HDF5/H5AD**: Native format for streaming output
- **CSV**: For export to other tools
- **AnnData**: For scanpy integration

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7

### Optional

- CuPy ≥ 10.0 (GPU acceleration)
- h5py ≥ 3.0 (HDF5 I/O)
- anndata ≥ 0.8 (scanpy integration)

## Citation

If you use SecActPy in your research, please cite:

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication.[Link](https://github.com/data2intelligence/SecAct)


## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
