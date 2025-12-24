# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SecActPy is a Python package for inferring secreted protein (e.g. cytokine/chemokine) activity from gene expression data using ridge regression with permutation-based significance testing.

**Key Features:**
- 🎯 **SecAct Compatible**: Produces identical results to the R SecAct/RidgeR package
- 🚀 **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- 📊 **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- 🔬 **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- 🧬 **Multi-Platform Support**: Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics (Visium, CosMx)
- 💾 **Smart Caching**: Permutation tables cached to disk for faster repeated analyses
- 🧮 **Sparse-Preserving**: Memory-efficient processing for sparse single-cell data

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

# With scRNA-seq support (scanpy)
pip install "secactpy[scrnaseq] @ git+https://github.com/psychemistz/SecActPy.git"

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

### Basic Usage (Bulk RNA-seq)

```python
import pandas as pd
from secactpy import secact_activity_inference, load_signature

# Load your differential expression data (genes × samples)
diff_expr = pd.read_csv("diff_expression.csv", index_col=0)

# Run inference with built-in SecAct signature
result = secact_activity_inference(
    diff_expr,
    is_differential=True,
    sig_matrix="secact",  # or "cytosig"
    verbose=True
)

# Access results (all are DataFrames with proper labels)
activity = result['zscore']    # Activity z-scores
pvalues = result['pvalue']     # P-values
coefficients = result['beta']  # Regression coefficients

# Find significant activities
significant = result['pvalue'] < 0.05
print(f"Significant: {significant.sum().sum()} / {significant.size}")
```

### Spatial Transcriptomics (10X Visium)

```python
from secactpy import secact_activity_inference_st

# From 10X Visium folder
result = secact_activity_inference_st(
    "path/to/visium_folder/",
    min_genes=1000,
    scale_factor=1e5,
    sig_matrix="secact",
    verbose=True
)

# Access spot-level activity
activity = result['zscore']  # (proteins × spots)
```

### Spatial Transcriptomics (CosMx / Single-Cell Resolution)

```python
import anndata as ad
from secactpy import secact_activity_inference_st

# Load CosMx h5ad file
adata = ad.read_h5ad("cosmx_data.h5ad")

# Run inference on single-cell resolution ST data
result = secact_activity_inference_st(
    adata,
    scale_factor=1000,  # Lower for CosMx
    sig_matrix="secact",
    sig_filter=True,    # Filter signatures to available genes
    verbose=True
)

activity = result['zscore']  # (proteins × cells)
```

### scRNA-seq Analysis

```python
import anndata as ad
from secactpy import secact_activity_inference_scrnaseq

# Load AnnData
adata = ad.read_h5ad("scrnaseq_data.h5ad")

# Pseudo-bulk analysis by cell type
result = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=False,  # Aggregate by cell type
    verbose=True
)

# Single-cell level analysis
result_sc = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=True,   # Per-cell inference
    verbose=True
)
```

### Large-Scale Analysis with Batch Processing

```python
from secactpy import (
    ridge_batch, 
    estimate_batch_size,
    precompute_population_stats,
    precompute_projection_components,
    ridge_batch_sparse_preserving
)

# Estimate optimal batch size
batch_size = estimate_batch_size(
    n_genes=20000, 
    n_features=50, 
    available_gb=8.0
)

# Standard batch processing
result = ridge_batch(
    X, Y,
    batch_size=batch_size,
    n_rand=1000,
    backend='cupy',  # Use GPU
    verbose=True
)

# Sparse-preserving batch processing (for sparse scRNA-seq/ST data)
# Keeps Y sparse throughout - critical for million-cell datasets
stats = precompute_population_stats(Y_sparse)
proj = precompute_projection_components(X, lambda_=5e5)

result = ridge_batch_sparse_preserving(
    proj, Y_sparse, stats,
    n_rand=1000,
    use_cache=True,  # Cache permutation tables
    verbose=True
)
```

### Permutation Table Caching

SecActPy caches permutation tables to disk for faster repeated analyses:

```python
from secactpy import (
    get_cached_inverse_perm_table,
    list_cached_tables,
    clear_perm_cache,
    DEFAULT_CACHE_DIR
)

# Tables are automatically cached during inference
# Check cached tables
print(list_cached_tables())
# {'files': ['inv_perm_n7919_nperm1000_seed0.npy'], 'total_size_mb': 60.5}

# Clear cache if needed
clear_perm_cache()

# Default cache location
print(DEFAULT_CACHE_DIR)  # ~/.cache/ridgesig_perm_tables
```

## API Reference

### High-Level Functions

| Function | Description |
|----------|-------------|
| `secact_activity_inference()` | Bulk RNA-seq inference |
| `secact_activity_inference_st()` | Spatial transcriptomics inference |
| `secact_activity_inference_scrnaseq()` | scRNA-seq inference |
| `load_signature(name='secact')` | Load built-in signature matrix |
| `load_visium_10x()` | Load 10X Visium data |

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
    save_st_results_to_h5ad, add_activity_to_anndata
)

# Save results to HDF5
save_results(result, "output.h5")

# Load results
results = load_results("output.h5")

# Save ST results to h5ad
save_st_results_to_h5ad(result, "st_activity.h5ad", adata)

# Add activity to existing AnnData
adata = add_activity_to_anndata(adata, result)
```

### Low-Level Ridge Functions

```python
from secactpy import ridge, compute_projection_matrix, ridge_with_precomputed_T

# Direct ridge regression
result = ridge(X, Y, lambda_=5e5, n_rand=1000, seed=0, backend='numpy')

# Precompute projection matrix for repeated use
T = compute_projection_matrix(X, lambda_=5e5)
result = ridge_with_precomputed_T(T, Y, n_rand=1000, seed=0)
```

## Parameters

### `secact_activity_inference()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_profile` | - | Expression DataFrame or path to file |
| `is_differential` | `False` | Whether input is already differential expression |
| `sig_matrix` | `"secact"` | Signature: "secact", "cytosig", or DataFrame |
| `lambda_` | `5e5` | Ridge regularization parameter |
| `n_rand` | `1000` | Number of permutations |
| `seed` | `0` | Random seed for reproducibility |
| `backend` | `'auto'` | Computation backend: 'auto', 'numpy', 'cupy' |

### `secact_activity_inference_st()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_data` | - | Path to Visium folder, DataFrame, or AnnData |
| `scale_factor` | `1e5` | Normalization scale factor |
| `sig_matrix` | `"secact"` | Signature matrix |
| `min_genes` | `0` | Minimum genes per spot |
| `sig_filter` | `False` | Filter signatures to available genes |
| `backend` | `'auto'` | Computation backend |

### `secact_activity_inference_scrnaseq()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_data` | - | AnnData object or path to h5ad file |
| `cell_type_col` | - | Column in obs for cell type annotation |
| `is_single_cell_level` | `False` | Per-cell or pseudo-bulk analysis |
| `sig_matrix` | `"secact"` | Signature matrix |
| `backend` | `'auto'` | Computation backend |

### `ridge_batch()`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `X` | - | Signature matrix (n_genes × n_features) |
| `Y` | - | Expression matrix (n_genes × n_samples) |
| `batch_size` | `5000` | Samples per batch |
| `output_path` | `None` | Stream to file instead of memory |
| `backend` | `'numpy'` | 'numpy' or 'cupy' |

## Reproducibility

SecActPy produces **identical results** to the R SecAct/RidgeR package when using the same parameters:

```python
# For exact RidgeR compatibility, use these defaults:
result = secact_activity_inference(
    expression,
    is_differential=True,
    sig_matrix="secact",
    lambda_=5e5,
    n_rand=1000,
    seed=0
)
```

### GSL Random Number Generator Compatibility

SecActPy implements a GSL-compatible MT19937 (Mersenne Twister) random number generator to ensure cross-platform reproducibility with R/RidgeR.

**Important Implementation Note:** GSL treats `seed=0` specially by using `4357` as the actual seed value. SecActPy replicates this behavior exactly:

```python
# When you specify seed=0 (the default):
result = secact_activity_inference(expression, seed=0)

# Internally, GSL (and SecActPy) actually uses seed=4357
# This ensures identical permutation sequences between R and Python
```

### Validation

The permutation sequences have been validated to match exactly:

| Test | R (GSL) | Python (SecActPy) |
|------|---------|-------------------|
| MT19937 raw value [0] | 4293858116 | 4293858116 ✓ |
| uniform_int(10) first 10 | 9,1,2,9,2,4,9,7,5,7 | 9,1,2,9,2,4,9,7,5,7 ✓ |
| Fisher-Yates shuffle [0..9] | 9,2,4,0,5,7,3,6,1,8 | 9,2,4,0,5,7,3,6,1,8 ✓ |

All output arrays (beta, se, zscore, pvalue) match R output within numerical precision (~1e-15).

## GPU Acceleration

For large datasets, enable GPU acceleration with CuPy:

```bash
# Install CuPy (choose version matching your CUDA)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

```python
from secactpy import secact_activity_inference, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")

# GPU will be used automatically when available
result = secact_activity_inference(expression, backend='auto')

# Force GPU usage
result = secact_activity_inference(expression, backend='cupy')
```

### GPU Performance

| Dataset | CPU (NumPy) | GPU (CuPy) | Speedup |
|---------|-------------|------------|---------|
| Bulk (1k samples) | 1.5s | 0.3s | 5x |
| scRNA-seq (5k cells) | 6.4s | 1.2s | 5.3x |
| ST (10k spots) | 13.9s | 2.5s | 5.6x |
| CosMx (100k cells) | 120s | 18s | 6.7x |

## File Formats

### Input

- **Bulk RNA-seq**: CSV, TSV, or any pandas-readable format
  - Rows: genes (gene symbols)
  - Columns: samples

- **Spatial Transcriptomics**: 
  - 10X Visium folder (with `filtered_feature_bc_matrix/`)
  - AnnData (.h5ad) for CosMx and other platforms

- **scRNA-seq**: AnnData (.h5ad)

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
- anndata ≥ 0.8 (AnnData support)
- scanpy ≥ 1.9 (scRNA-seq analysis)

## Citation

If you use SecActPy in your research, please cite:

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. [[Link](https://github.com/data2intelligence/SecAct)]

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v0.1.0
- Initial release
- Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics support
- GPU acceleration with CuPy
- Batch processing for million-sample datasets
- Sparse-preserving batch processing
- Permutation table caching
- GSL-compatible RNG for R/RidgeR reproducibility
