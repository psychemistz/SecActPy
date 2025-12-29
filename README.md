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
- 💾 **Smart Caching**: Optional permutation table caching for faster repeated analyses
- 🧮 **Sparse-Preserving**: Memory-efficient processing for sparse single-cell data

## Installation

### CPU Only

```bash
pip install git+https://github.com/psychemistz/SecActPy.git
```

### With GPU Support (CUDA 11.x)

```bash
pip install "secactpy[gpu] @ git+https://github.com/psychemistz/SecActPy.git"
```

> **Note**: For CUDA 12.x, install CuPy separately: `pip install cupy-cuda12x`

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
from secactpy import secact_activity_inference

# Load your differential expression data (genes × samples)
diff_expr = pd.read_csv("diff_expression.csv", index_col=0)

# Run inference
result = secact_activity_inference(
    diff_expr,
    is_differential=True,
    sig_matrix="secact",  # or "cytosig"
    verbose=True
)

# Access results
activity = result['zscore']    # Activity z-scores
pvalues = result['pvalue']     # P-values
coefficients = result['beta']  # Regression coefficients
```

### Spatial Transcriptomics (10X Visium)

```python
from secactpy import secact_activity_inference_st

# Spot-level analysis
result = secact_activity_inference_st(
    "path/to/visium_folder/",
    min_genes=1000,
    verbose=True
)

activity = result['zscore']  # (proteins × spots)
```

### Spatial Transcriptomics with Cell Type Resolution

```python
import anndata as ad
from secactpy import secact_activity_inference_st

# Load annotated spatial data
adata = ad.read_h5ad("spatial_annotated.h5ad")

# Cell-type resolution (pseudo-bulk by cell type)
result = secact_activity_inference_st(
    adata,
    cell_type_col="cell_type",  # Column in adata.obs
    is_spot_level=False,        # Aggregate by cell type
    verbose=True
)

activity = result['zscore']  # (proteins × cell_types)
```

### scRNA-seq Analysis

```python
import anndata as ad
from secactpy import secact_activity_inference_scrnaseq

adata = ad.read_h5ad("scrnaseq_data.h5ad")

# Pseudo-bulk by cell type
result = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=False,
    verbose=True
)

# Single-cell level
result_sc = secact_activity_inference_scrnaseq(
    adata,
    cell_type_col="cell_type",
    is_single_cell_level=True,
    verbose=True
)
```

### Large-Scale Batch Processing

```python
from secactpy import (
    ridge_batch,
    precompute_population_stats,
    precompute_projection_components,
    ridge_batch_sparse_preserving
)

# Standard batch processing
result = ridge_batch(
    X, Y,
    batch_size=5000,
    n_rand=1000,
    backend='cupy',  # Use GPU
    verbose=True
)

# Sparse-preserving for million-cell datasets
stats = precompute_population_stats(Y_sparse)
proj = precompute_projection_components(X, lambda_=5e5)

result = ridge_batch_sparse_preserving(
    proj, Y_sparse, stats,
    n_rand=1000,
    use_cache=True,
    verbose=True
)
```

## API Reference

### High-Level Functions

| Function | Description |
|----------|-------------|
| `secact_activity_inference()` | Bulk RNA-seq inference |
| `secact_activity_inference_st()` | Spatial transcriptomics inference |
| `secact_activity_inference_scrnaseq()` | scRNA-seq inference |
| `load_signature(name='secact')` | Load built-in signature matrix |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_matrix` | `"secact"` | Signature: "secact", "cytosig", or DataFrame |
| `lambda_` | `5e5` | Ridge regularization parameter |
| `n_rand` | `1000` | Number of permutations |
| `seed` | `0` | Random seed for reproducibility |
| `backend` | `'auto'` | 'auto', 'numpy', or 'cupy' |
| `use_cache` | `False` | Cache permutation tables to disk |

### ST-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_type_col` | `None` | Column in AnnData.obs for cell type |
| `is_spot_level` | `True` | If False, aggregate by cell type |
| `scale_factor` | `1e5` | Normalization scale factor |

## GPU Acceleration

```python
from secactpy import secact_activity_inference, CUPY_AVAILABLE

print(f"GPU available: {CUPY_AVAILABLE}")

# Auto-detect GPU
result = secact_activity_inference(expression, backend='auto')

# Force GPU
result = secact_activity_inference(expression, backend='cupy')
```

### Performance

| Dataset | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| Bulk (1k samples) | 1.5s | 0.3s | 5x |
| scRNA-seq (5k cells) | 6.4s | 1.2s | 5.3x |
| ST (10k spots) | 13.9s | 2.5s | 5.6x |
| CosMx (100k cells) | 120s | 18s | 6.7x |

## Reproducibility

SecActPy produces **identical results** to R SecAct/RidgeR:

```python
result = secact_activity_inference(
    expression,
    is_differential=True,
    sig_matrix="secact",
    lambda_=5e5,
    n_rand=1000,
    seed=0,
    use_gsl_rng=True  # Default: R-compatible RNG
)
```

For faster inference when R compatibility is not needed:

```python
result = secact_activity_inference(
    expression,
    use_gsl_rng=False,  # ~70x faster permutation generation
)
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- Pandas ≥ 1.3
- SciPy ≥ 1.7
- h5py ≥ 3.0
- anndata ≥ 0.8
- scanpy ≥ 1.9

**Optional:** CuPy ≥ 10.0 (GPU acceleration)

## Citation

If you use SecActPy in your research, please cite:

Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. Inference of secreted protein activities in intercellular communication. [[Link](https://github.com/data2intelligence/SecAct)]

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

### v0.1.1
- Added `use_cache` parameter to all inference functions (default: `False`)
- Added cell type resolution for spatial transcriptomics (`cell_type_col`, `is_spot_level`)
- Simplified installation (base includes all common dependencies)

### v0.1.0
- Initial release with bulk, scRNA-seq, and ST support
- GPU acceleration, batch processing, sparse-preserving mode
- GSL-compatible RNG for R/RidgeR reproducibility
