# SecActPy

**Secreted Protein Activity Inference using Ridge Regression**

[![PyPI version](https://badge.fury.io/py/secactpy.svg)](https://pypi.org/project/secactpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml/badge.svg)](https://github.com/data2intelligence/SecActPy/actions/workflows/tests.yml)
[![Docker](https://img.shields.io/docker/pulls/psychemistz/secactpy)](https://hub.docker.com/r/psychemistz/secactpy)

Python implementation of [SecAct](https://github.com/data2intelligence/SecAct) for inferring secreted protein activities from gene expression data.

**Key Features:**
- 🎯 **SecAct Compatible**: Produces identical results to the R SecAct/RidgeR package
- 🚀 **GPU Acceleration**: Optional CuPy backend for large-scale analysis
- 📊 **Million-Sample Scale**: Batch processing with streaming output for massive datasets
- 🔬 **Built-in Signatures**: Includes SecAct and CytoSig signature matrices
- 🧬 **Multi-Platform Support**: Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics (Visium, CosMx)
- 💾 **Smart Caching**: Optional permutation table caching for faster repeated analyses
- 🧮 **Sparse-Aware**: Automatic memory-efficient processing for sparse single-cell data
- 📈 **Visualization** *(v0.2.0)*: Spatial plots, heatmaps, and cell-cell communication visualizations
- 🗺️ **Spatial Analysis** *(v0.2.0)*: Colocalization, interface detection, and L-R network scoring
- 📉 **Survival Analysis** *(v0.2.0)*: Cox regression and Kaplan-Meier integration

## Installation

### From PyPI (Recommended)

```bash
# Core package
pip install secactpy

# With plotting support
pip install "secactpy[plotting]"

# With survival analysis
pip install "secactpy[survival]"

# With all optional features
pip install "secactpy[all]"
```

### From GitHub

```bash
# CPU Only
pip install git+https://github.com/data2intelligence/SecActPy.git

# With GPU Support (CUDA 11.x)
pip install "secactpy[gpu] @ git+https://github.com/data2intelligence/SecActPy.git"
```

> **Note**: For CUDA 12.x, install CuPy separately: `pip install cupy-cuda12x`

### Development Installation

```bash
git clone https://github.com/data2intelligence/SecActPy.git
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
from secactpy import ridge_batch

# Dense data (pre-scaled)
Y_scaled = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
result = ridge_batch(
    X, Y_scaled,
    batch_size=5000,
    n_rand=1000,
    backend='cupy',  # Use GPU
    verbose=True
)

# Sparse data (auto-scaled internally)
import scipy.sparse as sp
Y_sparse = sp.csr_matrix(counts)  # Raw counts
result = ridge_batch(
    X, Y_sparse,
    batch_size=10000,
    n_rand=1000,
    backend='auto',
    verbose=True
)

# Stream results to disk for very large datasets
ridge_batch(
    X, Y,
    batch_size=10000,
    output_path="results.h5ad",
    output_compression="gzip",
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

### Core Functions

| Function | Description |
|----------|-------------|
| `ridge()` | Single-call ridge regression with permutation testing |
| `ridge_batch()` | Batch processing for large datasets (dense or sparse) |
| `estimate_batch_size()` | Estimate optimal batch size for available memory |
| `estimate_memory()` | Estimate memory requirements |

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

### Batch Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `5000` | Samples per batch |
| `output_path` | `None` | Stream results to H5AD file |
| `output_compression` | `"gzip"` | Compression: "gzip", "lzf", or None |

## Spatial Analysis (v0.2.0)

SecActPy v0.2.0 introduces comprehensive spatial transcriptomics analysis tools.

### Cell-Cell Colocalization

```python
from secactpy.spatial import calc_colocalization, calc_neighborhood_enrichment

# Pairwise colocalization with permutation testing
coloc = calc_colocalization(
    cell_types, coords, radius=100,
    method="jaccard", n_permutations=1000
)
print(coloc['score'])   # Colocalization matrix
print(coloc['pvalue'])  # Significance

# Neighborhood enrichment (squidpy-style)
enrichment = calc_neighborhood_enrichment(cell_types, coords, n_neighbors=10)
```

### Spatial Statistics

```python
from secactpy.spatial import calc_morans_i, calc_getis_ord_g, calc_ripley_k

# Moran's I spatial autocorrelation
morans = calc_morans_i(gene_expr, coords, radius=100, n_permutations=999)
print(f"Moran's I: {morans['I']:.3f}, p={morans['pvalue']:.4f}")

# Hot spot detection (Getis-Ord G*)
g_star = calc_getis_ord_g(gene_expr, coords, radius=100)
hot_spots = g_star > 1.96  # p < 0.05

# Ripley's K for clustering analysis
radii = np.linspace(10, 200, 20)
ripley = calc_ripley_k(cell_types, coords, radii, cell_type="Tumor")
```

### Tumor-Stroma Interface Detection

```python
from secactpy.spatial import detect_interface, analyze_interface_activity

# Detect interface regions
interface = detect_interface(
    cell_types, coords,
    tumor_types=["Tumor", "Malignant"],
    stroma_types=["Fibroblast", "Endothelial"],
    radius=100
)

# Analyze activity at interface
results = analyze_interface_activity(activity, interface)
significant = results[results['padj'] < 0.05]
```

### Ligand-Receptor Network Scoring

```python
from secactpy.spatial import load_lr_database, score_lr_interactions

# Load L-R database
lr_pairs = load_lr_database()  # Built-in CellChat-style pairs

# Score interactions between cell types
lr_scores = score_lr_interactions(
    expression, coords, cell_types, lr_pairs,
    radius=100
)

# Get significant interactions
from secactpy.spatial import identify_significant_interactions
significant = identify_significant_interactions(lr_scores, pvalue_threshold=0.05)
```

### Spatial Analysis Functions

| Function | Description |
|----------|-------------|
| `calc_colocalization()` | Pairwise cell-type colocalization |
| `calc_neighborhood_enrichment()` | Neighborhood enrichment analysis |
| `calc_morans_i()` | Moran's I spatial autocorrelation |
| `calc_getis_ord_g()` | Getis-Ord G* hot spot detection |
| `calc_ripley_k()` | Ripley's K function |
| `detect_interface()` | Tumor-stroma boundary detection |
| `analyze_interface_activity()` | Differential activity at interface |
| `load_lr_database()` | Load L-R interaction database |
| `score_lr_interactions()` | Cell-type L-R interaction scoring |
| `score_lr_spatial()` | Spot-level L-R scoring |

## Visualization (v0.2.0)

### Spatial Plots

```python
from secactpy.plotting import plot_spatial_feature, plot_spatial_categorical

# Activity on spatial coordinates
plot_spatial_feature(
    coords, activity.loc['IL6'],
    cmap='RdBu_r', center_zero=True,
    title="IL6 Activity"
)

# Cell type visualization
plot_spatial_categorical(coords, cell_types, palette='tab20')
```

### Activity Heatmaps

```python
from secactpy.plotting import plot_activity_heatmap, plot_activity_bar

# Clustered heatmap
g = plot_activity_heatmap(result['zscore'], title="Secreted Protein Activity")

# Top activities bar plot
plot_activity_bar(result['zscore']['Sample1'], n_top=20, n_bottom=10)
```

### Cell-Cell Communication Plots

```python
from secactpy.plotting import plot_ccc_heatmap, plot_ccc_circle

# Sender-receiver heatmap
plot_ccc_heatmap(interaction_scores, title="IL6 Signaling")

# Circular network diagram
plot_ccc_circle(interaction_matrix)
```

### Survival Analysis Plots

```python
from secactpy.plotting import plot_kaplan_meier, plot_survival_by_activity

# Kaplan-Meier curves
plot_kaplan_meier(time, event, groups=treatment_group)

# Survival stratified by activity
plot_survival_by_activity(
    activity.loc['TNF'], time, event,
    method='optimal'  # Find optimal cutoff
)
```

### Plotting Functions

| Function | Description |
|----------|-------------|
| `plot_spatial_feature()` | Feature values on coordinates |
| `plot_spatial_categorical()` | Cell type labels on coordinates |
| `plot_spatial_multi()` | Grid of multiple features |
| `plot_activity_heatmap()` | Clustered activity heatmap |
| `plot_activity_bar()` | Top/bottom activities bar plot |
| `plot_ccc_heatmap()` | Cell-cell communication heatmap |
| `plot_ccc_circle()` | Circular network diagram |
| `plot_ccc_dotplot()` | Dot plot with size/color encoding |
| `plot_kaplan_meier()` | Kaplan-Meier survival curves |
| `plot_survival_by_activity()` | Survival by activity level |
| `plot_forest()` | Forest plot of Cox results |

## Utility Functions (v0.2.0)

### Sparse Matrix Operations

```python
from secactpy.utils import sweep_sparse, normalize_sparse

# R-style sweep for sparse matrices
result = sweep_sparse(X, row_values, margin=0, operation='subtract')

# Normalize sparse matrix (L1, L2, sum)
X_norm = normalize_sparse(X, method='l1', axis=1)
```

### Gene Symbol Utilities

```python
from secactpy.utils import rm_duplicates, match_genes

# Remove duplicate genes (keep highest sum)
expr_clean = rm_duplicates(expression)

# Case-insensitive gene matching
matched = match_genes(query_genes, reference_genes)
```

### Survival Analysis

```python
from secactpy.utils import survival_analysis, coxph_best_separation

# Batch survival analysis for all proteins
results = survival_analysis(activity, time, event)
significant = results[results['p'] < 0.05]

# Cox PH with optimal threshold finding
result = coxph_best_separation(activity.loc['IL6'], time, event)
```

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

| Dataset | R (Mac M1) | R (Linux) | Py (CPU) | Py (GPU) | Speedup |
|---------|------------|-----------|----------|----------|---------|
| Bulk (1,170 sp × 1,000 samples) | 74.4s | 141.6s | 128.8s | 6.7s | 11–19x |
| scRNA-seq (1,170 sp × 788 cells) | 54.9s | 117.4s | 104.8s | 6.8s | 8–15x |
| Visium (1,170 sp × 3,404 spots) | 141.7s | 379.8s | 381.4s | 11.2s | 13–34x |
| CosMx (151 sp × 443,515 cells) | 936.9s | 976.1s | 1226.7s | 99.9s | 9–12x |

<details>
<summary>Benchmark Environment</summary>

- **Mac CPU**: M1 Pro with VECLIB (8 cores)
- **Linux CPU**: AMD EPYC 7543P (4 cores)
- **Linux GPU**: NVIDIA A100-SXM4-80GB

</details>

## Command Line Interface

SecActPy provides a command line interface for common workflows:

```bash
# Bulk RNA-seq (differential expression)
secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v

# Bulk RNA-seq (raw counts)
secactpy bulk -i counts.tsv -o results.h5ad -v

# scRNA-seq with cell type aggregation
secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v

# scRNA-seq at single cell level
secactpy scrnaseq -i data.h5ad -o results.h5ad --single-cell -v

# Visium spatial transcriptomics
secactpy visium -i /path/to/visium/ -o results.h5ad -v

# CosMx (single-cell spatial)
secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v

# Use GPU acceleration
secactpy bulk -i data.tsv -o results.h5ad --backend cupy -v

# Use CytoSig signature
secactpy bulk -i data.tsv -o results.h5ad --signature cytosig -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file or directory |
| `-o, --output` | Output H5AD file |
| `-s, --signature` | Signature matrix (secact, cytosig) |
| `--lambda` | Ridge regularization (default: 5e5) |
| `-n, --n-rand` | Number of permutations (default: 1000) |
| `--backend` | Computation backend (auto, numpy, cupy) |
| `--batch-size` | Batch size for large datasets |
| `-v, --verbose` | Verbose output |

## Docker

Pre-built Docker images are available:

```bash
# CPU version
docker pull psychemistz/secactpy:latest

# GPU version
docker pull psychemistz/secactpy:gpu

# With R SecAct/RidgeR for cross-validation
docker pull psychemistz/secactpy:with-r
```

See [DOCKER.md](DOCKER.md) for detailed usage instructions.

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

For faster analysis (when R compatibility is not required):

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

**Optional:**
- CuPy ≥ 10.0 (GPU acceleration)
- matplotlib ≥ 3.5, seaborn ≥ 0.12 (plotting)
- lifelines ≥ 0.27 (survival analysis)

## Citation

If you use SecActPy in your research, please cite:

> Beibei Ru, Lanqi Gong, Emily Yang, Seongyong Park, George Zaki, Kenneth Aldape, Lalage Wakefield, Peng Jiang. 
> **Inference of secreted protein activities in intercellular communication.**
> [GitHub: data2intelligence/SecAct](https://github.com/data2intelligence/SecAct)

## Related Projects

- [SecAct](https://github.com/data2intelligence/SecAct) - Original R implementation
- [RidgeR](https://github.com/beibeiru/RidgeR) - R ridge regression package
- [SpaCET](https://github.com/data2intelligence/SpaCET) - Spatial transcriptomics cell type analysis
- [CytoSig](https://github.com/data2intelligence/CytoSig) - Cytokine signaling inference

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

### v0.2.0 (Current Release)
**New Modules:**
- **`secactpy.spatial`**: Spatial transcriptomics analysis
  - Cell-cell colocalization (`calc_colocalization`, `calc_neighborhood_enrichment`)
  - Spatial statistics (`calc_morans_i`, `calc_getis_ord_g`, `calc_ripley_k`)
  - Tumor-stroma interface detection (`detect_interface`, `analyze_interface_activity`)
  - Ligand-receptor network scoring (`score_lr_interactions`, `load_lr_database`)
- **`secactpy.plotting`**: Visualization functions
  - Spatial plots (`plot_spatial_feature`, `plot_spatial_categorical`)
  - Activity heatmaps (`plot_activity_heatmap`, `plot_activity_bar`)
  - Cell-cell communication plots (`plot_ccc_heatmap`, `plot_ccc_circle`)
  - Survival plots (`plot_kaplan_meier`, `plot_survival_by_activity`)
- **`secactpy.utils`**: Utility functions
  - Sparse matrix operations (`sweep_sparse`, `normalize_sparse`)
  - Gene symbol utilities (`rm_duplicates`, `match_genes`)
  - Survival analysis (`survival_analysis`, `coxph_best_separation`)

**Infrastructure:**
- Official release under data2intelligence 
- PyPI package available (`pip install secactpy`)
- Comprehensive test suite and CI/CD pipeline
- Docker images with GPU and R support

### v0.1.2 (Initial Development)
- Ridge regression with permutation-based significance testing
- GPU acceleration via CuPy backend (9–34x speedup)
- Batch processing with streaming H5AD output for million-sample datasets
- Automatic sparse matrix handling in `ridge_batch()`
- Built-in SecAct and CytoSig signature matrices
- GSL-compatible RNG for R/RidgeR reproducibility
- Support for Bulk RNA-seq, scRNA-seq, and Spatial Transcriptomics
- Cell type resolution for ST data (`cell_type_col`, `is_spot_level`)
- Optional permutation table caching (`use_cache`)
