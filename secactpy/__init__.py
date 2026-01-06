"""
SecActPy: Secreted Protein Activity Inference

A Python package for inferring secreted protein activity from
gene expression data using ridge regression with permutation testing.

Compatible with R's SecAct/RidgeR package - produces identical results.

Quick Start (Bulk RNA-seq):
---------------------------
    >>> from secactpy import secact_activity_inference
    >>>
    >>> # From file path (auto-detect format: CSV, TSV, TXT)
    >>> result = secact_activity_inference(
    ...     "diff_expression.csv",  # or .tsv, .txt
    ...     is_differential=True,
    ...     verbose=True
    ... )
    >>>
    >>> # If genes are in first column (not row names)
    >>> result = secact_activity_inference(
    ...     "data.csv",
    ...     gene_col=0,  # genes in first column
    ...     is_differential=True
    ... )
    >>>
    >>> # Or from DataFrame
    >>> import pandas as pd
    >>> diff_expr = pd.read_csv("diff_expression.csv", index_col=0)
    >>> result = secact_activity_inference(diff_expr, is_differential=True)
    >>>
    >>> # Access results
    >>> activity = result['zscore']    # Activity z-scores
    >>> pvalues = result['pvalue']     # Significance

Flexible Data Loading:
----------------------
    >>> from secactpy import load_expression_data
    >>>
    >>> # Auto-detect format
    >>> expr = load_expression_data("data.csv")
    >>> expr = load_expression_data("data.tsv")
    >>> expr = load_expression_data("data.txt")
    >>>
    >>> # Genes in first column (not row names)
    >>> expr = load_expression_data("data.csv", gene_col=0)

scRNA-seq Analysis:
-------------------
    >>> import anndata as ad
    >>> from secactpy import secact_activity_inference_scrnaseq
    >>>
    >>> # Load AnnData (h5ad file)
    >>> adata = ad.read_h5ad("scrnaseq_data.h5ad")
    >>>
    >>> # Run pseudo-bulk analysis by cell type
    >>> result = secact_activity_inference_scrnaseq(
    ...     adata,
    ...     cell_type_col="cell_type",
    ...     is_single_cell_level=False,  # Aggregate by cell type
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × cell_types)

Spatial Transcriptomics:
------------------------
    >>> from secactpy import secact_activity_inference_st, load_visium_10x
    >>>
    >>> # Load 10X Visium data
    >>> result = secact_activity_inference_st(
    ...     "path/to/visium_folder/",
    ...     min_genes=1000,
    ...     scale_factor=1e5,
    ...     verbose=True
    ... )
    >>> activity = result['zscore']  # (proteins × spots)

For large datasets (>100k samples):
-----------------------------------
    >>> from secactpy import ridge_batch, estimate_batch_size
    >>>
    >>> # Estimate optimal batch size
    >>> batch_size = estimate_batch_size(n_genes=20000, n_features=50)
    >>>
    >>> # Run batch processing with streaming output
    >>> ridge_batch(X, Y, batch_size=batch_size, output_path="results.h5ad")
    >>>
    >>> # Load results
    >>> from secactpy import load_results, results_to_anndata
    >>> results = load_results("results.h5ad")
    >>> adata = results_to_anndata(results)  # For scanpy integration

Utility Functions (NEW in v0.2.0):
----------------------------------
    >>> from secactpy.utils import sweep_sparse, rm_duplicates, normalize_sparse
    >>> from secactpy.spatial import calc_spatial_weights, spatial_lag
    >>>
    >>> # Sparse matrix operations
    >>> m_centered = sweep_sparse(m, margin=0, stats=row_means, fun="subtract")
    >>>
    >>> # Remove duplicate genes (keep highest sum)
    >>> expr_clean = rm_duplicates(expr_df)
    >>>
    >>> # Spatial weight calculation for ST data
    >>> W = calc_spatial_weights(coords, radius=100, sigma=50)
    >>> smoothed = spatial_lag(values, W)
"""

__version__ = "0.2.0"

# Batch processing for large datasets
from .batch import (
    StreamingResultWriter,
    estimate_batch_size,
    estimate_memory,
    ridge_batch,
)

# High-level API (most users need only these)
from .inference import (
    compute_differential,
    expand_rows,
    group_signatures,
    load_expression_data,
    load_visium_10x,
    prepare_data,
    scale_columns,
    secact_activity,
    secact_activity_inference,
    secact_activity_inference_scrnaseq,
    secact_activity_inference_st,
)

# I/O utilities
from .io import (
    ANNDATA_AVAILABLE,
    H5PY_AVAILABLE,
    add_activity_to_anndata,
    concatenate_results,
    get_file_info,
    load_as_anndata,
    load_results,
    results_to_anndata,
    results_to_dataframes,
    save_results,
    save_st_results_to_h5ad,
)

# Lower-level ridge functions (for advanced users)
from .ridge import (
    CUPY_AVAILABLE,
    CUPY_INIT_ERROR,
    DEFAULT_LAMBDA,
    DEFAULT_NRAND,
    DEFAULT_SEED,
    compute_projection_matrix,
    ridge,
    ridge_with_precomputed_T,
)

# RNG (for reproducibility testing)
from .rng import (
    DEFAULT_CACHE_DIR,
    GSLRNG,
    clear_perm_cache,
    generate_inverse_permutation_table,
    generate_inverse_permutation_table_fast,
    generate_permutation_table,
    generate_permutation_table_fast,
    get_cached_inverse_perm_table,
    get_cached_perm_table,
    list_cached_tables,
)

# Signature loading
from .signature import (
    AVAILABLE_SIGNATURES,
    get_signature_info,
    list_signatures,
    load_cytosig,
    load_secact,
    load_signature,
)

# ============================================================================
# NEW in v0.2.0: Utility and Spatial modules
# ============================================================================
# These are available as submodules: secactpy.utils and secactpy.spatial
# We also import commonly used functions to top-level for convenience

UTILS_AVAILABLE = False
SPATIAL_AVAILABLE = False

try:
    from .utils import (
        # Sparse matrix utilities
        sweep_sparse,
        normalize_sparse,
        sparse_column_scale,
        sparse_log1p,
        sparse_row_vars,
        sparse_col_vars,
        # Gene utilities  
        transfer_symbol,
        rm_duplicates as utils_rm_duplicates,  # Avoid conflict with inference.expand_rows
        scalar1,
        filter_genes,
        match_genes,
        # Survival utilities
        coxph_best_separation,
        survival_analysis,
        kaplan_meier_plot,
        log_rank_test,
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Define placeholders if utils not available
    sweep_sparse = None
    normalize_sparse = None
    sparse_column_scale = None
    sparse_log1p = None
    sparse_row_vars = None
    sparse_col_vars = None
    transfer_symbol = None
    utils_rm_duplicates = None
    scalar1 = None
    filter_genes = None
    match_genes = None
    coxph_best_separation = None
    survival_analysis = None
    kaplan_meier_plot = None
    log_rank_test = None

try:
    from .spatial import (
        # Weights (Phase 1)
        calc_spatial_weights,
        calc_spatial_weights_visium,
        row_normalize_weights,
        spatial_lag,
        get_neighbors,
        coords_to_distance_matrix,
        # Colocalization (Phase 3)
        calc_colocalization,
        calc_neighborhood_enrichment,
        calc_ripley_k,
        calc_cross_ripley_k,
        calc_morans_i,
        calc_getis_ord_g,
        # Interface (Phase 3)
        detect_interface,
        analyze_interface_activity,
        extract_interface_profile,
        find_interface_hotspots,
        calc_interface_width,
        # L-R Network (Phase 3)
        load_lr_database,
        score_lr_interactions,
        score_lr_spatial,
        calc_communication_probability,
        aggregate_pathway_scores,
        identify_significant_interactions,
        compare_lr_conditions,
    )
    SPATIAL_AVAILABLE = True
except ImportError:
    # Define placeholders if spatial not available
    calc_spatial_weights = None
    calc_spatial_weights_visium = None
    row_normalize_weights = None
    spatial_lag = None
    get_neighbors = None
    coords_to_distance_matrix = None
    calc_colocalization = None
    calc_neighborhood_enrichment = None
    calc_ripley_k = None
    calc_cross_ripley_k = None
    calc_morans_i = None
    calc_getis_ord_g = None
    detect_interface = None
    analyze_interface_activity = None
    extract_interface_profile = None
    find_interface_hotspots = None
    calc_interface_width = None
    load_lr_database = None
    score_lr_interactions = None
    score_lr_spatial = None
    calc_communication_probability = None
    aggregate_pathway_scores = None
    identify_significant_interactions = None
    compare_lr_conditions = None

try:
    from .plotting import (
        # Spatial plots
        plot_spatial,
        plot_spatial_feature,
        plot_spatial_multi,
        plot_spatial_categorical,
        # Heatmaps
        plot_activity_heatmap,
        plot_activity_heatmap_simple,
        plot_top_activities,
        # Cell-cell communication
        plot_ccc_heatmap,
        plot_ccc_dotplot,
        plot_ccc_circle,
        # Bar plots
        plot_activity_bar,
        plot_activity_lollipop,
        plot_activity_waterfall,
        plot_significance,
        # Survival plots
        plot_kaplan_meier,
        plot_survival_by_activity,
        plot_forest,
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Placeholders
    plot_spatial = None
    plot_spatial_feature = None
    plot_spatial_multi = None
    plot_spatial_categorical = None
    plot_activity_heatmap = None
    plot_activity_heatmap_simple = None
    plot_top_activities = None
    plot_ccc_heatmap = None
    plot_ccc_dotplot = None
    plot_ccc_circle = None
    plot_activity_bar = None
    plot_activity_lollipop = None
    plot_activity_waterfall = None
    plot_significance = None
    plot_kaplan_meier = None
    plot_survival_by_activity = None
    plot_forest = None

__all__ = [
    # Version
    "__version__",
    # Main API
    "secact_activity",
    "secact_activity_inference",
    "secact_activity_inference_scrnaseq",
    "secact_activity_inference_st",
    "load_visium_10x",
    "load_expression_data",
    "prepare_data",
    "scale_columns",
    "compute_differential",
    "group_signatures",
    "expand_rows",
    # Signature loading
    "load_signature",
    "load_secact",
    "load_cytosig",
    "list_signatures",
    "get_signature_info",
    "AVAILABLE_SIGNATURES",
    # Ridge functions
    "ridge",
    "compute_projection_matrix",
    "ridge_with_precomputed_T",
    # Batch processing
    "ridge_batch",
    "estimate_batch_size",
    "estimate_memory",
    "StreamingResultWriter",
    "PopulationStats",
    "ProjectionComponents",
    "precompute_population_stats",
    "precompute_projection_components",
    "ridge_batch_sparse_preserving",
    # I/O
    "load_results",
    "save_results",
    "results_to_anndata",
    "load_as_anndata",
    "results_to_dataframes",
    "get_file_info",
    "concatenate_results",
    "save_st_results_to_h5ad",
    "add_activity_to_anndata",
    # RNG and Caching
    "GSLRNG",
    "generate_permutation_table",
    "generate_inverse_permutation_table",
    "generate_permutation_table_fast",
    "generate_inverse_permutation_table_fast",
    "get_cached_perm_table",
    "get_cached_inverse_perm_table",
    "clear_perm_cache",
    "list_cached_tables",
    "DEFAULT_CACHE_DIR",
    # Availability flags
    "CUPY_AVAILABLE",
    "CUPY_INIT_ERROR",
    "H5PY_AVAILABLE",
    "ANNDATA_AVAILABLE",
    "UTILS_AVAILABLE",
    "SPATIAL_AVAILABLE",
    "PLOTTING_AVAILABLE",
    # Constants
    "DEFAULT_LAMBDA",
    "DEFAULT_NRAND",
    "DEFAULT_SEED",
    # -------------------------------------------------------------------------
    # NEW in v0.2.0: Utils (also available via secactpy.utils)
    # -------------------------------------------------------------------------
    # Sparse matrix utilities
    "sweep_sparse",
    "normalize_sparse",
    "sparse_column_scale",
    "sparse_log1p",
    "sparse_row_vars",
    "sparse_col_vars",
    # Gene utilities
    "transfer_symbol",
    "scalar1",
    "filter_genes",
    "match_genes",
    # Survival utilities
    "coxph_best_separation",
    "survival_analysis",
    "kaplan_meier_plot",
    "log_rank_test",
    # -------------------------------------------------------------------------
    # NEW in v0.2.0: Spatial (also available via secactpy.spatial)
    # -------------------------------------------------------------------------
    # Weights
    "calc_spatial_weights",
    "calc_spatial_weights_visium",
    "row_normalize_weights",
    "spatial_lag",
    "get_neighbors",
    "coords_to_distance_matrix",
    # Colocalization
    "calc_colocalization",
    "calc_neighborhood_enrichment",
    "calc_ripley_k",
    "calc_cross_ripley_k",
    "calc_morans_i",
    "calc_getis_ord_g",
    # Interface detection
    "detect_interface",
    "analyze_interface_activity",
    "extract_interface_profile",
    "find_interface_hotspots",
    "calc_interface_width",
    # L-R network
    "load_lr_database",
    "score_lr_interactions",
    "score_lr_spatial",
    "calc_communication_probability",
    "aggregate_pathway_scores",
    "identify_significant_interactions",
    "compare_lr_conditions",
    # -------------------------------------------------------------------------
    # NEW in v0.2.0: Plotting (also available via secactpy.plotting)
    # -------------------------------------------------------------------------
    # Spatial plots
    "plot_spatial",
    "plot_spatial_feature",
    "plot_spatial_multi",
    "plot_spatial_categorical",
    # Heatmaps
    "plot_activity_heatmap",
    "plot_activity_heatmap_simple",
    "plot_top_activities",
    # Cell-cell communication
    "plot_ccc_heatmap",
    "plot_ccc_dotplot",
    "plot_ccc_circle",
    # Bar plots
    "plot_activity_bar",
    "plot_activity_lollipop",
    "plot_activity_waterfall",
    "plot_significance",
    # Survival plots
    "plot_kaplan_meier",
    "plot_survival_by_activity",
    "plot_forest",
]
