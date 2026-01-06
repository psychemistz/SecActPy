"""
SecActPy spatial analysis functions.

This module provides utilities for spatial transcriptomics analysis:
- Spatial weight calculation
- Cell-cell colocalization
- Tumor-stroma interface detection
- Ligand-receptor network scoring

Examples
--------
>>> from secactpy.spatial import calc_spatial_weights, calc_colocalization
>>> from secactpy.spatial import detect_interface, score_lr_interactions
>>>
>>> # Spatial weights for neighborhood analysis
>>> W = calc_spatial_weights(coords, radius=100, sigma=50)
>>>
>>> # Cell-type colocalization with permutation testing
>>> coloc = calc_colocalization(cell_types, coords, radius=100, n_permutations=1000)
>>>
>>> # Tumor-stroma interface detection
>>> interface = detect_interface(
...     cell_types, coords,
...     tumor_types=["Tumor"],
...     stroma_types=["Fibroblast", "Endothelial"],
...     radius=100
... )
>>>
>>> # Ligand-receptor interaction scoring
>>> lr_pairs = load_lr_database()
>>> lr_scores = score_lr_interactions(expression, coords, cell_types, lr_pairs, radius=100)
"""

# Spatial weight calculation
from .weights import (
    calc_spatial_weights,
    calc_spatial_weights_visium,
    row_normalize_weights,
    spatial_lag,
    get_neighbors,
    coords_to_distance_matrix,
)

# Cell-cell colocalization
from .colocalization import (
    calc_colocalization,
    calc_neighborhood_enrichment,
    calc_ripley_k,
    calc_cross_ripley_k,
    calc_morans_i,
    calc_getis_ord_g,
)

# Tumor-stroma interface detection
from .interface import (
    detect_interface,
    analyze_interface_activity,
    extract_interface_profile,
    find_interface_hotspots,
    calc_interface_width,
)

# Ligand-receptor network scoring
from .lr_network import (
    load_lr_database,
    score_lr_interactions,
    score_lr_spatial,
    calc_communication_probability,
    aggregate_pathway_scores,
    identify_significant_interactions,
    compare_lr_conditions,
)

__all__ = [
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
    # Interface
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
]
