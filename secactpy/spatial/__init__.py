"""
SecActPy spatial analysis functions.

This module provides utilities for spatial transcriptomics analysis:
- Spatial weight calculation
- Cell-cell colocalization (coming soon)
- Tumor-stroma interface detection (coming soon)
- L-R network scoring (coming soon)
"""

from .weights import (
    calc_spatial_weights,
    calc_spatial_weights_visium,
    row_normalize_weights,
    spatial_lag,
    get_neighbors,
    coords_to_distance_matrix,
)

__all__ = [
    "calc_spatial_weights",
    "calc_spatial_weights_visium",
    "row_normalize_weights",
    "spatial_lag",
    "get_neighbors",
    "coords_to_distance_matrix",
]
