"""
Cell-type colocalization analysis for SecActPy.

Provides functions for analyzing spatial co-occurrence patterns
between cell types or features in spatial transcriptomics data.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats


def calc_colocalization(
    cell_types: np.ndarray,
    coords: np.ndarray,
    radius: float,
    method: Literal["jaccard", "dice", "overlap", "correlation", "odds_ratio"] = "jaccard",
    n_permutations: int = 0,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculate pairwise cell-type colocalization scores.

    Measures the spatial co-occurrence of different cell types
    within a specified neighborhood radius.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels for each spot/cell.
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    radius : float
        Neighborhood radius for defining co-occurrence.
    method : str, default "jaccard"
        Colocalization metric:
        - "jaccard": Jaccard index (intersection / union)
        - "dice": Dice coefficient (2 * intersection / sum)
        - "overlap": Overlap coefficient (intersection / min)
        - "correlation": Phi correlation coefficient
        - "odds_ratio": Log odds ratio
    n_permutations : int, default 0
        Number of permutations for significance testing.
        If 0, no permutation test is performed.
    seed : int, optional
        Random seed for permutation testing.

    Returns
    -------
    dict
        Dictionary containing:
        - 'score': Colocalization score matrix (cell_types × cell_types)
        - 'pvalue': P-value matrix (if n_permutations > 0)
        - 'zscore': Z-score matrix (if n_permutations > 0)

    Examples
    --------
    >>> result = calc_colocalization(
    ...     cell_types, coords,
    ...     radius=100,
    ...     method="jaccard",
    ...     n_permutations=1000
    ... )
    >>> coloc_matrix = result['score']
    >>> significant = result['pvalue'] < 0.05

    Notes
    -----
    R equivalent (SpaCET):
        calColocalization <- function(object, radius) {
          nn <- RANN::nn2(coords, searchtype="radius", radius=radius)
          ...
        }
    """
    from scipy.spatial import KDTree

    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    n_spots = len(cell_types)

    # Build spatial index
    tree = KDTree(coords)

    # Find neighbors within radius for each spot
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    # Create binary matrix: spots × cell_types
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    spot_types = np.array([type_to_idx[t] for t in cell_types])

    def compute_colocalization_matrix(spot_types_perm):
        """Compute colocalization for given type assignments."""
        # For each cell type pair, count co-occurrences
        coloc_matrix = np.zeros((n_types, n_types))
        count_matrix = np.zeros((n_types, n_types))

        for i, neighbors in enumerate(neighbors_list):
            type_i = spot_types_perm[i]
            for j in neighbors:
                if i != j:
                    type_j = spot_types_perm[j]
                    coloc_matrix[type_i, type_j] += 1

        # Normalize based on method
        type_counts = np.bincount(spot_types_perm, minlength=n_types)

        if method == "jaccard":
            # Jaccard: intersection / union
            for ti in range(n_types):
                for tj in range(n_types):
                    intersection = coloc_matrix[ti, tj]
                    # Union = all neighbors of type_i + all neighbors of type_j - intersection
                    union = (type_counts[ti] * np.mean([len(n) for n in neighbors_list]) +
                             type_counts[tj] * np.mean([len(n) for n in neighbors_list]) -
                             intersection)
                    if union > 0:
                        coloc_matrix[ti, tj] = intersection / union

        elif method == "dice":
            # Dice: 2 * intersection / (|A| + |B|)
            for ti in range(n_types):
                for tj in range(n_types):
                    intersection = coloc_matrix[ti, tj]
                    total = type_counts[ti] + type_counts[tj]
                    if total > 0:
                        coloc_matrix[ti, tj] = 2 * intersection / total

        elif method == "overlap":
            # Overlap: intersection / min(|A|, |B|)
            for ti in range(n_types):
                for tj in range(n_types):
                    intersection = coloc_matrix[ti, tj]
                    min_count = min(type_counts[ti], type_counts[tj])
                    if min_count > 0:
                        coloc_matrix[ti, tj] = intersection / min_count

        elif method == "correlation":
            # Phi coefficient (point-biserial for binary)
            # Already have co-occurrence counts, compute phi
            total_pairs = sum(len(n) - 1 for n in neighbors_list)  # Exclude self
            for ti in range(n_types):
                for tj in range(n_types):
                    a = coloc_matrix[ti, tj]  # Both present
                    b = type_counts[ti] * np.mean([len(n) for n in neighbors_list]) - a  # ti only
                    c = type_counts[tj] * np.mean([len(n) for n in neighbors_list]) - a  # tj only
                    d = total_pairs - a - b - c  # Neither
                    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
                    if denom > 0:
                        coloc_matrix[ti, tj] = (a * d - b * c) / denom

        elif method == "odds_ratio":
            # Log odds ratio
            total_pairs = sum(len(n) - 1 for n in neighbors_list)
            for ti in range(n_types):
                for tj in range(n_types):
                    a = coloc_matrix[ti, tj] + 0.5  # Add pseudocount
                    b = type_counts[ti] * np.mean([len(n) for n in neighbors_list]) - a + 0.5
                    c = type_counts[tj] * np.mean([len(n) for n in neighbors_list]) - a + 0.5
                    d = total_pairs - a - b - c + 0.5
                    if b > 0 and c > 0 and d > 0:
                        coloc_matrix[ti, tj] = np.log((a * d) / (b * c))

        return coloc_matrix

    # Compute observed colocalization
    observed = compute_colocalization_matrix(spot_types)

    result = {
        'score': pd.DataFrame(observed, index=unique_types, columns=unique_types)
    }

    # Permutation testing
    if n_permutations > 0:
        if seed is not None:
            np.random.seed(seed)

        perm_scores = np.zeros((n_permutations, n_types, n_types))

        for p in range(n_permutations):
            # Permute cell type labels
            perm_types = np.random.permutation(spot_types)
            perm_scores[p] = compute_colocalization_matrix(perm_types)

        # Calculate p-values (two-tailed)
        pvalues = np.zeros((n_types, n_types))
        zscores = np.zeros((n_types, n_types))

        for ti in range(n_types):
            for tj in range(n_types):
                obs = observed[ti, tj]
                perm = perm_scores[:, ti, tj]
                # Two-tailed p-value
                pvalues[ti, tj] = np.mean(np.abs(perm - np.mean(perm)) >= np.abs(obs - np.mean(perm)))
                # Z-score
                std = np.std(perm)
                if std > 0:
                    zscores[ti, tj] = (obs - np.mean(perm)) / std

        result['pvalue'] = pd.DataFrame(pvalues, index=unique_types, columns=unique_types)
        result['zscore'] = pd.DataFrame(zscores, index=unique_types, columns=unique_types)

    return result


def calc_neighborhood_enrichment(
    cell_types: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 10,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculate neighborhood enrichment scores (like squidpy).

    Measures whether cell types are enriched or depleted in each
    other's neighborhoods compared to random expectation.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels.
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    n_neighbors : int, default 10
        Number of nearest neighbors to consider.
    n_permutations : int, default 1000
        Number of permutations for significance.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with 'zscore' and 'count' matrices.

    Examples
    --------
    >>> result = calc_neighborhood_enrichment(cell_types, coords)
    >>> enrichment = result['zscore']
    """
    from scipy.spatial import KDTree

    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    n_spots = len(cell_types)

    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    spot_types = np.array([type_to_idx[t] for t in cell_types])

    # Find k nearest neighbors
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=n_neighbors + 1)  # +1 for self
    indices = indices[:, 1:]  # Remove self

    def count_neighbor_types(spot_types_arr):
        """Count neighbor types for each cell type."""
        counts = np.zeros((n_types, n_types))
        for i, neighbors in enumerate(indices):
            type_i = spot_types_arr[i]
            for j in neighbors:
                type_j = spot_types_arr[j]
                counts[type_i, type_j] += 1
        return counts

    # Observed counts
    observed = count_neighbor_types(spot_types)

    # Permutation testing
    if seed is not None:
        np.random.seed(seed)

    perm_counts = np.zeros((n_permutations, n_types, n_types))
    for p in range(n_permutations):
        perm_types = np.random.permutation(spot_types)
        perm_counts[p] = count_neighbor_types(perm_types)

    # Z-scores
    mean_perm = np.mean(perm_counts, axis=0)
    std_perm = np.std(perm_counts, axis=0)
    std_perm[std_perm == 0] = 1  # Avoid division by zero

    zscores = (observed - mean_perm) / std_perm

    return {
        'zscore': pd.DataFrame(zscores, index=unique_types, columns=unique_types),
        'count': pd.DataFrame(observed, index=unique_types, columns=unique_types)
    }


def calc_ripley_k(
    cell_types: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    cell_type: Optional[str] = None,
    boundary_correction: bool = True
) -> pd.DataFrame:
    """
    Calculate Ripley's K function for spatial point pattern analysis.

    Measures clustering (K > expected) or dispersion (K < expected)
    of cell types at different spatial scales.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels.
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    radii : np.ndarray
        Array of radii to evaluate K function.
    cell_type : str, optional
        Specific cell type to analyze. If None, analyzes all points.
    boundary_correction : bool, default True
        Whether to apply edge correction.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'radius', 'K', 'L', 'L_theo'
        where L = sqrt(K/pi) - r is the normalized version.

    Examples
    --------
    >>> radii = np.linspace(10, 200, 20)
    >>> ripley = calc_ripley_k(cell_types, coords, radii, cell_type="Tumor")
    """
    if cell_type is not None:
        mask = cell_types == cell_type
        points = coords[mask]
    else:
        points = coords

    n_points = len(points)
    if n_points < 2:
        return pd.DataFrame({'radius': radii, 'K': np.nan, 'L': np.nan, 'L_theo': 0})

    # Study area (bounding box)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)

    # Intensity
    intensity = n_points / area

    # Calculate K for each radius
    K_values = []

    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(points))

    for r in radii:
        # Count pairs within distance r
        n_pairs = np.sum(dist_matrix < r) - n_points  # Subtract diagonal

        if boundary_correction:
            # Simple Ripley edge correction
            # Weight by proportion of circle inside study area
            weights = np.ones(n_points)
            for i, (x, y) in enumerate(points):
                # Approximate edge correction
                d_edge = min(x - x_min, x_max - x, y - y_min, y_max - y)
                if d_edge < r:
                    weights[i] = 1 / (1 - np.arccos(d_edge / r) / np.pi)

            # Weighted count
            K = area / (n_points * (n_points - 1)) * n_pairs * np.mean(weights)
        else:
            K = area / (n_points * (n_points - 1)) * n_pairs

        K_values.append(K)

    K_values = np.array(K_values)

    # L function: L(r) = sqrt(K(r)/pi) - r
    # Under CSR, L(r) should be 0
    L_values = np.sqrt(K_values / np.pi) - radii
    L_theo = np.zeros_like(radii)  # Theoretical under CSR

    return pd.DataFrame({
        'radius': radii,
        'K': K_values,
        'L': L_values,
        'L_theo': L_theo
    })


def calc_cross_ripley_k(
    cell_types: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    type_a: str,
    type_b: str,
    boundary_correction: bool = True
) -> pd.DataFrame:
    """
    Calculate cross Ripley's K function between two cell types.

    Measures spatial association between two different cell types.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels.
    coords : np.ndarray
        Spatial coordinates.
    radii : np.ndarray
        Array of radii to evaluate.
    type_a, type_b : str
        The two cell types to analyze.
    boundary_correction : bool, default True
        Whether to apply edge correction.

    Returns
    -------
    pd.DataFrame
        DataFrame with K and L values.
    """
    mask_a = cell_types == type_a
    mask_b = cell_types == type_b

    points_a = coords[mask_a]
    points_b = coords[mask_b]

    n_a = len(points_a)
    n_b = len(points_b)

    if n_a < 1 or n_b < 1:
        return pd.DataFrame({'radius': radii, 'K': np.nan, 'L': np.nan})

    # Study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)

    # Cross-distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points_a, points_b)

    K_values = []
    for r in radii:
        n_pairs = np.sum(dist_matrix < r)
        K = area / (n_a * n_b) * n_pairs
        K_values.append(K)

    K_values = np.array(K_values)
    L_values = np.sqrt(K_values / np.pi) - radii

    return pd.DataFrame({
        'radius': radii,
        'K': K_values,
        'L': L_values
    })


def calc_morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    radius: Optional[float] = None,
    weights: Optional[sparse.spmatrix] = None,
    n_permutations: int = 0,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate Moran's I spatial autocorrelation statistic.

    Parameters
    ----------
    values : np.ndarray
        Feature values for each spot.
    coords : np.ndarray
        Spatial coordinates.
    radius : float, optional
        Neighborhood radius. Required if weights not provided.
    weights : sparse matrix, optional
        Pre-computed spatial weight matrix.
    n_permutations : int, default 0
        Number of permutations for significance testing.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with 'I' (Moran's I), 'expected' (expected under null),
        'pvalue' (if permutations > 0), 'zscore'.

    Examples
    --------
    >>> result = calc_morans_i(gene_expr, coords, radius=100, n_permutations=999)
    >>> print(f"Moran's I: {result['I']:.3f}, p-value: {result['pvalue']:.4f}")
    """
    try:
        from .weights import calc_spatial_weights, row_normalize_weights
    except ImportError:
        from weights import calc_spatial_weights, row_normalize_weights

    n = len(values)

    # Get or compute weights
    if weights is None:
        if radius is None:
            raise ValueError("Must provide either radius or weights")
        weights = calc_spatial_weights(coords, radius=radius, method="binary")

    W = row_normalize_weights(weights)

    # Center values
    z = values - np.mean(values)

    # Moran's I = (n / S0) * (z' W z) / (z' z)
    numerator = z @ W @ z
    denominator = z @ z
    S0 = W.sum()

    if denominator == 0 or S0 == 0:
        return {'I': np.nan, 'expected': np.nan, 'pvalue': np.nan, 'zscore': np.nan}

    I = (n / S0) * (numerator / denominator)

    # Expected value under null
    expected = -1 / (n - 1)

    result = {'I': I, 'expected': expected}

    # Permutation test
    if n_permutations > 0:
        if seed is not None:
            np.random.seed(seed)

        perm_I = np.zeros(n_permutations)
        for p in range(n_permutations):
            z_perm = np.random.permutation(z)
            num_perm = z_perm @ W @ z_perm
            perm_I[p] = (n / S0) * (num_perm / denominator)

        # P-value (two-tailed)
        result['pvalue'] = np.mean(np.abs(perm_I) >= np.abs(I))
        result['zscore'] = (I - np.mean(perm_I)) / np.std(perm_I)

    return result


def calc_getis_ord_g(
    values: np.ndarray,
    coords: np.ndarray,
    radius: float,
    star: bool = True
) -> np.ndarray:
    """
    Calculate Getis-Ord G* (hot spot) statistic for each location.

    Identifies statistically significant spatial clusters of high
    (hot spots) or low (cold spots) values.

    Parameters
    ----------
    values : np.ndarray
        Feature values.
    coords : np.ndarray
        Spatial coordinates.
    radius : float
        Neighborhood radius.
    star : bool, default True
        If True, compute G* (includes self). If False, compute G.

    Returns
    -------
    np.ndarray
        Z-scores for each location. High positive = hot spot,
        high negative = cold spot.

    Examples
    --------
    >>> g_star = calc_getis_ord_g(gene_expr, coords, radius=100)
    >>> hot_spots = g_star > 1.96  # p < 0.05
    """
    try:
        from .weights import calc_spatial_weights
    except ImportError:
        from weights import calc_spatial_weights

    n = len(values)
    W = calc_spatial_weights(coords, radius=radius, method="binary")

    if star:
        W.setdiag(1)  # Include self for G*

    W = W.tocsr()

    # Global statistics
    x_mean = np.mean(values)
    S = np.std(values)

    # G* for each location
    g_star = np.zeros(n)

    for i in range(n):
        # Neighbors of i
        neighbors = W[i].indices
        w_i = W[i].data

        # Numerator: sum of weighted values - expected
        sum_wj_xj = np.sum(w_i * values[neighbors])
        sum_wj = np.sum(w_i)

        numerator = sum_wj_xj - x_mean * sum_wj

        # Denominator
        sum_wj_sq = np.sum(w_i ** 2)
        denominator = S * np.sqrt((n * sum_wj_sq - sum_wj ** 2) / (n - 1))

        if denominator > 0:
            g_star[i] = numerator / denominator

    return g_star
