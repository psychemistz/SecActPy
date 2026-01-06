"""
Spatial weight calculation utilities for SecActPy.

Provides functions for computing spatial adjacency matrices
with distance-based weighting (e.g., Gaussian kernel).
"""

from typing import Literal, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree


def calc_spatial_weights(
    coords: np.ndarray,
    radius: float,
    sigma: float = 100.0,
    diag_as_zero: bool = True,
    method: Literal["gaussian", "binary", "inverse"] = "gaussian",
    n_neighbors: Optional[int] = None
) -> sparse.csr_matrix:
    """
    Calculate spatial weight matrix based on coordinates.

    Computes a sparse adjacency matrix where weights represent
    spatial proximity between spots/cells.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_spots, 2) or (n_spots, 3).
    radius : float
        Maximum distance for neighbors. Points beyond this are zero.
    sigma : float, default 100.0
        Bandwidth parameter for Gaussian kernel.
        Larger sigma = slower decay with distance.
    diag_as_zero : bool, default True
        Whether diagonal (self-connections) should be zero.
    method : {"gaussian", "binary", "inverse"}, default "gaussian"
        Weight calculation method:
        - "gaussian": w = exp(-d² / 2σ²)
        - "binary": w = 1 if d <= radius else 0
        - "inverse": w = 1 / (d + 1)
    n_neighbors : int, optional
        If set, use k-nearest neighbors instead of radius search.
        Overrides radius parameter for neighbor finding.

    Returns
    -------
    sparse.csr_matrix
        Sparse weight matrix of shape (n_spots, n_spots).

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.rand(1000, 2) * 1000
    >>> W = calc_spatial_weights(coords, radius=100, sigma=50)
    >>> print(f"Density: {W.nnz / W.shape[0]**2:.4f}")

    Notes
    -----
    R equivalent (SpaCET):
        calWeights <- function(spotCoordinates, radius, sigma=100, diagAsZero=TRUE) {
          nn_result <- RANN::nn2(spotCoordinates, searchtype="radius", radius=radius, k=nrow(spotCoordinates))
          ...
          x <- exp(-x^2 / (2*sigma^2))
          W <- sparseMatrix(i=i, j=j, x=x, dims=c(...))
          W
        }
    """
    n_spots = coords.shape[0]

    # Build KD-tree for efficient neighbor search
    tree = KDTree(coords)

    if n_neighbors is not None:
        # k-nearest neighbors search
        distances, indices = tree.query(coords, k=n_neighbors + 1)  # +1 for self
        # Flatten and filter
        i_list = []
        j_list = []
        d_list = []
        for spot_i in range(n_spots):
            for k in range(1, n_neighbors + 1):  # Skip self (k=0)
                if indices[spot_i, k] < n_spots:  # Valid neighbor
                    i_list.append(spot_i)
                    j_list.append(indices[spot_i, k])
                    d_list.append(distances[spot_i, k])
    else:
        # Radius-based search
        neighbor_lists = tree.query_ball_tree(tree, r=radius)

        i_list = []
        j_list = []
        d_list = []

        for spot_i, neighbors in enumerate(neighbor_lists):
            for spot_j in neighbors:
                if spot_i != spot_j or not diag_as_zero:
                    dist = np.linalg.norm(coords[spot_i] - coords[spot_j])
                    if dist <= radius and dist > 0:
                        i_list.append(spot_i)
                        j_list.append(spot_j)
                        d_list.append(dist)

    # Convert to arrays
    i_arr = np.array(i_list, dtype=np.int32)
    j_arr = np.array(j_list, dtype=np.int32)
    d_arr = np.array(d_list, dtype=np.float64)

    # Calculate weights based on method
    if method == "gaussian":
        weights = np.exp(-d_arr ** 2 / (2 * sigma ** 2))
    elif method == "binary":
        weights = np.ones_like(d_arr)
    elif method == "inverse":
        weights = 1.0 / (d_arr + 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create sparse matrix
    W = sparse.csr_matrix(
        (weights, (i_arr, j_arr)),
        shape=(n_spots, n_spots)
    )

    # Handle diagonal
    if not diag_as_zero:
        W.setdiag(1.0)

    return W


def calc_spatial_weights_visium(
    coords: np.ndarray,
    n_rings: int = 1,
    include_center: bool = False
) -> sparse.csr_matrix:
    """
    Calculate spatial weights for Visium hexagonal grid.

    Uses the hexagonal grid structure of Visium spots to define
    neighbors more precisely than distance-based methods.

    Parameters
    ----------
    coords : np.ndarray
        Array coordinates (row, col) for Visium spots.
        Shape: (n_spots, 2) with integer array coordinates.
    n_rings : int, default 1
        Number of hexagonal rings to include as neighbors.
        1 = immediate neighbors (6 spots)
        2 = two rings (18 spots)
    include_center : bool, default False
        Whether to include self-connections (diagonal = 1).

    Returns
    -------
    sparse.csr_matrix
        Binary adjacency matrix.

    Examples
    --------
    >>> # For Visium, use array_row and array_col from spatial metadata
    >>> coords = np.column_stack([adata.obs['array_row'], adata.obs['array_col']])
    >>> W = calc_spatial_weights_visium(coords, n_rings=1)
    """
    n_spots = coords.shape[0]

    # Build coordinate lookup
    coord_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}

    # Visium hexagonal offsets for n_rings
    # Ring 1: 6 immediate neighbors
    ring1_offsets = [
        (0, -2), (0, 2),      # Left, Right
        (-1, -1), (-1, 1),    # Upper-left, Upper-right
        (1, -1), (1, 1)       # Lower-left, Lower-right
    ]

    # Ring 2: 12 additional neighbors (if needed)
    ring2_offsets = [
        (0, -4), (0, 4),           # Far left, Far right
        (-2, -2), (-2, 0), (-2, 2),  # Upper row
        (2, -2), (2, 0), (2, 2),    # Lower row
        (-1, -3), (-1, 3),          # Upper diagonal
        (1, -3), (1, 3)             # Lower diagonal
    ]

    offsets = ring1_offsets[:] if n_rings >= 1 else []
    if n_rings >= 2:
        offsets.extend(ring2_offsets)

    i_list = []
    j_list = []

    for idx, (row, col) in enumerate(coords):
        row, col = int(row), int(col)

        for dr, dc in offsets:
            neighbor_coord = (row + dr, col + dc)
            if neighbor_coord in coord_to_idx:
                j_idx = coord_to_idx[neighbor_coord]
                i_list.append(idx)
                j_list.append(j_idx)

    # Create sparse matrix
    weights = np.ones(len(i_list), dtype=np.float64)
    W = sparse.csr_matrix(
        (weights, (i_list, j_list)),
        shape=(n_spots, n_spots)
    )

    if include_center:
        W.setdiag(1.0)

    return W


def row_normalize_weights(W: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Row-normalize weight matrix (each row sums to 1).

    This is commonly used for spatial lag calculations.

    Parameters
    ----------
    W : sparse.spmatrix
        Input weight matrix.

    Returns
    -------
    sparse.csr_matrix
        Row-normalized weight matrix.
    """
    W = W.tocsr().copy()
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero

    # Divide each row by its sum
    for i in range(W.shape[0]):
        start, end = W.indptr[i], W.indptr[i + 1]
        W.data[start:end] /= row_sums[i]

    return W


def spatial_lag(
    values: np.ndarray,
    W: sparse.spmatrix,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate spatial lag (weighted average of neighbors).

    Parameters
    ----------
    values : np.ndarray
        Values for each spot/cell. Shape: (n_spots,) or (n_spots, n_features).
    W : sparse.spmatrix
        Spatial weight matrix.
    normalize : bool, default True
        Whether to row-normalize W before calculation.

    Returns
    -------
    np.ndarray
        Spatial lag values with same shape as input.

    Examples
    --------
    >>> # Calculate spatially smoothed gene expression
    >>> W = calc_spatial_weights(coords, radius=100)
    >>> expr_smoothed = spatial_lag(expr, W)
    """
    if normalize:
        W = row_normalize_weights(W)

    if values.ndim == 1:
        return W.dot(values)
    else:
        return W.dot(values)


def get_neighbors(
    W: sparse.spmatrix,
    spot_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get neighbors and weights for a specific spot.

    Parameters
    ----------
    W : sparse.spmatrix
        Sparse weight matrix.
    spot_idx : int
        Index of the spot.

    Returns
    -------
    neighbor_indices : np.ndarray
        Indices of neighboring spots.
    weights : np.ndarray
        Corresponding weights.
    """
    W = W.tocsr()
    start, end = W.indptr[spot_idx], W.indptr[spot_idx + 1]
    neighbor_indices = W.indices[start:end]
    weights = W.data[start:end]
    return neighbor_indices, weights


def coords_to_distance_matrix(
    coords: np.ndarray,
    max_dist: Optional[float] = None
) -> Union[np.ndarray, sparse.csr_matrix]:
    """
    Convert coordinates to pairwise distance matrix.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_points, n_dims).
    max_dist : float, optional
        If provided, return sparse matrix with distances > max_dist as zero.

    Returns
    -------
    np.ndarray or sparse.csr_matrix
        Distance matrix.
    """
    from scipy.spatial.distance import pdist, squareform

    if max_dist is None:
        # Full dense matrix
        return squareform(pdist(coords))
    else:
        # Sparse matrix with threshold
        W = calc_spatial_weights(
            coords,
            radius=max_dist,
            method="binary",
            diag_as_zero=True
        )
        # Replace binary weights with actual distances
        tree = KDTree(coords)
        for i in range(W.shape[0]):
            neighbors, weights = get_neighbors(W, i)
            for j_idx, j in enumerate(neighbors):
                dist = np.linalg.norm(coords[i] - coords[j])
                W[i, j] = dist
        return W
