"""
Sparse matrix utilities for SecActPy.

Provides efficient operations on scipy sparse matrices that mirror
R's Matrix package functionality.
"""

from typing import Callable, Literal, Union
import numpy as np
from scipy import sparse


def sweep_sparse(
    m: sparse.spmatrix,
    margin: Literal[0, 1],
    stats: np.ndarray,
    fun: Union[str, Callable] = "subtract"
) -> sparse.spmatrix:
    """
    Sweep out array summaries from sparse matrices.

    Analogous to R's sweep() function but optimized for sparse matrices.
    Applies an operation between each row/column and corresponding statistic.

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix (CSR or CSC format recommended).
    margin : {0, 1}
        0 = operate on rows (apply stats to each row)
        1 = operate on columns (apply stats to each column)
    stats : np.ndarray
        Statistics array. Length must match the size of the margin dimension.
    fun : str or callable, default "subtract"
        Operation to apply. Built-in options:
        - "subtract" or "-": m - stats
        - "add" or "+": m + stats
        - "multiply" or "*": m * stats
        - "divide" or "/": m / stats
        Or pass a custom function f(x, s) -> result.

    Returns
    -------
    sparse.spmatrix
        Result matrix in same format as input.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> m = sparse.random(100, 50, density=0.1, format='csr')
    >>> row_means = np.array(m.mean(axis=1)).flatten()
    >>> m_centered = sweep_sparse(m, margin=0, stats=row_means, fun="subtract")

    Notes
    -----
    This is equivalent to R's:
        sweep_sparse <- function(m, margin, stats, fun) {
          f <- match.fun(fun)
          if(margin==1) {
            idx <- m@i + 1
          } else {
            idx <- rep(1:m@Dim[2], diff(m@p))
          }
          m@x <- f(m@x, stats[idx])
          m
        }
    """
    # Resolve function
    if isinstance(fun, str):
        fun_map = {
            "subtract": lambda x, s: x - s,
            "-": lambda x, s: x - s,
            "add": lambda x, s: x + s,
            "+": lambda x, s: x + s,
            "multiply": lambda x, s: x * s,
            "*": lambda x, s: x * s,
            "divide": lambda x, s: x / s,
            "/": lambda x, s: x / s,
        }
        if fun not in fun_map:
            raise ValueError(f"Unknown function '{fun}'. Use one of {list(fun_map.keys())} or pass a callable.")
        f = fun_map[fun]
    else:
        f = fun

    # Convert to appropriate format
    if margin == 0:  # Row operation - CSR is efficient
        m = m.tocsr().copy()
        # For CSR: indices gives column indices, indptr separates rows
        # We need row index for each non-zero element
        row_indices = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        m.data = f(m.data, stats[row_indices])
    else:  # Column operation - CSC is efficient
        m = m.tocsc().copy()
        # For CSC: indices gives row indices, indptr separates columns
        # We need column index for each non-zero element
        col_indices = np.repeat(np.arange(m.shape[1]), np.diff(m.indptr))
        m.data = f(m.data, stats[col_indices])

    return m


def normalize_sparse(
    m: sparse.spmatrix,
    method: Literal["l1", "l2", "max", "sum"] = "l1",
    axis: Literal[0, 1] = 1
) -> sparse.spmatrix:
    """
    Normalize sparse matrix rows or columns.

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix.
    method : {"l1", "l2", "max", "sum"}, default "l1"
        Normalization method:
        - "l1": Divide by L1 norm (sum of absolute values)
        - "l2": Divide by L2 norm (Euclidean length)
        - "max": Divide by maximum absolute value
        - "sum": Divide by sum (for count data, gives proportions)
    axis : {0, 1}, default 1
        0 = normalize columns, 1 = normalize rows

    Returns
    -------
    sparse.spmatrix
        Normalized matrix.

    Examples
    --------
    >>> # Normalize rows to sum to 1 (like TPM normalization)
    >>> m_norm = normalize_sparse(m, method="sum", axis=1)
    """
    m = m.tocsr() if axis == 1 else m.tocsc()
    m = m.copy().astype(np.float64)

    # axis=1 means normalize rows -> compute stats along axis=1 (sum columns for each row)
    # axis=0 means normalize columns -> compute stats along axis=0 (sum rows for each column)
    if method == "l1":
        norms = np.array(np.abs(m).sum(axis=axis)).flatten()
    elif method == "l2":
        norms = np.sqrt(np.array(m.power(2).sum(axis=axis)).flatten())
    elif method == "max":
        norms = np.array(np.abs(m).max(axis=axis)).flatten()
    elif method == "sum":
        norms = np.array(m.sum(axis=axis)).flatten()
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Avoid division by zero
    norms[norms == 0] = 1.0

    # Apply normalization:
    # axis=1 (normalize rows) -> margin=0 (apply stats per row)
    # axis=0 (normalize columns) -> margin=1 (apply stats per column)
    sweep_margin = 1 - axis
    return sweep_sparse(m, margin=sweep_margin, stats=norms, fun="divide")


def sparse_column_scale(
    m: sparse.spmatrix,
    scale_factor: float = 1e6
) -> sparse.spmatrix:
    """
    Scale sparse matrix columns (e.g., for CPM/TPM normalization).

    Equivalent to: m * scale_factor / colSums(m)

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix (genes × cells/spots).
    scale_factor : float, default 1e6
        Scale factor (1e6 for CPM, 1e4 for Seurat default).

    Returns
    -------
    sparse.spmatrix
        Scaled matrix.

    Examples
    --------
    >>> # CPM normalization
    >>> m_cpm = sparse_column_scale(m, scale_factor=1e6)
    """
    m = m.tocsc().copy().astype(np.float64)
    col_sums = np.array(m.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0  # Avoid division by zero
    scale = scale_factor / col_sums
    return sweep_sparse(m, margin=1, stats=scale, fun="multiply")


def sparse_log1p(m: sparse.spmatrix, base: float = np.e) -> sparse.spmatrix:
    """
    Compute log(1 + x) for sparse matrix, preserving sparsity.

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix (non-negative values).
    base : float, default e
        Logarithm base. Use 2 for log2, 10 for log10.

    Returns
    -------
    sparse.spmatrix
        Log-transformed matrix.

    Examples
    --------
    >>> # Log2 transform
    >>> m_log = sparse_log1p(m, base=2)
    """
    m = m.copy()
    if base == np.e:
        m.data = np.log1p(m.data)
    else:
        m.data = np.log1p(m.data) / np.log(base)
    return m


def sparse_row_vars(m: sparse.spmatrix, ddof: int = 1) -> np.ndarray:
    """
    Calculate variance for each row of a sparse matrix.

    Memory-efficient implementation that doesn't densify the matrix.

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix.
    ddof : int, default 1
        Delta degrees of freedom (1 for sample variance).

    Returns
    -------
    np.ndarray
        Variance for each row.
    """
    m = m.tocsr()
    n_cols = m.shape[1]

    # E[X^2]
    sq_mean = np.array(m.power(2).mean(axis=1)).flatten()

    # E[X]^2
    mean_sq = np.array(m.mean(axis=1)).flatten() ** 2

    # Var = E[X^2] - E[X]^2, with Bessel's correction
    var = (sq_mean - mean_sq) * n_cols / (n_cols - ddof)

    return var


def sparse_col_vars(m: sparse.spmatrix, ddof: int = 1) -> np.ndarray:
    """
    Calculate variance for each column of a sparse matrix.

    Parameters
    ----------
    m : sparse.spmatrix
        Input sparse matrix.
    ddof : int, default 1
        Delta degrees of freedom (1 for sample variance).

    Returns
    -------
    np.ndarray
        Variance for each column.
    """
    m = m.tocsc()
    n_rows = m.shape[0]

    sq_mean = np.array(m.power(2).mean(axis=0)).flatten()
    mean_sq = np.array(m.mean(axis=0)).flatten() ** 2
    var = (sq_mean - mean_sq) * n_rows / (n_rows - ddof)

    return var
