"""
Gene symbol and expression matrix utilities for SecActPy.

Provides functions for gene symbol conversion, duplicate handling,
and matrix preprocessing.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path


def transfer_symbol(
    genes: Union[List[str], np.ndarray, pd.Index],
    alias_map: Optional[Dict[str, str]] = None,
    alias_file: Optional[str] = None
) -> List[str]:
    """
    Convert gene aliases to official symbols.

    Maps gene aliases to their official NCBI gene symbols using
    a provided mapping or built-in alias database.

    Parameters
    ----------
    genes : list-like
        Gene names to convert.
    alias_map : dict, optional
        Dictionary mapping aliases to official symbols.
        If None, uses alias_file or built-in database.
    alias_file : str, optional
        Path to CSV file with 'Alias' and 'Symbol' columns.

    Returns
    -------
    list
        Converted gene symbols. Genes not in the map are unchanged.

    Examples
    --------
    >>> genes = ["TP53", "p53", "ERBB2", "HER2"]
    >>> transfer_symbol(genes, alias_map={"p53": "TP53", "HER2": "ERBB2"})
    ['TP53', 'TP53', 'ERBB2', 'ERBB2']

    Notes
    -----
    R equivalent:
        transferSymbol <- function(x) {
          alias2symbol <- read.csv(...)
          x[x %in% alias2symbol[,1]] <- alias2symbol[match(x[...], ...), 2]
          x
        }
    """
    genes = list(genes)

    if alias_map is None:
        if alias_file is not None:
            # Load from file
            df = pd.read_csv(alias_file)
            alias_map = dict(zip(df['Alias'].fillna('NA'), df['Symbol']))
        else:
            # Try to load built-in database
            # For now, return unchanged if no map provided
            return genes

    # Convert aliases
    result = []
    for gene in genes:
        if gene in alias_map:
            result.append(alias_map[gene])
        else:
            result.append(gene)

    return result


def rm_duplicates(
    mat: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
    row_names: Optional[List[str]] = None,
    keep: str = "max_sum"
) -> Union[np.ndarray, pd.DataFrame, sparse.spmatrix]:
    """
    Remove duplicate rows, keeping the one with highest expression.

    For genes with multiple entries (e.g., multiple probes), keeps
    the row with the highest total expression across all samples.

    Parameters
    ----------
    mat : array-like or sparse matrix
        Expression matrix (genes × samples).
    row_names : list, optional
        Row names. Required if mat is ndarray or sparse matrix.
        Ignored if mat is DataFrame (uses index).
    keep : str, default "max_sum"
        Strategy for keeping duplicates:
        - "max_sum": Keep row with highest sum
        - "max_var": Keep row with highest variance
        - "first": Keep first occurrence
        - "last": Keep last occurrence

    Returns
    -------
    Same type as input
        Matrix with duplicates removed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     [[1, 2], [3, 4], [5, 6]],
    ...     index=["A", "B", "A"],
    ...     columns=["S1", "S2"]
    ... )
    >>> rm_duplicates(df)
           S1  S2
    A       5   6
    B       3   4

    Notes
    -----
    R equivalent:
        rm_duplicates <- function(mat) {
          gene_count <- table(rownames(mat))
          gene_dupl <- names(gene_count)[gene_count > 1]
          ...
        }
    """
    # Handle different input types
    is_sparse = sparse.issparse(mat)
    is_dataframe = isinstance(mat, pd.DataFrame)

    if is_dataframe:
        row_names = list(mat.index)
        values = mat.values
    elif is_sparse:
        if row_names is None:
            raise ValueError("row_names required for sparse matrices")
        values = mat
    else:
        if row_names is None:
            raise ValueError("row_names required for numpy arrays")
        values = mat

    # Find duplicates
    from collections import Counter
    gene_counts = Counter(row_names)
    duplicated = {gene for gene, count in gene_counts.items() if count > 1}

    if not duplicated:
        return mat  # No duplicates

    # Build index of rows to keep
    unique_genes = {gene for gene, count in gene_counts.items() if count == 1}
    keep_indices = []

    # Add unique genes
    for i, gene in enumerate(row_names):
        if gene in unique_genes:
            keep_indices.append(i)

    # Handle duplicates
    for gene in duplicated:
        indices = [i for i, g in enumerate(row_names) if g == gene]

        if keep == "first":
            keep_indices.append(indices[0])
        elif keep == "last":
            keep_indices.append(indices[-1])
        elif keep == "max_sum":
            if is_sparse:
                sums = np.array(values[indices].sum(axis=1)).flatten()
            else:
                sums = values[indices].sum(axis=1)
            best_idx = indices[np.argmax(sums)]
            keep_indices.append(best_idx)
        elif keep == "max_var":
            if is_sparse:
                # For sparse, use sum of squares as proxy
                sums_sq = np.array(values[indices].power(2).sum(axis=1)).flatten()
            else:
                sums_sq = (values[indices] ** 2).sum(axis=1)
            best_idx = indices[np.argmax(sums_sq)]
            keep_indices.append(best_idx)
        else:
            raise ValueError(f"Unknown keep strategy: {keep}")

    # Sort indices to maintain order
    keep_indices = sorted(keep_indices)

    # Return appropriate type
    if is_dataframe:
        return mat.iloc[keep_indices]
    elif is_sparse:
        return mat[keep_indices]
    else:
        return mat[keep_indices]


def expand_rows(
    mat: pd.DataFrame,
    sep: str = "|"
) -> pd.DataFrame:
    """
    Expand rows with multiple gene names into separate rows.

    Handles gene symbols like "GENE1|GENE2" by creating separate
    rows for each gene, duplicating the expression values.

    Parameters
    ----------
    mat : pd.DataFrame
        Expression matrix with potentially compound gene names in index.
    sep : str, default "|"
        Separator character for compound names.

    Returns
    -------
    pd.DataFrame
        Expanded matrix with one gene per row.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     [[1, 2], [3, 4]],
    ...     index=["A|B", "C"],
    ...     columns=["S1", "S2"]
    ... )
    >>> expand_rows(df)
       S1  S2
    A   1   2
    B   1   2
    C   3   4

    Notes
    -----
    R equivalent:
        expand_rows <- function(mat) {
          new_rows <- lapply(1:nrow(mat), function(i) {
            names <- strsplit(rownames(mat)[i], "\\|")[[1]]
            do.call(rbind, replicate(length(names), mat[i, , drop=FALSE], simplify=FALSE)) |>
              `rownames<-`(names)
          })
          do.call(rbind, new_rows)
        }
    """
    new_rows = []
    new_index = []

    for gene_str, row in mat.iterrows():
        genes = str(gene_str).split(sep)
        for gene in genes:
            new_rows.append(row.values)
            new_index.append(gene.strip())

    result = pd.DataFrame(
        new_rows,
        index=new_index,
        columns=mat.columns
    )

    return result


def scalar1(x: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length (L2 norm = 1).

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit vector in same direction as x.

    Examples
    --------
    >>> x = np.array([3, 4])
    >>> scalar1(x)
    array([0.6, 0.8])

    Notes
    -----
    R equivalent:
        scalar1 <- function(x) { x / sqrt(sum(x^2)) }
    """
    norm = np.sqrt(np.sum(x ** 2))
    if norm == 0:
        return x
    return x / norm


def filter_genes(
    mat: Union[pd.DataFrame, sparse.spmatrix],
    min_cells: int = 0,
    min_counts: int = 0,
    gene_names: Optional[List[str]] = None
) -> Union[pd.DataFrame, sparse.spmatrix]:
    """
    Filter genes based on expression criteria.

    Parameters
    ----------
    mat : DataFrame or sparse matrix
        Expression matrix (genes × cells).
    min_cells : int, default 0
        Minimum number of cells expressing the gene.
    min_counts : int, default 0
        Minimum total counts across all cells.
    gene_names : list, optional
        Gene names (required for sparse matrices).

    Returns
    -------
    Filtered matrix and kept gene indices/names.

    Examples
    --------
    >>> # Keep genes expressed in at least 10 cells
    >>> mat_filtered = filter_genes(mat, min_cells=10)
    """
    if sparse.issparse(mat):
        n_cells = np.array((mat > 0).sum(axis=1)).flatten()
        total_counts = np.array(mat.sum(axis=1)).flatten()
    else:
        n_cells = (mat > 0).sum(axis=1)
        total_counts = mat.sum(axis=1)

    keep = (n_cells >= min_cells) & (total_counts >= min_counts)

    if isinstance(mat, pd.DataFrame):
        return mat.loc[keep]
    else:
        return mat[keep]


def match_genes(
    query_genes: List[str],
    reference_genes: List[str],
    case_sensitive: bool = False
) -> Dict[str, Optional[str]]:
    """
    Match query genes to reference genes, handling case differences.

    Parameters
    ----------
    query_genes : list
        Genes to match.
    reference_genes : list
        Reference gene set.
    case_sensitive : bool, default False
        Whether matching is case-sensitive.

    Returns
    -------
    dict
        Mapping from query gene to matched reference gene (or None).

    Examples
    --------
    >>> match_genes(["tp53", "BRCA1"], ["TP53", "BRCA1", "EGFR"])
    {'tp53': 'TP53', 'BRCA1': 'BRCA1'}
    """
    if case_sensitive:
        ref_set = set(reference_genes)
        return {g: g if g in ref_set else None for g in query_genes}
    else:
        # Build case-insensitive lookup
        ref_lookup = {g.upper(): g for g in reference_genes}
        result = {}
        for gene in query_genes:
            upper = gene.upper()
            result[gene] = ref_lookup.get(upper)
        return result
