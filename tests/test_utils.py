#!/usr/bin/env python3
"""
Tests for SecActPy utility and spatial modules.

Run with: python -m pytest tests/test_utils.py -v
Or directly: python tests/test_utils.py
"""

import numpy as np
import pandas as pd
from scipy import sparse


def test_sparse_utilities():
    """Test sparse matrix utilities."""
    from secactpy.utils import (
        sweep_sparse,
        normalize_sparse,
        sparse_column_scale,
        sparse_log1p,
    )

    print("Testing sparse utilities...")

    # Test sweep_sparse row subtraction
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    m = sparse.csr_matrix(data)
    row_means = np.array([2, 5, 8], dtype=float)
    result = sweep_sparse(m, margin=0, stats=row_means, fun="subtract")
    expected = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    assert np.allclose(result.toarray(), expected), "sweep_sparse row failed"
    print("  ✓ sweep_sparse row subtraction")

    # Test column operation
    data = np.array([[2, 4, 6], [4, 8, 12]], dtype=float)
    m = sparse.csr_matrix(data)
    col_sums = np.array([6, 12, 18], dtype=float)
    result = sweep_sparse(m, margin=1, stats=col_sums, fun="divide")
    expected = np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 2 / 3, 2 / 3]], dtype=float)
    assert np.allclose(result.toarray(), expected), "sweep_sparse col failed"
    print("  ✓ sweep_sparse column division")

    # Test normalize_sparse L1 rows
    data = np.array([[1, 2, 3], [4, 0, 6]], dtype=float)
    m = sparse.csr_matrix(data)
    result = normalize_sparse(m, method="l1", axis=1)
    row_sums = np.array(result.sum(axis=1)).flatten()
    assert np.allclose(row_sums, [1, 1]), f"normalize_sparse failed: {row_sums}"
    print("  ✓ normalize_sparse L1 rows")

    # Test column normalization
    result = normalize_sparse(m, method="sum", axis=0)
    col_sums = np.array(result.sum(axis=0)).flatten()
    assert np.allclose(col_sums, [1, 1, 1]), f"normalize_sparse cols failed: {col_sums}"
    print("  ✓ normalize_sparse sum columns")

    # Test sparse_column_scale CPM
    data = np.array([[100, 200], [200, 300], [700, 500]], dtype=float)
    m = sparse.csr_matrix(data)
    result = sparse_column_scale(m, scale_factor=1e6)
    col_sums = np.array(result.sum(axis=0)).flatten()
    assert np.allclose(col_sums, [1e6, 1e6]), "sparse_column_scale failed"
    print("  ✓ sparse_column_scale CPM")

    # Test sparse_log1p
    data = np.array([[0, 1, 3], [7, 0, 15]], dtype=float)
    m = sparse.csr_matrix(data)
    result = sparse_log1p(m, base=2)
    expected = np.log2(data + 1)
    assert np.allclose(result.toarray(), expected), "sparse_log1p failed"
    print("  ✓ sparse_log1p")


def test_gene_utilities():
    """Test gene symbol utilities."""
    from secactpy.utils import rm_duplicates, expand_rows, scalar1, match_genes

    print("\nTesting gene utilities...")

    # Test rm_duplicates
    df = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]], index=["A", "B", "A"], columns=["S1", "S2"]
    )
    result = rm_duplicates(df)
    assert len(result) == 2, "rm_duplicates length failed"
    assert result.loc["A", "S1"] == 5, "rm_duplicates wrong row kept"
    print("  ✓ rm_duplicates")

    # Test expand_rows
    df = pd.DataFrame([[1, 2], [3, 4]], index=["A|B", "C"], columns=["S1", "S2"])
    result = expand_rows(df, sep="|")
    assert len(result) == 3, "expand_rows length failed"
    assert list(result.index) == ["A", "B", "C"], "expand_rows index failed"
    print("  ✓ expand_rows")

    # Test scalar1
    x = np.array([3, 4])
    result = scalar1(x)
    assert np.isclose(np.linalg.norm(result), 1.0), "scalar1 norm failed"
    print("  ✓ scalar1")

    # Test match_genes
    query = ["tp53", "BRCA1", "egfr"]
    reference = ["TP53", "BRCA1", "EGFR", "MYC"]
    result = match_genes(query, reference, case_sensitive=False)
    assert result["tp53"] == "TP53", "match_genes failed"
    print("  ✓ match_genes")


def test_spatial_utilities():
    """Test spatial weight calculation."""
    from secactpy.spatial import (
        calc_spatial_weights,
        row_normalize_weights,
        spatial_lag,
    )

    print("\nTesting spatial utilities...")

    # Test calc_spatial_weights gaussian
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    W = calc_spatial_weights(coords, radius=2.0, sigma=1.0, method="gaussian")
    assert sparse.issparse(W), "calc_spatial_weights not sparse"
    assert np.allclose(W.toarray(), W.T.toarray()), "calc_spatial_weights not symmetric"
    assert np.all(W.diagonal() == 0), "calc_spatial_weights diagonal not zero"
    print("  ✓ calc_spatial_weights gaussian")

    # Test binary weights
    coords = np.array([[0, 0], [1, 0], [10, 10]], dtype=float)
    W = calc_spatial_weights(coords, radius=2.0, method="binary")
    assert W[0, 1] == 1, "binary weights failed - neighbors"
    assert W[0, 2] == 0, "binary weights failed - distant"
    print("  ✓ calc_spatial_weights binary")

    # Test row_normalize_weights
    W = sparse.random(10, 10, density=0.3, format="csr")
    W = W + W.T
    W_norm = row_normalize_weights(W)
    row_sums = np.array(W_norm.sum(axis=1)).flatten()
    for i, s in enumerate(row_sums):
        if W[i].nnz > 0:
            assert np.isclose(s, 1.0), f"row {i} does not sum to 1"
    print("  ✓ row_normalize_weights")

    # Test spatial_lag
    W = sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    )
    values = np.array([1, 2, 3], dtype=float)
    result = spatial_lag(values, W, normalize=True)
    expected = np.array([2, 2, 2], dtype=float)
    assert np.allclose(result, expected), "spatial_lag failed"
    print("  ✓ spatial_lag")


def main():
    """Run all tests."""
    print("=" * 50)
    print("SecActPy Utility Module Tests")
    print("=" * 50)

    test_sparse_utilities()
    test_gene_utilities()
    test_spatial_utilities()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
