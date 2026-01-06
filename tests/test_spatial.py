#!/usr/bin/env python3
"""
Tests for SecActPy spatial analysis modules.

Run with: python tests/test_spatial.py
"""

import numpy as np
import pandas as pd
from scipy import sparse


def test_colocalization():
    """Test colocalization analysis functions."""
    from secactpy.spatial import (
        calc_colocalization,
        calc_neighborhood_enrichment,
        calc_morans_i,
        calc_getis_ord_g,
    )

    print("Testing colocalization functions...")

    np.random.seed(42)
    n_spots = 200

    # Create clustered spatial pattern
    coords = np.zeros((n_spots, 2))
    cell_types = np.empty(n_spots, dtype=object)

    # Cluster 1: TypeA cells
    coords[:50] = np.random.randn(50, 2) * 20 + [100, 100]
    cell_types[:50] = 'TypeA'

    # Cluster 2: TypeB cells
    coords[50:100] = np.random.randn(50, 2) * 20 + [200, 100]
    cell_types[50:100] = 'TypeB'

    # Mixed region: TypeA and TypeB
    coords[100:150] = np.random.randn(50, 2) * 30 + [150, 100]
    cell_types[100:150] = np.random.choice(['TypeA', 'TypeB'], 50)

    # Scattered TypeC
    coords[150:] = np.random.rand(50, 2) * 300
    cell_types[150:] = 'TypeC'

    # Test calc_colocalization
    result = calc_colocalization(
        cell_types, coords,
        radius=50,
        method="jaccard"
    )
    assert 'score' in result
    assert result['score'].shape == (3, 3)  # 3 cell types
    print("  ✓ calc_colocalization")

    # Test with permutation
    result_perm = calc_colocalization(
        cell_types, coords,
        radius=50,
        method="jaccard",
        n_permutations=100,
        seed=42
    )
    assert 'pvalue' in result_perm
    assert 'zscore' in result_perm
    print("  ✓ calc_colocalization with permutation")

    # Test calc_neighborhood_enrichment
    enrich = calc_neighborhood_enrichment(
        cell_types, coords,
        n_neighbors=10,
        n_permutations=100,
        seed=42
    )
    assert 'zscore' in enrich
    assert 'count' in enrich
    print("  ✓ calc_neighborhood_enrichment")

    # Test calc_morans_i
    values = np.zeros(n_spots)
    values[:50] = 1  # High values in cluster 1
    values[50:100] = -1  # Low values in cluster 2

    morans = calc_morans_i(values, coords, radius=50, n_permutations=99, seed=42)
    assert 'I' in morans
    assert morans['I'] > 0  # Should be positive (clustered pattern)
    print("  ✓ calc_morans_i")

    # Test calc_getis_ord_g
    g_star = calc_getis_ord_g(values, coords, radius=50)
    assert len(g_star) == n_spots
    # High values in cluster 1 should have positive G*
    assert np.mean(g_star[:50]) > np.mean(g_star[50:100])
    print("  ✓ calc_getis_ord_g")


def test_interface():
    """Test interface detection functions."""
    from secactpy.spatial import (
        detect_interface,
        analyze_interface_activity,
        extract_interface_profile,
    )

    print("\nTesting interface functions...")

    np.random.seed(42)
    n_spots = 300

    # Create tumor-stroma spatial pattern
    coords = np.zeros((n_spots, 2))
    cell_types = np.empty(n_spots, dtype=object)

    # Tumor region (left side)
    coords[:100] = np.random.rand(100, 2) * [100, 200]
    cell_types[:100] = 'Tumor'

    # Stroma region (right side)
    coords[100:200] = np.random.rand(100, 2) * [100, 200] + [150, 0]
    cell_types[100:200] = 'Fibroblast'

    # Interface region (middle)
    coords[200:] = np.random.rand(100, 2) * [50, 200] + [100, 0]
    cell_types[200:] = np.random.choice(['Tumor', 'Fibroblast'], 100)

    # Test detect_interface
    interface = detect_interface(
        cell_types, coords,
        tumor_types=['Tumor'],
        stroma_types=['Fibroblast'],
        radius=30,
        method='neighbor'
    )
    assert 'is_interface' in interface
    assert 'interface_score' in interface
    assert 'compartment' in interface
    assert np.sum(interface['is_interface']) > 0
    print("  ✓ detect_interface")

    # Test analyze_interface_activity
    activity = pd.DataFrame(
        np.random.randn(10, n_spots),
        index=[f'Protein{i}' for i in range(10)],
        columns=range(n_spots)
    )

    # Make some proteins interface-enriched
    activity.iloc[0, interface['is_interface']] += 2

    results = analyze_interface_activity(activity, interface, test='wilcoxon')
    assert 'protein' in results.columns
    assert 'pvalue' in results.columns
    assert 'log2fc' in results.columns
    print("  ✓ analyze_interface_activity")

    # Test extract_interface_profile
    profile = extract_interface_profile(activity, coords, interface, n_bins=5)
    assert 'protein' in profile.columns
    assert 'distance_to_interface' in profile.columns
    assert 'mean_activity' in profile.columns
    print("  ✓ extract_interface_profile")


def test_lr_network():
    """Test ligand-receptor network functions."""
    from secactpy.spatial import (
        load_lr_database,
        score_lr_interactions,
        score_lr_spatial,
        aggregate_pathway_scores,
        identify_significant_interactions,
    )

    print("\nTesting L-R network functions...")

    np.random.seed(42)
    n_spots = 100
    n_genes = 50

    # Create test data
    coords = np.random.rand(n_spots, 2) * 200
    cell_types = np.random.choice(['Tcell', 'Macrophage', 'Fibroblast'], n_spots)

    gene_names = ['IL6', 'IL6R', 'TNF', 'TNFRSF1A', 'IFNG', 'IFNGR1'] + \
                 [f'Gene{i}' for i in range(n_genes - 6)]

    expression = pd.DataFrame(
        np.random.rand(n_genes, n_spots) * 10,
        index=gene_names,
        columns=range(n_spots)
    )

    # Test load_lr_database
    lr_pairs = load_lr_database()
    assert 'ligand' in lr_pairs.columns
    assert 'receptor' in lr_pairs.columns
    assert len(lr_pairs) > 0
    print("  ✓ load_lr_database")

    # Test score_lr_interactions
    lr_result = score_lr_interactions(
        expression, coords, cell_types, lr_pairs,
        radius=50,
        method='product'
    )
    assert 'scores' in lr_result
    assert 'lr_summary' in lr_result
    assert len(lr_result['scores']) > 0
    print("  ✓ score_lr_interactions")

    # Test score_lr_spatial
    spatial_scores = score_lr_spatial(expression, coords, lr_pairs, radius=50)
    assert spatial_scores.shape[1] == n_spots
    print("  ✓ score_lr_spatial")

    # Test aggregate_pathway_scores
    pathway_scores = aggregate_pathway_scores(lr_result['scores'], lr_pairs)
    assert len(pathway_scores) > 0
    print("  ✓ aggregate_pathway_scores")

    # Test identify_significant_interactions (with mock pvalues)
    # Create mock result with pvalues
    mock_result = {
        'scores': lr_result['scores'],
        'pvalue': {}
    }
    for pair_name, df in lr_result['scores'].items():
        mock_result['pvalue'][pair_name] = pd.DataFrame(
            np.random.rand(*df.shape) * 0.1,  # Low p-values
            index=df.index,
            columns=df.columns
        )

    significant = identify_significant_interactions(mock_result, pvalue_threshold=0.05)
    assert 'sender' in significant.columns
    assert 'receiver' in significant.columns
    assert 'lr_pair' in significant.columns
    print("  ✓ identify_significant_interactions")


def test_ripley_k():
    """Test Ripley's K function."""
    from secactpy.spatial import calc_ripley_k, calc_cross_ripley_k

    print("\nTesting Ripley's K functions...")

    np.random.seed(42)
    n_spots = 200
    coords = np.zeros((n_spots, 2))
    cell_types = np.empty(n_spots, dtype=object)

    # Clustered pattern for TypeA
    coords[:100] = np.random.randn(100, 2) * 20 + [100, 100]
    cell_types[:100] = 'TypeA'

    # Random pattern for TypeB
    coords[100:] = np.random.rand(100, 2) * 200
    cell_types[100:] = 'TypeB'

    radii = np.linspace(10, 100, 10)

    # Test calc_ripley_k
    ripley_a = calc_ripley_k(cell_types, coords, radii, cell_type='TypeA')
    assert 'K' in ripley_a.columns
    assert 'L' in ripley_a.columns
    # Clustered pattern should have positive L
    assert ripley_a['L'].iloc[-1] > 0
    print("  ✓ calc_ripley_k")

    ripley_b = calc_ripley_k(cell_types, coords, radii, cell_type='TypeB')
    # Random pattern should have L near 0
    assert abs(ripley_b['L'].iloc[-1]) < ripley_a['L'].iloc[-1]
    print("  ✓ calc_ripley_k (random pattern)")

    # Test calc_cross_ripley_k
    cross_k = calc_cross_ripley_k(cell_types, coords, radii, 'TypeA', 'TypeB')
    assert 'K' in cross_k.columns
    print("  ✓ calc_cross_ripley_k")


def main():
    """Run all spatial analysis tests."""
    print("=" * 50)
    print("SecActPy Spatial Analysis Module Tests")
    print("=" * 50)

    test_colocalization()
    test_interface()
    test_lr_network()
    test_ripley_k()

    print("\n" + "=" * 50)
    print("All spatial analysis tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
