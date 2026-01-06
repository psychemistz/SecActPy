#!/usr/bin/env python3
"""
Tests for SecActPy plotting module.

Run with: python tests/test_plotting.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt


def test_spatial_plots():
    """Test spatial visualization functions."""
    from secactpy.plotting import (
        plot_spatial,
        plot_spatial_feature,
        plot_spatial_multi,
        plot_spatial_categorical,
    )

    print("Testing spatial plots...")

    np.random.seed(42)
    n_spots = 100
    coords = np.random.rand(n_spots, 2) * 1000
    values = np.random.randn(n_spots)
    categories = np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_spots)

    # Test plot_spatial
    fig, ax = plt.subplots()
    plot_spatial(coords, values=values, ax=ax)
    plt.close()
    print("  ✓ plot_spatial")

    # Test plot_spatial_feature
    fig, ax = plt.subplots()
    plot_spatial_feature(coords, values, feature_name="Test", ax=ax)
    plt.close()
    print("  ✓ plot_spatial_feature")

    # Test plot_spatial_multi
    features = {
        'Feature1': np.random.randn(n_spots),
        'Feature2': np.random.randn(n_spots),
    }
    fig = plot_spatial_multi(coords, features, ncols=2)
    plt.close()
    print("  ✓ plot_spatial_multi")

    # Test plot_spatial_categorical
    fig, ax = plt.subplots()
    plot_spatial_categorical(coords, categories, ax=ax)
    plt.close()
    print("  ✓ plot_spatial_categorical")


def test_heatmap_plots():
    """Test heatmap functions."""
    from secactpy.plotting import (
        plot_activity_heatmap,
        plot_activity_heatmap_simple,
        plot_top_activities,
    )

    print("\nTesting heatmap plots...")

    np.random.seed(42)
    activity = pd.DataFrame(
        np.random.randn(20, 10),
        index=[f'Protein{i}' for i in range(20)],
        columns=[f'Sample{i}' for i in range(10)]
    )

    # Test plot_activity_heatmap
    g = plot_activity_heatmap(activity, figsize=(8, 6))
    plt.close()
    print("  ✓ plot_activity_heatmap")

    # Test plot_activity_heatmap_simple
    fig, ax = plt.subplots()
    plot_activity_heatmap_simple(activity, ax=ax, annot=False)
    plt.close()
    print("  ✓ plot_activity_heatmap_simple")

    # Test plot_top_activities
    fig, ax = plt.subplots()
    plot_top_activities(activity, n_top=10, ax=ax)
    plt.close()
    print("  ✓ plot_top_activities")


def test_ccc_plots():
    """Test cell-cell communication plots."""
    from secactpy.plotting import (
        plot_ccc_heatmap,
        plot_ccc_dotplot,
        plot_ccc_circle,
    )

    print("\nTesting CCC plots...")

    np.random.seed(42)
    cell_types = ['Tcell', 'Bcell', 'Macrophage', 'Fibroblast']
    interaction_matrix = pd.DataFrame(
        np.random.rand(4, 4),
        index=cell_types,
        columns=cell_types
    )

    # Test plot_ccc_heatmap
    fig, ax = plt.subplots()
    plot_ccc_heatmap(interaction_matrix, ax=ax, annot=False)
    plt.close()
    print("  ✓ plot_ccc_heatmap")

    # Test plot_ccc_dotplot
    interactions = pd.DataFrame({
        'sender': ['Tcell', 'Tcell', 'Bcell', 'Macrophage'],
        'receiver': ['Bcell', 'Macrophage', 'Tcell', 'Fibroblast'],
        'score': [0.8, 0.3, 0.5, 0.9],
        'pvalue': [0.01, 0.1, 0.05, 0.001]
    })
    fig, ax = plt.subplots()
    plot_ccc_dotplot(interactions, ax=ax)
    plt.close()
    print("  ✓ plot_ccc_dotplot")

    # Test plot_ccc_circle
    fig, ax = plt.subplots()
    plot_ccc_circle(interaction_matrix, ax=ax)
    plt.close()
    print("  ✓ plot_ccc_circle")


def test_bar_plots():
    """Test bar and lollipop plots."""
    from secactpy.plotting import (
        plot_activity_bar,
        plot_activity_lollipop,
        plot_activity_waterfall,
        plot_significance,
    )

    print("\nTesting bar plots...")

    np.random.seed(42)
    activity = pd.DataFrame(
        np.random.randn(50, 5),
        index=[f'Protein{i}' for i in range(50)],
        columns=[f'Sample{i}' for i in range(5)]
    )
    pvalues = pd.DataFrame(
        np.random.rand(50, 5),
        index=activity.index,
        columns=activity.columns
    )

    # Test plot_activity_bar
    fig, ax = plt.subplots()
    plot_activity_bar(activity['Sample0'], n_top=10, n_bottom=5, ax=ax)
    plt.close()
    print("  ✓ plot_activity_bar")

    # Test plot_activity_lollipop
    fig, ax = plt.subplots()
    plot_activity_lollipop(activity['Sample0'], n_top=10, ax=ax)
    plt.close()
    print("  ✓ plot_activity_lollipop")

    # Test plot_activity_waterfall
    fig, ax = plt.subplots()
    plot_activity_waterfall(activity['Sample0'], ax=ax)
    plt.close()
    print("  ✓ plot_activity_waterfall")

    # Test plot_significance
    fig, ax = plt.subplots()
    plot_significance(activity, pvalues, 'Sample0', n_top=15, ax=ax)
    plt.close()
    print("  ✓ plot_significance")


def test_survival_plots():
    """Test survival plots (if lifelines available)."""
    try:
        from lifelines import KaplanMeierFitter
        LIFELINES_AVAILABLE = True
    except ImportError:
        LIFELINES_AVAILABLE = False
        print("\nSkipping survival plots (lifelines not installed)")
        return

    from secactpy.plotting import (
        plot_kaplan_meier,
        plot_survival_by_activity,
        plot_forest,
    )

    print("\nTesting survival plots...")

    np.random.seed(42)
    n_samples = 100
    time = np.random.exponential(10, n_samples)
    event = np.random.binomial(1, 0.7, n_samples)
    groups = np.random.choice(['A', 'B'], n_samples)
    activity = pd.Series(np.random.randn(n_samples))

    # Test plot_kaplan_meier
    fig, ax = plt.subplots()
    plot_kaplan_meier(time, event, groups=groups, ax=ax)
    plt.close()
    print("  ✓ plot_kaplan_meier")

    # Test plot_survival_by_activity
    fig, ax = plt.subplots()
    plot_survival_by_activity(activity, time, event, ax=ax)
    plt.close()
    print("  ✓ plot_survival_by_activity")

    # Test plot_forest
    results = pd.DataFrame({
        'coef': np.random.randn(10),
        'se': np.abs(np.random.randn(10)) * 0.1 + 0.1,
        'p': np.random.rand(10)
    }, index=[f'Protein{i}' for i in range(10)])
    fig, ax = plt.subplots()
    plot_forest(results, ax=ax)
    plt.close()
    print("  ✓ plot_forest")


def main():
    """Run all plotting tests."""
    print("=" * 50)
    print("SecActPy Plotting Module Tests")
    print("=" * 50)

    test_spatial_plots()
    test_heatmap_plots()
    test_ccc_plots()
    test_bar_plots()
    test_survival_plots()

    print("\n" + "=" * 50)
    print("All plotting tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
