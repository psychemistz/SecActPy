"""
SecActPy plotting functions.

This module provides visualization tools for:
- Spatial transcriptomics data
- Activity heatmaps
- Cell-cell communication networks
- Bar and lollipop plots
- Survival analysis

Dependencies
------------
Required: matplotlib
Recommended: seaborn
For survival plots: lifelines

Examples
--------
>>> from secactpy.plotting import plot_activity_heatmap, plot_spatial_feature
>>> from secactpy.plotting import plot_ccc_heatmap, plot_activity_bar
>>>
>>> # Activity heatmap with clustering
>>> g = plot_activity_heatmap(result['zscore'])
>>>
>>> # Spatial visualization
>>> plot_spatial_feature(coords, activity.loc['IL6'], cmap='RdBu_r')
>>>
>>> # Top activities bar plot
>>> plot_activity_bar(activity['Sample1'], n_top=20, n_bottom=10)
"""

# Check for matplotlib availability
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Spatial visualization
from .spatial import (
    plot_spatial,
    plot_spatial_feature,
    plot_spatial_multi,
    plot_spatial_categorical,
    plot_spatial_comparison,
    plot_spatial_with_image,
)

# Heatmaps
from .heatmap import (
    plot_activity_heatmap,
    plot_activity_heatmap_simple,
    plot_top_activities,
    plot_activity_correlation,
    plot_sample_correlation,
    create_annotation_colors,
    add_legend_for_annotations,
)

# Cell-cell communication
from .ccc import (
    plot_ccc_heatmap,
    plot_ccc_dotplot,
    plot_ccc_circle,
    plot_ccc_chord,
    plot_ccc_sankey,
    plot_lr_pairs,
)

# Bar plots
from .bar import (
    plot_activity_bar,
    plot_activity_lollipop,
    plot_activity_waterfall,
    plot_activity_comparison,
    plot_activity_distribution,
    plot_significance,
)

# Survival plots
from .survival import (
    plot_kaplan_meier,
    plot_survival_by_activity,
    plot_forest,
    plot_survival_volcano,
    plot_survival_heatmap,
)

__all__ = [
    # Availability flags
    "MATPLOTLIB_AVAILABLE",
    "SEABORN_AVAILABLE",
    # Spatial
    "plot_spatial",
    "plot_spatial_feature",
    "plot_spatial_multi",
    "plot_spatial_categorical",
    "plot_spatial_comparison",
    "plot_spatial_with_image",
    # Heatmaps
    "plot_activity_heatmap",
    "plot_activity_heatmap_simple",
    "plot_top_activities",
    "plot_activity_correlation",
    "plot_sample_correlation",
    "create_annotation_colors",
    "add_legend_for_annotations",
    # Cell-cell communication
    "plot_ccc_heatmap",
    "plot_ccc_dotplot",
    "plot_ccc_circle",
    "plot_ccc_chord",
    "plot_ccc_sankey",
    "plot_lr_pairs",
    # Bar plots
    "plot_activity_bar",
    "plot_activity_lollipop",
    "plot_activity_waterfall",
    "plot_activity_comparison",
    "plot_activity_distribution",
    "plot_significance",
    # Survival
    "plot_kaplan_meier",
    "plot_survival_by_activity",
    "plot_forest",
    "plot_survival_volcano",
    "plot_survival_heatmap",
]
