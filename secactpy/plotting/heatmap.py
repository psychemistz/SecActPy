"""
Heatmap visualization functions for SecActPy.

Provides functions for visualizing activity matrices as heatmaps
with clustering and annotations.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _check_deps():
    """Check required dependencies."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required: pip install matplotlib")
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")


def plot_activity_heatmap(
    activity: pd.DataFrame,
    row_cluster: bool = True,
    col_cluster: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    row_colors: Optional[pd.DataFrame] = None,
    col_colors: Optional[pd.DataFrame] = None,
    xticklabels: Union[bool, int] = True,
    yticklabels: Union[bool, int] = True,
    dendrogram_ratio: float = 0.15,
    cbar_pos: Tuple[float, float, float, float] = (0.02, 0.8, 0.03, 0.15),
    title: Optional[str] = None,
    **kwargs
) -> "sns.matrix.ClusterGrid":
    """
    Create clustered heatmap of activity scores.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    row_cluster : bool, default True
        Whether to cluster rows (proteins).
    col_cluster : bool, default True
        Whether to cluster columns (samples).
    cmap : str, default "RdBu_r"
        Colormap. RdBu_r is good for z-scores.
    center : float, default 0
        Value to center colormap at.
    vmin, vmax : float, optional
        Color scale limits. Auto-calculated if None.
    figsize : tuple, optional
        Figure size.
    row_colors : pd.DataFrame, optional
        Row annotation colors (index should match activity index).
    col_colors : pd.DataFrame, optional
        Column annotation colors (index should match activity columns).
    xticklabels : bool or int, default True
        Show x-axis labels. Int for every nth label.
    yticklabels : bool or int, default True
        Show y-axis labels. Int for every nth label.
    dendrogram_ratio : float, default 0.15
        Ratio of dendrogram size to heatmap.
    cbar_pos : tuple, default (0.02, 0.8, 0.03, 0.15)
        Colorbar position (left, bottom, width, height).
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.clustermap.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        ClusterGrid object for further customization.

    Examples
    --------
    >>> # Basic heatmap
    >>> g = plot_activity_heatmap(result['zscore'])
    >>>
    >>> # With sample annotations
    >>> col_colors = pd.DataFrame({
    ...     'Group': sample_groups.map({'A': 'red', 'B': 'blue'})
    ... })
    >>> g = plot_activity_heatmap(
    ...     result['zscore'],
    ...     col_colors=col_colors,
    ...     title="Secreted Protein Activity"
    ... )
    """
    _check_deps()

    # Auto-calculate figure size
    if figsize is None:
        n_rows, n_cols = activity.shape
        width = max(8, min(20, n_cols * 0.15 + 3))
        height = max(6, min(15, n_rows * 0.15 + 2))
        figsize = (width, height)

    # Auto-calculate color limits for z-scores
    if vmin is None and vmax is None:
        abs_max = np.nanpercentile(np.abs(activity.values), 98)
        vmin, vmax = -abs_max, abs_max

    # Create clustermap
    g = sns.clustermap(
        activity,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=col_colors,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        dendrogram_ratio=dendrogram_ratio,
        cbar_pos=cbar_pos,
        **kwargs
    )

    if title:
        g.fig.suptitle(title, y=1.02)

    return g


def plot_activity_heatmap_simple(
    activity: pd.DataFrame,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "RdBu_r",
    center: float = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annot: bool = False,
    fmt: str = ".2f",
    cbar: bool = True,
    cbar_label: str = "Activity (z-score)",
    xticklabels: Union[bool, List[str]] = True,
    yticklabels: Union[bool, List[str]] = True,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Simple heatmap without clustering (uses seaborn.heatmap).

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "RdBu_r"
        Colormap.
    center : float, default 0
        Value to center colormap at.
    vmin, vmax : float, optional
        Color scale limits.
    annot : bool, default False
        Whether to annotate cells with values.
    fmt : str, default ".2f"
        Format string for annotations.
    cbar : bool, default True
        Whether to show colorbar.
    cbar_label : str, default "Activity (z-score)"
        Colorbar label.
    xticklabels, yticklabels : bool or list
        Tick labels.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the heatmap.
    """
    _check_deps()

    if ax is None:
        n_rows, n_cols = activity.shape
        figsize = (max(6, n_cols * 0.4), max(4, n_rows * 0.3))
        fig, ax = plt.subplots(figsize=figsize)

    if vmin is None and vmax is None:
        abs_max = np.nanpercentile(np.abs(activity.values), 98)
        vmin, vmax = -abs_max, abs_max

    cbar_kws = {'label': cbar_label} if cbar else None

    sns.heatmap(
        activity,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        cbar=cbar,
        cbar_kws=cbar_kws,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        **kwargs
    )

    if title:
        ax.set_title(title)

    return ax


def plot_top_activities(
    activity: pd.DataFrame,
    n_top: int = 20,
    sample: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Heatmap of top variable or extreme activities.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    n_top : int, default 20
        Number of top proteins to show.
    sample : str, optional
        If provided, select top proteins for this sample.
        Otherwise, select by variance across all samples.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "RdBu_r"
        Colormap.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot_activity_heatmap_simple.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the heatmap.
    """
    _check_deps()

    if sample is not None:
        # Top by absolute value for specific sample
        sample_values = activity[sample].abs()
        top_proteins = sample_values.nlargest(n_top).index
        if title is None:
            title = f"Top {n_top} Activities - {sample}"
    else:
        # Top by variance across samples
        variances = activity.var(axis=1)
        top_proteins = variances.nlargest(n_top).index
        if title is None:
            title = f"Top {n_top} Variable Proteins"

    activity_subset = activity.loc[top_proteins]

    return plot_activity_heatmap_simple(
        activity_subset, ax=ax, cmap=cmap, title=title, **kwargs
    )


def plot_activity_correlation(
    activity: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    ax: Optional["plt.Axes"] = None,
    cmap: str = "RdBu_r",
    annot: bool = False,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Correlation heatmap of activity patterns.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    method : str, default "pearson"
        Correlation method.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "RdBu_r"
        Colormap.
    annot : bool, default False
        Whether to annotate cells.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the correlation heatmap.
    """
    _check_deps()

    corr = activity.T.corr(method=method)

    if ax is None:
        n = len(corr)
        figsize = (max(6, n * 0.3), max(5, n * 0.3))
        fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Activity Correlation ({method.capitalize()})"

    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        square=True,
        **kwargs
    )

    ax.set_title(title)
    return ax


def plot_sample_correlation(
    activity: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    cluster: bool = True,
    row_colors: Optional[pd.DataFrame] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    **kwargs
) -> Union["plt.Axes", "sns.matrix.ClusterGrid"]:
    """
    Correlation heatmap between samples based on activity profiles.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    method : str, default "pearson"
        Correlation method.
    cluster : bool, default True
        Whether to cluster samples.
    row_colors : pd.DataFrame, optional
        Sample annotation colors.
    cmap : str, default "viridis"
        Colormap.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to clustermap or heatmap.

    Returns
    -------
    ClusterGrid or Axes
        Plot object.
    """
    _check_deps()

    corr = activity.corr(method=method)

    if title is None:
        title = f"Sample Correlation ({method.capitalize()})"

    if cluster:
        g = sns.clustermap(
            corr,
            cmap=cmap,
            vmin=0,
            vmax=1,
            row_colors=row_colors,
            col_colors=row_colors,
            **kwargs
        )
        g.fig.suptitle(title, y=1.02)
        return g
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(corr, ax=ax, cmap=cmap, vmin=0, vmax=1, square=True, **kwargs)
        ax.set_title(title)
        return ax


def create_annotation_colors(
    annotations: pd.Series,
    palette: Optional[Union[str, Dict]] = None
) -> Tuple[pd.Series, Dict]:
    """
    Create color mapping for categorical annotations.

    Parameters
    ----------
    annotations : pd.Series
        Categorical annotations.
    palette : str or dict, optional
        Color palette name or category-to-color mapping.

    Returns
    -------
    colors : pd.Series
        Color for each sample.
    legend_map : dict
        Category to color mapping for legend.

    Examples
    --------
    >>> col_colors, legend = create_annotation_colors(
    ...     sample_groups,
    ...     palette={'Control': 'blue', 'Treatment': 'red'}
    ... )
    >>> g = plot_activity_heatmap(activity, col_colors=col_colors.to_frame())
    """
    _check_deps()

    categories = annotations.unique()
    n_cats = len(categories)

    if palette is None:
        colors = sns.color_palette("tab10", n_cats)
        color_map = dict(zip(categories, colors))
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, n_cats)
        color_map = dict(zip(categories, colors))
    else:
        color_map = palette

    color_series = annotations.map(color_map)
    return color_series, color_map


def add_legend_for_annotations(
    ax_or_fig,
    color_map: Dict,
    title: str = "Group",
    loc: str = "upper right",
    **kwargs
):
    """
    Add a legend for annotation colors.

    Parameters
    ----------
    ax_or_fig : Axes or Figure
        Where to add legend.
    color_map : dict
        Category to color mapping.
    title : str, default "Group"
        Legend title.
    loc : str, default "upper right"
        Legend location.
    **kwargs
        Additional arguments passed to legend.
    """
    _check_deps()

    patches = [
        Patch(facecolor=color, label=cat)
        for cat, color in color_map.items()
    ]

    if hasattr(ax_or_fig, 'legend'):
        ax_or_fig.legend(handles=patches, title=title, loc=loc, **kwargs)
    else:
        ax_or_fig.legend(handles=patches, title=title, loc=loc, **kwargs)
