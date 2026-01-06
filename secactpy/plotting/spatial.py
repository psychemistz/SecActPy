"""
Spatial visualization functions for SecActPy.

Provides functions for visualizing gene expression and activity
scores on spatial coordinates (Visium, CosMx, etc.).
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_spatial(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "viridis",
    size: float = 10,
    alpha: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    show_axis: bool = False,
    aspect: str = "equal",
    invert_y: bool = True,
    **kwargs
) -> "plt.Axes":
    """
    Plot spatial scatter of spots/cells.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_spots, 2).
    values : np.ndarray, optional
        Values to color points by. If None, all points same color.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    cmap : str, default "viridis"
        Colormap name.
    size : float, default 10
        Point size.
    alpha : float, default 1.0
        Point transparency.
    vmin, vmax : float, optional
        Color scale limits.
    title : str, optional
        Plot title.
    colorbar : bool, default True
        Whether to show colorbar (only if values provided).
    colorbar_label : str, optional
        Label for colorbar.
    show_axis : bool, default False
        Whether to show axis ticks and labels.
    aspect : str, default "equal"
        Aspect ratio.
    invert_y : bool, default True
        Whether to invert y-axis (common for image coordinates).
    **kwargs
        Additional arguments passed to plt.scatter.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.rand(100, 2) * 1000
    >>> values = np.random.randn(100)
    >>> ax = plot_spatial(coords, values, cmap="RdBu_r", title="Activity")
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Handle values
    if values is not None:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, cmap=cmap, s=size, alpha=alpha,
            vmin=vmin, vmax=vmax, **kwargs
        )
        if colorbar:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            if colorbar_label:
                cbar.set_label(colorbar_label)
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=size, alpha=alpha, **kwargs
        )

    if invert_y:
        ax.invert_yaxis()

    ax.set_aspect(aspect)

    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if title:
        ax.set_title(title)

    return ax


def plot_spatial_feature(
    coords: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str = "Feature",
    ax: Optional["plt.Axes"] = None,
    cmap: str = "viridis",
    size: float = 10,
    percentile_clip: Optional[Tuple[float, float]] = (1, 99),
    center_zero: bool = False,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Plot a single feature on spatial coordinates with automatic scaling.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_spots, 2).
    feature_values : np.ndarray
        Feature values for each spot.
    feature_name : str, default "Feature"
        Name of the feature (used for colorbar label).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "viridis"
        Colormap. Use "RdBu_r" for diverging data.
    size : float, default 10
        Point size.
    percentile_clip : tuple, optional
        Percentile range for clipping outliers. None to disable.
    center_zero : bool, default False
        Whether to center colormap at zero (for diverging data).
    title : str, optional
        Plot title. Defaults to feature_name.
    **kwargs
        Additional arguments passed to plot_spatial.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    Examples
    --------
    >>> # Plot z-score activity (centered at 0)
    >>> plot_spatial_feature(
    ...     coords, activity_zscore,
    ...     feature_name="IL6 Activity",
    ...     cmap="RdBu_r",
    ...     center_zero=True
    ... )
    """
    _check_matplotlib()

    values = feature_values.copy()

    # Clip outliers
    if percentile_clip is not None:
        vmin_pct, vmax_pct = np.percentile(values[~np.isnan(values)], percentile_clip)
        values = np.clip(values, vmin_pct, vmax_pct)
        vmin, vmax = vmin_pct, vmax_pct
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)

    # Center at zero for diverging colormaps
    if center_zero:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    if title is None:
        title = feature_name

    return plot_spatial(
        coords, values, ax=ax, cmap=cmap, size=size,
        vmin=vmin, vmax=vmax, title=title,
        colorbar_label=feature_name, **kwargs
    )


def plot_spatial_multi(
    coords: np.ndarray,
    features: Dict[str, np.ndarray],
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    size: float = 5,
    **kwargs
) -> "plt.Figure":
    """
    Plot multiple features on spatial coordinates in a grid.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_spots, 2).
    features : dict
        Dictionary mapping feature names to value arrays.
    ncols : int, default 4
        Number of columns in grid.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    cmap : str, default "viridis"
        Colormap.
    size : float, default 5
        Point size.
    **kwargs
        Additional arguments passed to plot_spatial_feature.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plots.

    Examples
    --------
    >>> features = {
    ...     "IL6": activity.loc["IL6"].values,
    ...     "TNF": activity.loc["TNF"].values,
    ...     "IFNG": activity.loc["IFNG"].values,
    ... }
    >>> fig = plot_spatial_multi(coords, features, ncols=3)
    """
    _check_matplotlib()

    n_features = len(features)
    nrows = int(np.ceil(n_features / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for ax, (name, values) in zip(axes, features.items()):
        plot_spatial_feature(
            coords, values, feature_name=name,
            ax=ax, cmap=cmap, size=size, **kwargs
        )

    # Hide empty axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def plot_spatial_categorical(
    coords: np.ndarray,
    categories: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    size: float = 10,
    alpha: float = 1.0,
    legend: bool = True,
    legend_loc: str = "right",
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Plot categorical labels on spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array of shape (n_spots, 2).
    categories : np.ndarray
        Categorical labels for each spot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    palette : str, dict, or list, optional
        Color palette. Can be seaborn palette name, dict mapping
        categories to colors, or list of colors.
    size : float, default 10
        Point size.
    alpha : float, default 1.0
        Point transparency.
    legend : bool, default True
        Whether to show legend.
    legend_loc : str, default "right"
        Legend location: "right", "bottom", or standard matplotlib loc.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plt.scatter.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    Examples
    --------
    >>> plot_spatial_categorical(
    ...     coords, cell_types,
    ...     palette="tab20",
    ...     title="Cell Types"
    ... )
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique categories
    unique_cats = np.unique(categories[~pd.isna(categories)])
    n_cats = len(unique_cats)

    # Set up colors
    if palette is None:
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("tab20", n_cats)
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_cats))
        color_map = dict(zip(unique_cats, colors))
    elif isinstance(palette, str):
        if SEABORN_AVAILABLE:
            colors = sns.color_palette(palette, n_cats)
        else:
            cmap = plt.get_cmap(palette)
            colors = [cmap(i / n_cats) for i in range(n_cats)]
        color_map = dict(zip(unique_cats, colors))
    elif isinstance(palette, dict):
        color_map = palette
    else:  # list
        color_map = dict(zip(unique_cats, palette))

    # Plot each category
    for cat in unique_cats:
        mask = categories == cat
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[cat]], s=size, alpha=alpha,
            label=cat, **kwargs
        )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if legend:
        if legend_loc == "right":
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=0., frameon=False
            )
        elif legend_loc == "bottom":
            ax.legend(
                bbox_to_anchor=(0.5, -0.1), loc='upper center',
                ncol=min(5, n_cats), frameon=False
            )
        else:
            ax.legend(loc=legend_loc, frameon=False)

    if title:
        ax.set_title(title)

    return ax


def plot_spatial_comparison(
    coords: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    name1: str = "Feature 1",
    name2: str = "Feature 2",
    figsize: Tuple[float, float] = (12, 5),
    cmap: str = "viridis",
    **kwargs
) -> "plt.Figure":
    """
    Side-by-side comparison of two features on spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array.
    values1, values2 : np.ndarray
        Values for the two features.
    name1, name2 : str
        Names of the features.
    figsize : tuple, default (12, 5)
        Figure size.
    cmap : str, default "viridis"
        Colormap.
    **kwargs
        Additional arguments passed to plot_spatial_feature.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with both plots.
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_spatial_feature(coords, values1, name1, ax=ax1, cmap=cmap, **kwargs)
    plot_spatial_feature(coords, values2, name2, ax=ax2, cmap=cmap, **kwargs)

    plt.tight_layout()
    return fig


def plot_spatial_with_image(
    coords: np.ndarray,
    image: np.ndarray,
    values: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "viridis",
    size: float = 30,
    alpha: float = 0.8,
    image_alpha: float = 1.0,
    **kwargs
) -> "plt.Axes":
    """
    Overlay spatial data on H&E or tissue image.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates array (in image space or scaled).
    image : np.ndarray
        Background image (H × W × 3 for RGB).
    values : np.ndarray, optional
        Values to color spots by.
    scale_factor : float, default 1.0
        Scale factor to convert coordinates to image pixels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "viridis"
        Colormap for values.
    size : float, default 30
        Spot size.
    alpha : float, default 0.8
        Spot transparency.
    image_alpha : float, default 1.0
        Background image transparency.
    **kwargs
        Additional arguments passed to plot_spatial.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.

    Examples
    --------
    >>> # Load Visium data with image
    >>> image = plt.imread("tissue_hires_image.png")
    >>> plot_spatial_with_image(
    ...     coords * scalefactors['tissue_hires_scalef'],
    ...     image,
    ...     values=activity.loc["IL6"].values
    ... )
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Show background image
    ax.imshow(image, alpha=image_alpha)

    # Scale coordinates
    scaled_coords = coords * scale_factor

    # Overlay spots
    plot_spatial(
        scaled_coords, values, ax=ax, cmap=cmap,
        size=size, alpha=alpha, invert_y=False, **kwargs
    )

    return ax
