"""
Bar and lollipop plot functions for SecActPy.

Provides functions for visualizing activity scores as bar plots,
lollipop plots, and ranked lists.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
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
        raise ImportError("matplotlib required: pip install matplotlib")


def plot_activity_bar(
    activity: Union[pd.Series, pd.DataFrame],
    sample: Optional[str] = None,
    n_top: int = 20,
    n_bottom: int = 0,
    ax: Optional["plt.Axes"] = None,
    color_pos: str = "#d62728",
    color_neg: str = "#1f77b4",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Bar plot of top (and optionally bottom) activity scores.

    Parameters
    ----------
    activity : pd.Series or pd.DataFrame
        Activity scores. If DataFrame, must specify sample.
    sample : str, optional
        Sample name if activity is DataFrame.
    n_top : int, default 20
        Number of top (positive) activities to show.
    n_bottom : int, default 0
        Number of bottom (negative) activities to show.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    color_pos : str, default "#d62728" (red)
        Color for positive values.
    color_neg : str, default "#1f77b4" (blue)
        Color for negative values.
    orientation : str, default "horizontal"
        Bar orientation: "horizontal" or "vertical".
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    **kwargs
        Additional arguments passed to plt.barh or plt.bar.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the bar plot.

    Examples
    --------
    >>> # Top 20 positive and top 10 negative activities
    >>> plot_activity_bar(
    ...     result['zscore']['Sample1'],
    ...     n_top=20,
    ...     n_bottom=10,
    ...     title="Top Activities - Sample1"
    ... )
    """
    _check_matplotlib()

    # Handle DataFrame vs Series
    if isinstance(activity, pd.DataFrame):
        if sample is None:
            raise ValueError("Must specify sample when activity is DataFrame")
        values = activity[sample]
    else:
        values = activity

    # Get top and bottom
    sorted_values = values.sort_values(ascending=False)

    if n_top > 0:
        top = sorted_values.head(n_top)
    else:
        top = pd.Series(dtype=float)

    if n_bottom > 0:
        bottom = sorted_values.tail(n_bottom)
    else:
        bottom = pd.Series(dtype=float)

    # Combine and sort by value
    selected = pd.concat([top, bottom]).sort_values(ascending=True)

    # Colors based on sign
    colors = [color_pos if v > 0 else color_neg for v in selected.values]

    # Create plot
    if ax is None:
        if orientation == "horizontal":
            figsize = (8, max(4, len(selected) * 0.3))
        else:
            figsize = (max(6, len(selected) * 0.4), 6)
        fig, ax = plt.subplots(figsize=figsize)

    if orientation == "horizontal":
        ax.barh(range(len(selected)), selected.values, color=colors, **kwargs)
        ax.set_yticks(range(len(selected)))
        ax.set_yticklabels(selected.index)
        ax.axvline(0, color='black', linewidth=0.5)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("Activity (z-score)")
    else:
        ax.bar(range(len(selected)), selected.values, color=colors, **kwargs)
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels(selected.index, rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=0.5)
        if xlabel:
            ax.set_ylabel(xlabel)
        else:
            ax.set_ylabel("Activity (z-score)")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_activity_lollipop(
    activity: Union[pd.Series, pd.DataFrame],
    sample: Optional[str] = None,
    n_top: int = 20,
    n_bottom: int = 0,
    ax: Optional["plt.Axes"] = None,
    color_pos: str = "#d62728",
    color_neg: str = "#1f77b4",
    marker_size: float = 80,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Lollipop plot of activity scores.

    Similar to bar plot but with lines and dots (cleaner for many items).

    Parameters
    ----------
    activity : pd.Series or pd.DataFrame
        Activity scores.
    sample : str, optional
        Sample name if activity is DataFrame.
    n_top : int, default 20
        Number of top activities to show.
    n_bottom : int, default 0
        Number of bottom activities to show.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    color_pos : str, default "#d62728"
        Color for positive values.
    color_neg : str, default "#1f77b4"
        Color for negative values.
    marker_size : float, default 80
        Size of the dots.
    linewidth : float, default 1.5
        Width of the stems.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the lollipop plot.
    """
    _check_matplotlib()

    # Handle DataFrame vs Series
    if isinstance(activity, pd.DataFrame):
        if sample is None:
            raise ValueError("Must specify sample when activity is DataFrame")
        values = activity[sample]
    else:
        values = activity

    # Get top and bottom
    sorted_values = values.sort_values(ascending=False)

    if n_top > 0:
        top = sorted_values.head(n_top)
    else:
        top = pd.Series(dtype=float)

    if n_bottom > 0:
        bottom = sorted_values.tail(n_bottom)
    else:
        bottom = pd.Series(dtype=float)

    selected = pd.concat([top, bottom]).sort_values(ascending=True)
    colors = [color_pos if v > 0 else color_neg for v in selected.values]

    if ax is None:
        figsize = (8, max(4, len(selected) * 0.35))
        fig, ax = plt.subplots(figsize=figsize)

    # Draw stems
    for i, (name, val) in enumerate(selected.items()):
        color = color_pos if val > 0 else color_neg
        ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=linewidth)

    # Draw dots
    ax.scatter(selected.values, range(len(selected)), c=colors, s=marker_size, zorder=10)

    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Activity (z-score)")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_activity_waterfall(
    activity: Union[pd.Series, pd.DataFrame],
    sample: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Waterfall plot showing all activities ranked by value.

    Parameters
    ----------
    activity : pd.Series or pd.DataFrame
        Activity scores.
    sample : str, optional
        Sample name if activity is DataFrame.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "RdBu_r"
        Colormap.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the waterfall plot.
    """
    _check_matplotlib()

    if isinstance(activity, pd.DataFrame):
        if sample is None:
            raise ValueError("Must specify sample when activity is DataFrame")
        values = activity[sample]
    else:
        values = activity

    sorted_values = values.sort_values(ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Color by value
    cmap_obj = plt.get_cmap(cmap)
    vmin, vmax = sorted_values.min(), sorted_values.max()
    abs_max = max(abs(vmin), abs(vmax))
    norm = plt.Normalize(-abs_max, abs_max)
    colors = [cmap_obj(norm(v)) for v in sorted_values.values]

    ax.bar(range(len(sorted_values)), sorted_values.values, color=colors, **kwargs)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Activity (z-score)")

    # Remove x-axis labels (too many)
    ax.set_xticks([])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Activity")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_activity_comparison(
    activity: pd.DataFrame,
    proteins: List[str],
    samples: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    palette: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Grouped bar plot comparing activities across samples.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    proteins : list
        Proteins to compare.
    samples : list, optional
        Samples to include. If None, uses all.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    palette : str, optional
        Color palette.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.barplot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the grouped bar plot.
    """
    _check_matplotlib()
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")

    if samples is None:
        samples = activity.columns.tolist()

    # Subset and melt data
    subset = activity.loc[proteins, samples]
    melted = subset.reset_index().melt(
        id_vars='index',
        var_name='Sample',
        value_name='Activity'
    )
    melted = melted.rename(columns={'index': 'Protein'})

    if ax is None:
        figsize = (max(6, len(proteins) * len(samples) * 0.3), 5)
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=melted,
        x='Protein', y='Activity', hue='Sample',
        ax=ax, palette=palette, **kwargs
    )

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Activity (z-score)")
    ax.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_activity_distribution(
    activity: pd.DataFrame,
    proteins: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    kind: Literal["box", "violin", "strip"] = "violin",
    palette: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Distribution plot of activities across samples.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    proteins : list, optional
        Proteins to show. If None, shows top 10 by variance.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    kind : str, default "violin"
        Plot type: "box", "violin", or "strip".
    palette : str, optional
        Color palette.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the distribution plot.
    """
    _check_matplotlib()
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")

    if proteins is None:
        # Top 10 by variance
        variances = activity.var(axis=1)
        proteins = variances.nlargest(10).index.tolist()

    subset = activity.loc[proteins]
    melted = subset.reset_index().melt(
        id_vars='index',
        var_name='Sample',
        value_name='Activity'
    )
    melted = melted.rename(columns={'index': 'Protein'})

    if ax is None:
        figsize = (max(6, len(proteins) * 0.8), 5)
        fig, ax = plt.subplots(figsize=figsize)

    if kind == "box":
        sns.boxplot(
            data=melted, x='Protein', y='Activity',
            ax=ax, palette=palette, **kwargs
        )
    elif kind == "violin":
        sns.violinplot(
            data=melted, x='Protein', y='Activity',
            ax=ax, palette=palette, **kwargs
        )
    elif kind == "strip":
        sns.stripplot(
            data=melted, x='Protein', y='Activity',
            ax=ax, palette=palette, **kwargs
        )

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("")
    ax.set_ylabel("Activity (z-score)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_significance(
    activity: pd.DataFrame,
    pvalues: pd.DataFrame,
    sample: str,
    n_top: int = 20,
    alpha: float = 0.05,
    ax: Optional["plt.Axes"] = None,
    color_sig: str = "#d62728",
    color_nonsig: str = "#7f7f7f",
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Bar plot highlighting significant activities.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity scores (proteins × samples).
    pvalues : pd.DataFrame
        P-values (proteins × samples).
    sample : str
        Sample to plot.
    n_top : int, default 20
        Number of top activities to show.
    alpha : float, default 0.05
        Significance threshold.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    color_sig : str, default "#d62728"
        Color for significant values.
    color_nonsig : str, default "#7f7f7f"
        Color for non-significant values.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the significance plot.
    """
    _check_matplotlib()

    values = activity[sample].sort_values(ascending=False).head(n_top)
    pvals = pvalues[sample].loc[values.index]

    # Sort by value for display
    sorted_idx = values.sort_values(ascending=True).index
    values = values.loc[sorted_idx]
    pvals = pvals.loc[sorted_idx]

    # Colors based on significance
    colors = [color_sig if p < alpha else color_nonsig for p in pvals]

    if ax is None:
        figsize = (8, max(4, n_top * 0.35))
        fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(values)), values.values, color=colors, **kwargs)
    ax.set_yticks(range(len(values)))

    # Add significance markers to labels
    labels = []
    for name, p in zip(values.index, pvals):
        if p < 0.001:
            labels.append(f"{name} ***")
        elif p < 0.01:
            labels.append(f"{name} **")
        elif p < alpha:
            labels.append(f"{name} *")
        else:
            labels.append(name)

    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Activity (z-score)")

    if title is None:
        title = f"Top Activities - {sample}"
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_sig, label=f'p < {alpha}'),
        Patch(facecolor=color_nonsig, label=f'p ≥ {alpha}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return ax
