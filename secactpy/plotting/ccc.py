"""
Cell-cell communication visualization functions for SecActPy.

Provides functions for visualizing cell-cell interaction patterns
including heatmaps, circle plots, chord diagrams, and dot plots.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Circle, Wedge
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors
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


def plot_ccc_heatmap(
    interaction_matrix: pd.DataFrame,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "Reds",
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = 0,
    vmax: Optional[float] = None,
    cbar_label: str = "Interaction Score",
    title: Optional[str] = None,
    xlabel: str = "Receiver",
    ylabel: str = "Sender",
    **kwargs
) -> "plt.Axes":
    """
    Heatmap of cell-cell communication scores.

    Parameters
    ----------
    interaction_matrix : pd.DataFrame
        Interaction scores with senders as rows, receivers as columns.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "Reds"
        Colormap.
    annot : bool, default True
        Whether to annotate cells with values.
    fmt : str, default ".2f"
        Format string for annotations.
    vmin : float, default 0
        Minimum color scale value.
    vmax : float, optional
        Maximum color scale value.
    cbar_label : str, default "Interaction Score"
        Colorbar label.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    **kwargs
        Additional arguments passed to seaborn.heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the heatmap.

    Examples
    --------
    >>> # Sender → Receiver interaction matrix
    >>> plot_ccc_heatmap(
    ...     interaction_scores,
    ...     title="IL6 Signaling",
    ...     cmap="YlOrRd"
    ... )
    """
    _check_deps()
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")

    if ax is None:
        n_rows, n_cols = interaction_matrix.shape
        figsize = (max(6, n_cols * 0.6 + 2), max(5, n_rows * 0.5 + 1))
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        interaction_matrix,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label},
        **kwargs
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_ccc_dotplot(
    interactions: pd.DataFrame,
    x: str = "receiver",
    y: str = "sender",
    size: str = "pvalue",
    color: str = "score",
    size_range: Tuple[float, float] = (20, 200),
    cmap: str = "Reds",
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
    size_legend_title: str = "-log10(p)",
    color_legend_title: str = "Score",
    **kwargs
) -> "plt.Axes":
    """
    Dot plot for cell-cell communication results.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame with interaction results.
        Must contain columns for x, y, size, and color variables.
    x : str, default "receiver"
        Column for x-axis categories.
    y : str, default "sender"
        Column for y-axis categories.
    size : str, default "pvalue"
        Column for dot size. P-values are auto-transformed to -log10.
    color : str, default "score"
        Column for dot color.
    size_range : tuple, default (20, 200)
        Min and max dot sizes.
    cmap : str, default "Reds"
        Colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    size_legend_title : str
        Title for size legend.
    color_legend_title : str
        Title for color legend.
    **kwargs
        Additional arguments passed to plt.scatter.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the dot plot.

    Examples
    --------
    >>> interactions = pd.DataFrame({
    ...     'sender': ['A', 'A', 'B', 'B'],
    ...     'receiver': ['B', 'C', 'A', 'C'],
    ...     'score': [0.8, 0.3, 0.5, 0.9],
    ...     'pvalue': [0.01, 0.1, 0.05, 0.001]
    ... })
    >>> plot_ccc_dotplot(interactions)
    """
    _check_deps()

    df = interactions.copy()

    # Transform p-values to -log10
    if size == "pvalue" and "pvalue" in df.columns:
        df["_size"] = -np.log10(df["pvalue"].clip(lower=1e-10))
        size_col = "_size"
    else:
        size_col = size

    # Create categorical codes for x and y
    x_cats = pd.Categorical(df[x])
    y_cats = pd.Categorical(df[y])
    df["_x"] = x_cats.codes
    df["_y"] = y_cats.codes

    if ax is None:
        n_x = len(x_cats.categories)
        n_y = len(y_cats.categories)
        figsize = (max(6, n_x * 0.8 + 2), max(5, n_y * 0.6 + 1))
        fig, ax = plt.subplots(figsize=figsize)

    # Normalize sizes
    size_values = df[size_col].values
    size_norm = (size_values - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
    sizes = size_range[0] + size_norm * (size_range[1] - size_range[0])

    scatter = ax.scatter(
        df["_x"], df["_y"],
        c=df[color], s=sizes,
        cmap=cmap, edgecolors='gray', linewidth=0.5,
        **kwargs
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color_legend_title)

    # Set tick labels
    ax.set_xticks(range(len(x_cats.categories)))
    ax.set_xticklabels(x_cats.categories, rotation=45, ha='right')
    ax.set_yticks(range(len(y_cats.categories)))
    ax.set_yticklabels(y_cats.categories)

    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())

    if title:
        ax.set_title(title)

    # Size legend
    size_handles = []
    for s_val in [size_values.min(), size_values.mean(), size_values.max()]:
        s_norm = (s_val - size_values.min()) / (size_values.max() - size_values.min() + 1e-10)
        s_size = size_range[0] + s_norm * (size_range[1] - size_range[0])
        size_handles.append(
            plt.scatter([], [], s=s_size, c='gray', label=f'{s_val:.2f}')
        )

    ax.legend(
        handles=size_handles,
        title=size_legend_title,
        loc='upper left',
        bbox_to_anchor=(1.15, 1),
        frameon=False
    )

    plt.tight_layout()
    return ax


def plot_ccc_circle(
    interaction_matrix: pd.DataFrame,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "Reds",
    node_size: float = 0.15,
    edge_width_range: Tuple[float, float] = (1, 10),
    node_colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    show_self_loops: bool = False,
    min_weight: float = 0,
    **kwargs
) -> "plt.Axes":
    """
    Circle plot for cell-cell communication network.

    Nodes represent cell types, edges represent interactions.
    Edge width and color represent interaction strength.

    Parameters
    ----------
    interaction_matrix : pd.DataFrame
        Interaction scores (senders × receivers).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "Reds"
        Colormap for edge colors.
    node_size : float, default 0.15
        Node radius (relative to plot).
    edge_width_range : tuple, default (1, 10)
        Min and max edge widths.
    node_colors : dict, optional
        Mapping of cell type to color.
    title : str, optional
        Plot title.
    show_self_loops : bool, default False
        Whether to show self-interactions.
    min_weight : float, default 0
        Minimum weight to show edge.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the circle plot.
    """
    _check_deps()

    cell_types = list(interaction_matrix.index)
    n_types = len(cell_types)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Node positions in a circle
    angles = np.linspace(0, 2 * np.pi, n_types, endpoint=False)
    positions = {
        ct: (np.cos(a), np.sin(a))
        for ct, a in zip(cell_types, angles)
    }

    # Node colors
    if node_colors is None:
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("husl", n_types)
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_types))
        node_colors = dict(zip(cell_types, colors))

    # Get colormap for edges
    cmap_obj = plt.get_cmap(cmap)

    # Normalize weights for edge properties
    weights = interaction_matrix.values.flatten()
    weights = weights[weights > min_weight]
    if len(weights) > 0:
        w_min, w_max = weights.min(), weights.max()
    else:
        w_min, w_max = 0, 1

    # Draw edges
    for sender in cell_types:
        for receiver in cell_types:
            if sender == receiver and not show_self_loops:
                continue

            weight = interaction_matrix.loc[sender, receiver]
            if weight <= min_weight:
                continue

            # Normalize weight
            w_norm = (weight - w_min) / (w_max - w_min + 1e-10)
            edge_width = edge_width_range[0] + w_norm * (edge_width_range[1] - edge_width_range[0])
            edge_color = cmap_obj(w_norm)

            # Draw curved arrow
            start = positions[sender]
            end = positions[receiver]

            # Calculate control point for curve
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            # Push control point toward center for curve
            ctrl = (mid[0] * 0.3, mid[1] * 0.3)

            arrow = FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad=0.2",
                arrowstyle="-|>",
                mutation_scale=15,
                lw=edge_width,
                color=edge_color,
                alpha=0.7
            )
            ax.add_patch(arrow)

    # Draw nodes
    for ct, pos in positions.items():
        circle = Circle(
            pos, node_size,
            facecolor=node_colors[ct],
            edgecolor='black',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)

        # Label
        label_pos = (pos[0] * 1.3, pos[1] * 1.3)
        ha = 'left' if pos[0] > 0 else 'right' if pos[0] < 0 else 'center'
        ax.text(
            label_pos[0], label_pos[1], ct,
            ha=ha, va='center',
            fontsize=10, fontweight='bold'
        )

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(w_min, w_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Interaction Score')

    return ax


def plot_ccc_chord(
    interaction_matrix: pd.DataFrame,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "tab20",
    alpha: float = 0.7,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Chord diagram for cell-cell communication.

    Parameters
    ----------
    interaction_matrix : pd.DataFrame
        Interaction scores (symmetric or asymmetric).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "tab20"
        Colormap for cell types.
    alpha : float, default 0.7
        Chord transparency.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the chord diagram.

    Notes
    -----
    For better chord diagrams, consider using the 'chord' package:
    pip install chord
    """
    _check_deps()

    cell_types = list(interaction_matrix.index)
    n_types = len(cell_types)
    values = interaction_matrix.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate arc sizes based on total interactions
    totals = values.sum(axis=0) + values.sum(axis=1)
    totals = totals / totals.sum() * 360  # Convert to degrees

    # Get colors
    if SEABORN_AVAILABLE:
        colors = sns.color_palette(cmap, n_types)
    else:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / n_types) for i in range(n_types)]

    # Draw arcs
    start_angle = 0
    arc_positions = {}

    for i, ct in enumerate(cell_types):
        arc_extent = totals[i]
        end_angle = start_angle + arc_extent

        arc_positions[ct] = (start_angle, end_angle)

        # Draw outer arc
        wedge = Wedge(
            (0, 0), 1, start_angle, end_angle,
            width=0.1, facecolor=colors[i], edgecolor='white'
        )
        ax.add_patch(wedge)

        # Label
        mid_angle = np.radians((start_angle + end_angle) / 2)
        label_x = 1.15 * np.cos(mid_angle)
        label_y = 1.15 * np.sin(mid_angle)
        rotation = (start_angle + end_angle) / 2
        if 90 < rotation < 270:
            rotation += 180
        ax.text(
            label_x, label_y, ct,
            ha='center', va='center',
            fontsize=9, rotation=rotation - 90
        )

        start_angle = end_angle + 2  # Small gap

    # Draw chords (simplified version)
    # Note: Full chord implementation is complex; this is simplified
    for i, sender in enumerate(cell_types):
        for j, receiver in enumerate(cell_types):
            if i >= j:  # Only draw once for each pair
                continue

            weight = values[i, j] + values[j, i]
            if weight <= 0:
                continue

            # Simple line connection (full chords need bezier curves)
            s_mid = np.radians((arc_positions[sender][0] + arc_positions[sender][1]) / 2)
            r_mid = np.radians((arc_positions[receiver][0] + arc_positions[receiver][1]) / 2)

            ax.plot(
                [0.9 * np.cos(s_mid), 0.9 * np.cos(r_mid)],
                [0.9 * np.sin(s_mid), 0.9 * np.sin(r_mid)],
                color=colors[i], alpha=alpha * weight / values.max(),
                lw=2
            )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', y=1.05)

    return ax


def plot_ccc_sankey(
    interactions: pd.DataFrame,
    sender_col: str = "sender",
    receiver_col: str = "receiver",
    weight_col: str = "score",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> "plt.Figure":
    """
    Sankey diagram for cell-cell communication flow.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame with sender, receiver, and weight columns.
    sender_col : str, default "sender"
        Column name for senders.
    receiver_col : str, default "receiver"
        Column name for receivers.
    weight_col : str, default "score"
        Column name for weights.
    title : str, optional
        Plot title.
    figsize : tuple, default (10, 8)
        Figure size.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with Sankey diagram.

    Notes
    -----
    For better Sankey diagrams, consider plotly:
        import plotly.graph_objects as go
    """
    _check_deps()

    try:
        from matplotlib.sankey import Sankey
    except ImportError:
        raise ImportError("Sankey not available in this matplotlib version")

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique senders and receivers
    senders = interactions[sender_col].unique()
    receivers = interactions[receiver_col].unique()

    # Create simplified sankey (matplotlib's Sankey is limited)
    # This is a simplified flow visualization

    n_senders = len(senders)
    n_receivers = len(receivers)

    # Sender positions (left side)
    sender_y = np.linspace(0.1, 0.9, n_senders)
    sender_pos = {s: (0.1, y) for s, y in zip(senders, sender_y)}

    # Receiver positions (right side)
    receiver_y = np.linspace(0.1, 0.9, n_receivers)
    receiver_pos = {r: (0.9, y) for r, y in zip(receivers, receiver_y)}

    # Normalize weights
    max_weight = interactions[weight_col].max()

    # Draw flows
    for _, row in interactions.iterrows():
        s = row[sender_col]
        r = row[receiver_col]
        w = row[weight_col]

        start = sender_pos[s]
        end = receiver_pos[r]

        # Draw curved line
        x_mid = 0.5
        lw = 1 + (w / max_weight) * 10

        # Bezier-like curve using plot
        t = np.linspace(0, 1, 50)
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * x_mid + t ** 2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * (start[1] + end[1]) / 2 + t ** 2 * end[1]

        ax.plot(x, y, lw=lw, alpha=0.5, color='steelblue')

    # Draw nodes
    for s, pos in sender_pos.items():
        ax.scatter(*pos, s=200, c='coral', zorder=10)
        ax.text(pos[0] - 0.05, pos[1], s, ha='right', va='center', fontsize=10)

    for r, pos in receiver_pos.items():
        ax.scatter(*pos, s=200, c='lightgreen', zorder=10)
        ax.text(pos[0] + 0.05, pos[1], r, ha='left', va='center', fontsize=10)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.scatter([], [], c='coral', s=100, label='Sender')
    ax.scatter([], [], c='lightgreen', s=100, label='Receiver')
    ax.legend(loc='lower center', ncol=2, frameon=False)

    plt.tight_layout()
    return fig


def plot_lr_pairs(
    lr_results: pd.DataFrame,
    n_top: int = 20,
    x: str = "score",
    y: str = "lr_pair",
    hue: str = "sender",
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Horizontal bar plot of top ligand-receptor pairs.

    Parameters
    ----------
    lr_results : pd.DataFrame
        DataFrame with L-R pair results.
    n_top : int, default 20
        Number of top pairs to show.
    x : str, default "score"
        Column for bar length.
    y : str, default "lr_pair"
        Column for bar labels.
    hue : str, default "sender"
        Column for bar colors.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.barplot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the bar plot.
    """
    _check_deps()
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")

    # Get top pairs
    df = lr_results.nlargest(n_top, x)

    if ax is None:
        figsize = (8, max(4, n_top * 0.3))
        fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x=x, y=y, hue=hue,
        ax=ax, **kwargs
    )

    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel("")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax
