"""
Survival analysis visualization functions for SecActPy.

Provides functions for Kaplan-Meier curves, forest plots,
and survival-related visualizations.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
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


def _check_lifelines():
    """Check if lifelines is available."""
    try:
        import lifelines
        return True
    except ImportError:
        raise ImportError("lifelines required: pip install lifelines")


def plot_kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_labels: Optional[Dict] = None,
    ax: Optional["plt.Axes"] = None,
    colors: Optional[List[str]] = None,
    show_censors: bool = True,
    show_ci: bool = True,
    ci_alpha: float = 0.2,
    show_at_risk: bool = False,
    at_risk_positions: Optional[List[float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    **kwargs
) -> "plt.Axes":
    """
    Kaplan-Meier survival curves.

    Parameters
    ----------
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators (1 = event, 0 = censored).
    groups : np.ndarray, optional
        Group labels for stratification.
    group_labels : dict, optional
        Mapping from group values to display labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    colors : list, optional
        Colors for each group.
    show_censors : bool, default True
        Whether to show censor marks.
    show_ci : bool, default True
        Whether to show confidence intervals.
    ci_alpha : float, default 0.2
        Confidence interval transparency.
    show_at_risk : bool, default False
        Whether to show at-risk table.
    at_risk_positions : list, optional
        Time points for at-risk table.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    **kwargs
        Additional arguments passed to KaplanMeierFitter.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the KM curves.

    Examples
    --------
    >>> # Single group
    >>> plot_kaplan_meier(time, event, title="Overall Survival")
    >>>
    >>> # Multiple groups
    >>> plot_kaplan_meier(
    ...     time, event,
    ...     groups=treatment,
    ...     group_labels={'A': 'Treatment A', 'B': 'Treatment B'}
    ... )
    """
    _check_matplotlib()
    _check_lifelines()
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    if groups is None:
        # Single group
        kmf.fit(time, event, label="All")
        kmf.plot_survival_function(
            ax=ax, ci_show=show_ci, show_censors=show_censors,
            ci_alpha=ci_alpha, **kwargs
        )
    else:
        # Multiple groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if colors is None:
            if SEABORN_AVAILABLE:
                colors = sns.color_palette("Set1", n_groups)
            else:
                colors = plt.cm.Set1(np.linspace(0, 1, n_groups))

        for i, group in enumerate(unique_groups):
            mask = groups == group
            label = group_labels.get(group, str(group)) if group_labels else str(group)

            # Add sample size to label
            n = np.sum(mask)
            n_events = np.sum(event[mask])
            label = f"{label} (n={n}, events={n_events})"

            kmf.fit(time[mask], event[mask], label=label)
            kmf.plot_survival_function(
                ax=ax, ci_show=show_ci, show_censors=show_censors,
                ci_alpha=ci_alpha, color=colors[i], **kwargs
            )

        # Log-rank test for 2 groups
        if n_groups == 2:
            g1, g2 = unique_groups
            mask1, mask2 = groups == g1, groups == g2
            result = logrank_test(
                time[mask1], time[mask2],
                event[mask1], event[mask2]
            )

            p = result.p_value
            if p < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p:.3f}"

            ax.text(
                0.95, 0.95, f"Log-rank test: {p_text}",
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=True)

    if title:
        ax.set_title(title)

    return ax


def plot_survival_by_activity(
    activity: pd.Series,
    time: np.ndarray,
    event: np.ndarray,
    threshold: Optional[float] = None,
    method: Literal["median", "mean", "optimal"] = "median",
    labels: Tuple[str, str] = ("Low", "High"),
    colors: Tuple[str, str] = ("#1f77b4", "#d62728"),
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Kaplan-Meier curves stratified by activity level.

    Parameters
    ----------
    activity : pd.Series
        Activity scores for each sample.
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators.
    threshold : float, optional
        Cutoff for high/low. If None, determined by method.
    method : str, default "median"
        Method to determine threshold: "median", "mean", or "optimal".
    labels : tuple, default ("Low", "High")
        Labels for low and high groups.
    colors : tuple, default (blue, red)
        Colors for low and high groups.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to plot_kaplan_meier.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the survival plot.

    Examples
    --------
    >>> plot_survival_by_activity(
    ...     result['zscore'].loc['IL6', :],
    ...     survival_time,
    ...     survival_event,
    ...     title="Survival by IL6 Activity"
    ... )
    """
    _check_matplotlib()
    _check_lifelines()

    values = activity.values

    # Determine threshold
    if threshold is None:
        if method == "median":
            threshold = np.median(values)
        elif method == "mean":
            threshold = np.mean(values)
        elif method == "optimal":
            # Find threshold that maximizes log-rank statistic
            from lifelines.statistics import logrank_test

            sorted_vals = np.sort(values)
            margin = len(values) // 10  # At least 10% in each group

            best_stat = 0
            best_thresh = np.median(values)

            for thresh in sorted_vals[margin:-margin]:
                high_mask = values >= thresh
                if np.sum(high_mask) < margin or np.sum(~high_mask) < margin:
                    continue

                try:
                    result = logrank_test(
                        time[high_mask], time[~high_mask],
                        event[high_mask], event[~high_mask]
                    )
                    if result.test_statistic > best_stat:
                        best_stat = result.test_statistic
                        best_thresh = thresh
                except Exception:
                    continue

            threshold = best_thresh

    # Create groups
    groups = np.where(values >= threshold, labels[1], labels[0])
    group_labels = {labels[0]: labels[0], labels[1]: labels[1]}

    return plot_kaplan_meier(
        time, event,
        groups=groups,
        group_labels=group_labels,
        colors=colors,
        ax=ax,
        title=title,
        **kwargs
    )


def plot_forest(
    results: pd.DataFrame,
    coef_col: str = "coef",
    ci_low_col: Optional[str] = None,
    ci_high_col: Optional[str] = None,
    se_col: Optional[str] = "se",
    pvalue_col: str = "p",
    label_col: Optional[str] = None,
    alpha: float = 0.05,
    ax: Optional["plt.Axes"] = None,
    color_sig: str = "#d62728",
    color_nonsig: str = "#7f7f7f",
    title: Optional[str] = None,
    xlabel: str = "log(Hazard Ratio)",
    **kwargs
) -> "plt.Axes":
    """
    Forest plot of Cox regression coefficients.

    Parameters
    ----------
    results : pd.DataFrame
        Cox regression results with coefficients and confidence intervals.
    coef_col : str, default "coef"
        Column with coefficients (log hazard ratios).
    ci_low_col, ci_high_col : str, optional
        Columns with CI bounds. If None, computed from SE.
    se_col : str, default "se"
        Column with standard errors (used if CI columns not provided).
    pvalue_col : str, default "p"
        Column with p-values.
    label_col : str, optional
        Column for row labels. If None, uses index.
    alpha : float, default 0.05
        Significance threshold.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    color_sig : str, default "#d62728"
        Color for significant results.
    color_nonsig : str, default "#7f7f7f"
        Color for non-significant results.
    title : str, optional
        Plot title.
    xlabel : str, default "log(Hazard Ratio)"
        X-axis label.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the forest plot.

    Examples
    --------
    >>> # From survival_analysis results
    >>> plot_forest(
    ...     survival_results.nsmallest(20, 'p'),
    ...     title="Top 20 Significant Proteins"
    ... )
    """
    _check_matplotlib()

    df = results.copy()
    n_vars = len(df)

    # Get labels
    if label_col:
        labels = df[label_col].values
    else:
        labels = df.index.tolist()

    # Get coefficients
    coefs = df[coef_col].values

    # Get confidence intervals
    if ci_low_col and ci_high_col:
        ci_low = df[ci_low_col].values
        ci_high = df[ci_high_col].values
    elif se_col:
        se = df[se_col].values
        ci_low = coefs - 1.96 * se
        ci_high = coefs + 1.96 * se
    else:
        raise ValueError("Must provide CI columns or SE column")

    # Get p-values
    pvalues = df[pvalue_col].values

    if ax is None:
        figsize = (8, max(4, n_vars * 0.4))
        fig, ax = plt.subplots(figsize=figsize)

    # Plot each variable
    y_positions = range(n_vars)

    for i, (coef, ci_l, ci_h, p) in enumerate(zip(coefs, ci_low, ci_high, pvalues)):
        color = color_sig if p < alpha else color_nonsig

        # CI line
        ax.plot([ci_l, ci_h], [i, i], color=color, linewidth=2)

        # Point estimate
        ax.scatter([coef], [i], color=color, s=80, zorder=10)

    # Reference line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.set_xlabel(xlabel)

    # Significance markers on right side
    for i, p in enumerate(pvalues):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < alpha:
            marker = "*"
        else:
            marker = ""

        ax.text(
            ax.get_xlim()[1], i, f"  {marker}",
            va='center', ha='left', fontsize=10
        )

    if title:
        ax.set_title(title)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_sig, marker='o', linestyle='-',
               markersize=8, label=f'p < {alpha}'),
        Line2D([0], [0], color=color_nonsig, marker='o', linestyle='-',
               markersize=8, label=f'p ≥ {alpha}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return ax


def plot_survival_volcano(
    results: pd.DataFrame,
    coef_col: str = "coef",
    pvalue_col: str = "p",
    label_col: Optional[str] = None,
    alpha: float = 0.05,
    coef_threshold: float = 0,
    n_label: int = 10,
    ax: Optional["plt.Axes"] = None,
    colors: Tuple[str, str, str] = ("#1f77b4", "#7f7f7f", "#d62728"),
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Volcano plot of survival analysis results.

    Parameters
    ----------
    results : pd.DataFrame
        Survival analysis results.
    coef_col : str, default "coef"
        Column with coefficients.
    pvalue_col : str, default "p"
        Column with p-values.
    label_col : str, optional
        Column for labels. If None, uses index.
    alpha : float, default 0.05
        Significance threshold.
    coef_threshold : float, default 0
        Coefficient threshold for highlighting.
    n_label : int, default 10
        Number of top points to label.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    colors : tuple, default (blue, gray, red)
        Colors for (protective, neutral, risk).
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the volcano plot.
    """
    _check_matplotlib()

    df = results.copy()

    if label_col:
        labels = df[label_col].values
    else:
        labels = df.index.tolist()

    coefs = df[coef_col].values
    pvalues = df[pvalue_col].values
    neg_log_p = -np.log10(np.clip(pvalues, 1e-300, 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Categorize points
    significant = pvalues < alpha
    protective = significant & (coefs < -coef_threshold)
    risk = significant & (coefs > coef_threshold)
    neutral = ~(protective | risk)

    # Plot each category
    ax.scatter(coefs[neutral], neg_log_p[neutral], c=colors[1], alpha=0.5, s=20)
    ax.scatter(coefs[protective], neg_log_p[protective], c=colors[0], alpha=0.7, s=30)
    ax.scatter(coefs[risk], neg_log_p[risk], c=colors[2], alpha=0.7, s=30)

    # Add threshold lines
    ax.axhline(-np.log10(alpha), color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    if coef_threshold > 0:
        ax.axvline(-coef_threshold, color='gray', linestyle=':', linewidth=1)
        ax.axvline(coef_threshold, color='gray', linestyle=':', linewidth=1)

    # Label top points
    top_idx = np.argsort(pvalues)[:n_label]
    for i in top_idx:
        ax.annotate(
            labels[i],
            (coefs[i], neg_log_p[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    ax.set_xlabel("Coefficient (log HR)")
    ax.set_ylabel("-log10(p-value)")

    if title:
        ax.set_title(title)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors[0], label='Protective'),
        mpatches.Patch(color=colors[1], label='Not significant'),
        mpatches.Patch(color=colors[2], label='Risk'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return ax


def plot_survival_heatmap(
    survival_results: pd.DataFrame,
    activity: pd.DataFrame,
    n_top: int = 20,
    sort_by: str = "p",
    ax: Optional["plt.Axes"] = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    **kwargs
) -> "plt.Axes":
    """
    Heatmap of top survival-associated proteins.

    Parameters
    ----------
    survival_results : pd.DataFrame
        Survival analysis results.
    activity : pd.DataFrame
        Activity matrix (proteins × samples).
    n_top : int, default 20
        Number of top proteins to show.
    sort_by : str, default "p"
        Column to sort by.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default "RdBu_r"
        Colormap.
    title : str, optional
        Plot title.
    **kwargs
        Additional arguments passed to seaborn.heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the heatmap.
    """
    _check_matplotlib()
    if not SEABORN_AVAILABLE:
        raise ImportError("seaborn required: pip install seaborn")

    # Get top proteins
    top_proteins = survival_results.nsmallest(n_top, sort_by).index

    # Subset activity matrix
    subset = activity.loc[top_proteins]

    if ax is None:
        figsize = (max(8, subset.shape[1] * 0.15), max(5, n_top * 0.3))
        fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        subset,
        ax=ax,
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Activity (z-score)'},
        **kwargs
    )

    if title is None:
        title = f"Top {n_top} Survival-Associated Proteins"
    ax.set_title(title)

    plt.tight_layout()
    return ax
