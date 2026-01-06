"""
Survival analysis utilities for SecActPy.

Provides functions for Cox proportional hazards regression
and optimal cutoff finding.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings


def coxph_best_separation(
    X: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    margin: int = 5,
    covariates: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, float]]:
    """
    Find optimal threshold for Cox PH regression with best separation.

    Performs Cox proportional hazards regression and finds the threshold
    that maximizes the absolute z-score for dichotomized analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features).
    time : np.ndarray
        Survival time for each sample.
    event : np.ndarray
        Event indicator (1 = event, 0 = censored).
    margin : int, default 5
        Minimum samples required in each group when searching thresholds.
    covariates : pd.DataFrame, optional
        Additional covariates to adjust for.

    Returns
    -------
    dict or None
        Dictionary with regression results and optimal threshold.
        Returns None if regression fails.
        Keys: 'coef', 'se', 'z', 'p', 'thres_opt', 'z_opt'

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = pd.DataFrame({'gene1': np.random.randn(100)})
    >>> time = np.random.exponential(10, 100)
    >>> event = np.random.binomial(1, 0.7, 100)
    >>> result = coxph_best_separation(X, time, event)

    Notes
    -----
    Requires lifelines package. Install with: pip install lifelines

    R equivalent:
        CoxPH_best_separation = function(X, Y, margin) {
          coxph.fit = coxph(Y ~ ., data=X)
          ...
          for(i in (margin+1):(n_r-margin)) {
            X[, n_c] = as.numeric(arr >= vthres[i])
            ...
          }
        }
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise ImportError(
            "lifelines package required for survival analysis. "
            "Install with: pip install lifelines"
        )

    n_samples = len(time)

    # Prepare data
    survival_df = pd.DataFrame({
        'time': time,
        'event': event
    })

    # Add features
    if isinstance(X, pd.Series):
        X = X.to_frame()

    feature_cols = list(X.columns)
    survival_df = pd.concat([survival_df, X.reset_index(drop=True)], axis=1)

    # Add covariates if provided
    if covariates is not None:
        survival_df = pd.concat([survival_df, covariates.reset_index(drop=True)], axis=1)

    # Part 1: Continuous regression
    try:
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(survival_df, duration_col='time', event_col='event')

        # Get results for last feature (the one we're testing)
        last_feature = feature_cols[-1]
        coef = cph.params_[last_feature]
        se = cph.standard_errors_[last_feature]
        z = coef / se
        p = cph.summary.loc[last_feature, 'p']

    except Exception:
        return None

    result = {
        'coef': coef,
        'se': se,
        'z': z,
        'p': p,
        'thres_opt': np.nan,
        'z_opt': np.nan
    }

    # Part 2: Find optimal threshold (if margin is set)
    if margin is None or margin <= 0:
        return result

    arr = X[last_feature].values
    vthres = np.sort(arr)

    z_opt = np.nan
    thres_opt = np.nan

    for i in range(margin, n_samples - margin):
        threshold = vthres[i]

        # Dichotomize
        survival_df_binary = survival_df.copy()
        survival_df_binary[last_feature] = (arr >= threshold).astype(float)

        try:
            cph_binary = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_binary.fit(
                    survival_df_binary,
                    duration_col='time',
                    event_col='event'
                )

            z_binary = (
                cph_binary.params_[last_feature] /
                cph_binary.standard_errors_[last_feature]
            )

            if np.isnan(z_binary):
                continue

            # Update optimal if better
            if np.isnan(z_opt):
                z_opt = z_binary
                thres_opt = threshold
            elif z > 0:  # Positive effect direction
                if z_binary > z_opt:
                    z_opt = z_binary
                    thres_opt = threshold
            else:  # Negative effect direction
                if z_binary < z_opt:
                    z_opt = z_binary
                    thres_opt = threshold

        except Exception:
            continue

    result['thres_opt'] = thres_opt
    result['z_opt'] = z_opt

    return result


def survival_analysis(
    activity_matrix: pd.DataFrame,
    survival_data: pd.DataFrame,
    time_col: str = 'time',
    event_col: str = 'event',
    margin: int = 5,
    covariates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform survival analysis for all secreted proteins.

    Parameters
    ----------
    activity_matrix : pd.DataFrame
        Activity scores (proteins × samples).
    survival_data : pd.DataFrame
        Survival information with time and event columns.
        Index should match activity_matrix columns.
    time_col : str, default 'time'
        Column name for survival time.
    event_col : str, default 'event'
        Column name for event indicator.
    margin : int, default 5
        Minimum samples per group for threshold search.
    covariates : list, optional
        List of covariate column names in survival_data.

    Returns
    -------
    pd.DataFrame
        Results for each protein with columns:
        coef, se, z, p, thres_opt, z_opt, mean_activity, n_samples

    Examples
    --------
    >>> # Analyze all proteins
    >>> results = survival_analysis(
    ...     activity['zscore'],
    ...     clinical_data,
    ...     time_col='OS_time',
    ...     event_col='OS_event'
    ... )
    >>> # Get significant proteins
    >>> significant = results[results['p'] < 0.05]
    """
    # Align samples
    common_samples = list(
        set(activity_matrix.columns) &
        set(survival_data.index)
    )

    if len(common_samples) == 0:
        raise ValueError("No common samples between activity and survival data")

    activity_aligned = activity_matrix[common_samples]
    survival_aligned = survival_data.loc[common_samples]

    time = survival_aligned[time_col].values
    event = survival_aligned[event_col].values

    # Prepare covariates
    if covariates:
        cov_df = survival_aligned[covariates]
    else:
        cov_df = None

    # Analyze each protein
    results = []
    proteins = activity_aligned.index

    for protein in proteins:
        values = activity_aligned.loc[protein].values
        X = pd.DataFrame({protein: values})

        result = coxph_best_separation(
            X, time, event,
            margin=margin,
            covariates=cov_df
        )

        if result is not None:
            result['protein'] = protein
            result['mean_activity'] = np.mean(values)
            result['n_samples'] = len(common_samples)
            results.append(result)

    # Create DataFrame
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.set_index('protein')
        # Reorder columns
        cols = ['coef', 'se', 'z', 'p', 'thres_opt', 'z_opt', 'mean_activity', 'n_samples']
        results_df = results_df[[c for c in cols if c in results_df.columns]]

    return results_df


def kaplan_meier_plot(
    values: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    threshold: Optional[float] = None,
    method: str = "median",
    ax=None,
    labels: Optional[Tuple[str, str]] = None,
    title: Optional[str] = None,
    colors: Tuple[str, str] = ("blue", "red")
):
    """
    Create Kaplan-Meier survival plot.

    Parameters
    ----------
    values : np.ndarray
        Values to stratify by (e.g., activity scores).
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators.
    threshold : float, optional
        Cutoff for high/low groups. If None, uses method.
    method : str, default "median"
        Method for automatic threshold: "median", "mean", "optimal".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    labels : tuple, optional
        Labels for (low, high) groups.
    title : str, optional
        Plot title.
    colors : tuple, default ("blue", "red")
        Colors for (low, high) groups.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "lifelines and matplotlib required. "
            "Install with: pip install lifelines matplotlib"
        )

    # Determine threshold
    if threshold is None:
        if method == "median":
            threshold = np.median(values)
        elif method == "mean":
            threshold = np.mean(values)
        elif method == "optimal":
            # Use coxph_best_separation to find optimal
            X = pd.DataFrame({'value': values})
            result = coxph_best_separation(X, time, event, margin=5)
            threshold = result['thres_opt'] if result else np.median(values)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Split into groups
    high_mask = values >= threshold
    low_mask = ~high_mask

    n_high = np.sum(high_mask)
    n_low = np.sum(low_mask)

    # Set up labels
    if labels is None:
        labels = (f"Low (n={n_low})", f"High (n={n_high})")

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    # Plot low group
    kmf.fit(time[low_mask], event[low_mask], label=labels[0])
    kmf.plot_survival_function(ax=ax, color=colors[0])

    # Plot high group
    kmf.fit(time[high_mask], event[high_mask], label=labels[1])
    kmf.plot_survival_function(ax=ax, color=colors[1])

    # Log-rank test
    lr_result = logrank_test(
        time[low_mask], time[high_mask],
        event[low_mask], event[high_mask]
    )

    # Add p-value to plot
    p_val = lr_result.p_value
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"

    ax.text(
        0.95, 0.95, p_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=12
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc='lower left')

    return ax


def log_rank_test(
    time1: np.ndarray,
    event1: np.ndarray,
    time2: np.ndarray,
    event2: np.ndarray
) -> Dict[str, float]:
    """
    Perform log-rank test between two groups.

    Parameters
    ----------
    time1, time2 : np.ndarray
        Survival times for each group.
    event1, event2 : np.ndarray
        Event indicators for each group.

    Returns
    -------
    dict
        Test results with 'statistic' and 'p_value'.
    """
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        raise ImportError("lifelines required: pip install lifelines")

    result = logrank_test(time1, time2, event1, event2)

    return {
        'statistic': result.test_statistic,
        'p_value': result.p_value
    }
