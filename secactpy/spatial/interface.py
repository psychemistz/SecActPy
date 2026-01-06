"""
Tumor-stroma interface detection for SecActPy.

Provides functions for identifying and analyzing the boundary
between tumor and stromal regions in spatial transcriptomics data.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import ndimage


def detect_interface(
    cell_types: np.ndarray,
    coords: np.ndarray,
    tumor_types: Union[str, List[str]],
    stroma_types: Union[str, List[str]],
    radius: float,
    method: Literal["neighbor", "distance", "gradient"] = "neighbor"
) -> Dict[str, np.ndarray]:
    """
    Detect tumor-stroma interface regions.

    Identifies spots/cells at the boundary between tumor and stromal
    compartments based on neighborhood composition.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels for each spot/cell.
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    tumor_types : str or list
        Cell type(s) considered as tumor.
    stroma_types : str or list
        Cell type(s) considered as stroma.
    radius : float
        Neighborhood radius for interface detection.
    method : str, default "neighbor"
        Detection method:
        - "neighbor": Interface if both tumor and stroma neighbors
        - "distance": Interface based on distance to compartments
        - "gradient": Interface at steepest composition gradient

    Returns
    -------
    dict
        Dictionary containing:
        - 'is_interface': Boolean mask for interface spots
        - 'interface_score': Continuous interface score
        - 'compartment': Compartment assignment ('tumor', 'stroma', 'interface', 'other')
        - 'tumor_fraction': Fraction of tumor neighbors
        - 'stroma_fraction': Fraction of stroma neighbors

    Examples
    --------
    >>> result = detect_interface(
    ...     cell_types, coords,
    ...     tumor_types=["Tumor", "Malignant"],
    ...     stroma_types=["Fibroblast", "Endothelial"],
    ...     radius=100
    ... )
    >>> interface_spots = result['is_interface']

    Notes
    -----
    R equivalent (SpaCET):
        getInteractionInterface <- function(object, cellTypeA, cellTypeB, radius) {
          ...
        }
    """
    from scipy.spatial import KDTree

    # Normalize inputs
    if isinstance(tumor_types, str):
        tumor_types = [tumor_types]
    if isinstance(stroma_types, str):
        stroma_types = [stroma_types]

    n_spots = len(cell_types)

    # Create compartment masks
    is_tumor = np.isin(cell_types, tumor_types)
    is_stroma = np.isin(cell_types, stroma_types)

    # Build spatial index
    tree = KDTree(coords)
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    # Calculate neighborhood composition for each spot
    tumor_fraction = np.zeros(n_spots)
    stroma_fraction = np.zeros(n_spots)
    n_neighbors = np.zeros(n_spots)

    for i, neighbors in enumerate(neighbors_list):
        neighbors = [j for j in neighbors if j != i]  # Exclude self
        if len(neighbors) > 0:
            n_neighbors[i] = len(neighbors)
            tumor_fraction[i] = np.mean(is_tumor[neighbors])
            stroma_fraction[i] = np.mean(is_stroma[neighbors])

    # Detect interface based on method
    if method == "neighbor":
        # Interface: spots with both tumor and stroma neighbors
        min_fraction = 0.1  # At least 10% of each
        is_interface = (tumor_fraction >= min_fraction) & (stroma_fraction >= min_fraction)
        # Interface score: geometric mean of fractions
        interface_score = np.sqrt(tumor_fraction * stroma_fraction)

    elif method == "distance":
        # Interface: spots close to both tumor and stroma regions
        tumor_coords = coords[is_tumor]
        stroma_coords = coords[is_stroma]

        if len(tumor_coords) > 0 and len(stroma_coords) > 0:
            tumor_tree = KDTree(tumor_coords)
            stroma_tree = KDTree(stroma_coords)

            dist_to_tumor, _ = tumor_tree.query(coords)
            dist_to_stroma, _ = stroma_tree.query(coords)

            # Interface score: inverse of max distance to either compartment
            max_dist = np.maximum(dist_to_tumor, dist_to_stroma)
            interface_score = 1 / (1 + max_dist / radius)
            is_interface = (dist_to_tumor <= radius) & (dist_to_stroma <= radius)
        else:
            interface_score = np.zeros(n_spots)
            is_interface = np.zeros(n_spots, dtype=bool)

    elif method == "gradient":
        # Interface: spots with steep gradient in tumor/stroma composition
        # Compute local gradient using neighbors
        gradient = np.zeros(n_spots)

        for i, neighbors in enumerate(neighbors_list):
            neighbors = [j for j in neighbors if j != i]
            if len(neighbors) > 1:
                # Variance in tumor fraction among neighbors
                neighbor_tumor_frac = tumor_fraction[neighbors]
                gradient[i] = np.std(neighbor_tumor_frac)

        # High gradient = interface
        threshold = np.percentile(gradient[gradient > 0], 75)
        is_interface = gradient >= threshold
        interface_score = gradient / (gradient.max() + 1e-10)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Assign compartments
    compartment = np.full(n_spots, "other", dtype=object)
    compartment[is_tumor] = "tumor"
    compartment[is_stroma] = "stroma"
    compartment[is_interface] = "interface"

    return {
        'is_interface': is_interface,
        'interface_score': interface_score,
        'compartment': compartment,
        'tumor_fraction': tumor_fraction,
        'stroma_fraction': stroma_fraction,
        'n_neighbors': n_neighbors
    }


def analyze_interface_activity(
    activity: pd.DataFrame,
    interface_result: Dict[str, np.ndarray],
    test: Literal["ttest", "wilcoxon", "anova"] = "wilcoxon"
) -> pd.DataFrame:
    """
    Analyze activity differences at tumor-stroma interface.

    Compares secreted protein activity between interface and
    non-interface regions.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × spots).
    interface_result : dict
        Output from detect_interface().
    test : str, default "wilcoxon"
        Statistical test: "ttest", "wilcoxon", or "anova".

    Returns
    -------
    pd.DataFrame
        Results with columns: protein, interface_mean, non_interface_mean,
        log2fc, statistic, pvalue, compartment_specific_means.

    Examples
    --------
    >>> interface = detect_interface(cell_types, coords, ...)
    >>> results = analyze_interface_activity(activity, interface)
    >>> significant = results[results['pvalue'] < 0.05]
    """
    from scipy import stats

    is_interface = interface_result['is_interface']
    compartment = interface_result['compartment']

    # Ensure activity columns match
    if len(is_interface) != activity.shape[1]:
        raise ValueError("Activity matrix columns must match number of spots")

    results = []

    for protein in activity.index:
        values = activity.loc[protein].values

        interface_vals = values[is_interface]
        non_interface_vals = values[~is_interface]

        if len(interface_vals) < 3 or len(non_interface_vals) < 3:
            continue

        # Compute statistics
        interface_mean = np.mean(interface_vals)
        non_interface_mean = np.mean(non_interface_vals)
        
        # Safe log2 fold change calculation
        # Shift values to positive range if needed
        min_val = min(interface_mean, non_interface_mean)
        if min_val < 0:
            shift = abs(min_val) + 0.01
            log2fc = np.log2((interface_mean + shift) / (non_interface_mean + shift))
        else:
            log2fc = np.log2((interface_mean + 0.01) / (non_interface_mean + 0.01))

        # Statistical test
        if test == "ttest":
            stat, pval = stats.ttest_ind(interface_vals, non_interface_vals)
        elif test == "wilcoxon":
            stat, pval = stats.mannwhitneyu(
                interface_vals, non_interface_vals,
                alternative='two-sided'
            )
        elif test == "anova":
            # ANOVA across compartments
            groups = [values[compartment == c] for c in ['tumor', 'stroma', 'interface']
                      if np.sum(compartment == c) >= 3]
            if len(groups) >= 2:
                stat, pval = stats.f_oneway(*groups)
            else:
                stat, pval = np.nan, np.nan
        else:
            raise ValueError(f"Unknown test: {test}")

        # Compartment-specific means
        tumor_mean = np.mean(values[compartment == 'tumor']) if np.sum(compartment == 'tumor') > 0 else np.nan
        stroma_mean = np.mean(values[compartment == 'stroma']) if np.sum(compartment == 'stroma') > 0 else np.nan

        results.append({
            'protein': protein,
            'interface_mean': interface_mean,
            'non_interface_mean': non_interface_mean,
            'tumor_mean': tumor_mean,
            'stroma_mean': stroma_mean,
            'log2fc': log2fc,
            'statistic': stat,
            'pvalue': pval
        })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    if len(results_df) > 0:
        from scipy.stats import false_discovery_control
        try:
            results_df['padj'] = false_discovery_control(results_df['pvalue'].values)
        except Exception:
            # Fallback: Benjamini-Hochberg manually
            pvals = results_df['pvalue'].values
            n = len(pvals)
            ranked = np.argsort(pvals)
            padj = np.zeros(n)
            padj[ranked] = pvals[ranked] * n / (np.arange(n) + 1)
            padj = np.minimum.accumulate(padj[::-1])[::-1]
            padj = np.clip(padj, 0, 1)
            results_df['padj'] = padj

    return results_df


def extract_interface_profile(
    activity: pd.DataFrame,
    coords: np.ndarray,
    interface_result: Dict[str, np.ndarray],
    n_bins: int = 10,
    max_distance: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract activity profiles across the tumor-stroma interface.

    Creates binned profiles showing how activity changes from
    tumor core through interface to stroma.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × spots).
    coords : np.ndarray
        Spatial coordinates.
    interface_result : dict
        Output from detect_interface().
    n_bins : int, default 10
        Number of bins across the interface gradient.
    max_distance : float, optional
        Maximum distance from interface to include.

    Returns
    -------
    pd.DataFrame
        Profile data with columns: protein, bin, distance_to_interface,
        mean_activity, std_activity, n_spots.

    Examples
    --------
    >>> profile = extract_interface_profile(activity, coords, interface)
    >>> # Plot profile for specific protein
    >>> il6 = profile[profile['protein'] == 'IL6']
    >>> plt.plot(il6['distance_to_interface'], il6['mean_activity'])
    """
    from scipy.spatial import KDTree

    is_interface = interface_result['is_interface']
    tumor_fraction = interface_result['tumor_fraction']

    if not np.any(is_interface):
        raise ValueError("No interface spots detected")

    interface_coords = coords[is_interface]
    interface_tree = KDTree(interface_coords)

    # Distance to interface for all spots
    dist_to_interface, _ = interface_tree.query(coords)

    # Signed distance: negative in tumor, positive in stroma
    # Use tumor_fraction as proxy
    signed_dist = dist_to_interface * np.sign(0.5 - tumor_fraction)

    if max_distance is not None:
        valid = np.abs(signed_dist) <= max_distance
    else:
        max_distance = np.percentile(np.abs(signed_dist), 95)
        valid = np.abs(signed_dist) <= max_distance

    # Bin spots by signed distance
    bins = np.linspace(-max_distance, max_distance, n_bins + 1)
    bin_labels = (bins[:-1] + bins[1:]) / 2
    spot_bins = np.digitize(signed_dist, bins) - 1
    spot_bins = np.clip(spot_bins, 0, n_bins - 1)

    results = []

    for protein in activity.index:
        values = activity.loc[protein].values

        for b in range(n_bins):
            mask = (spot_bins == b) & valid
            if np.sum(mask) >= 3:
                results.append({
                    'protein': protein,
                    'bin': b,
                    'distance_to_interface': bin_labels[b],
                    'mean_activity': np.mean(values[mask]),
                    'std_activity': np.std(values[mask]),
                    'n_spots': np.sum(mask)
                })

    return pd.DataFrame(results)


def find_interface_hotspots(
    activity: pd.DataFrame,
    coords: np.ndarray,
    interface_result: Dict[str, np.ndarray],
    protein: str,
    radius: float,
    min_spots: int = 5
) -> pd.DataFrame:
    """
    Find hotspots of activity along the interface.

    Identifies localized regions of high or low activity
    specifically at the tumor-stroma interface.

    Parameters
    ----------
    activity : pd.DataFrame
        Activity matrix (proteins × spots).
    coords : np.ndarray
        Spatial coordinates.
    interface_result : dict
        Output from detect_interface().
    protein : str
        Protein to analyze.
    radius : float
        Radius for local averaging.
    min_spots : int, default 5
        Minimum spots for a valid hotspot.

    Returns
    -------
    pd.DataFrame
        Hotspot information with coordinates, mean activity, z-score.
    """
    from scipy.spatial import KDTree

    is_interface = interface_result['is_interface']
    interface_idx = np.where(is_interface)[0]

    if len(interface_idx) < min_spots:
        return pd.DataFrame()

    interface_coords = coords[is_interface]
    values = activity.loc[protein].values[is_interface]

    # Local averaging
    tree = KDTree(interface_coords)
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    local_means = np.zeros(len(interface_idx))
    local_counts = np.zeros(len(interface_idx))

    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) >= min_spots:
            local_means[i] = np.mean(values[neighbors])
            local_counts[i] = len(neighbors)

    # Identify hotspots (local maxima/minima)
    global_mean = np.mean(values)
    global_std = np.std(values)

    if global_std > 0:
        zscores = (local_means - global_mean) / global_std
    else:
        zscores = np.zeros_like(local_means)

    # Filter significant hotspots
    significant = (np.abs(zscores) > 2) & (local_counts >= min_spots)

    return pd.DataFrame({
        'spot_idx': interface_idx[significant],
        'x': interface_coords[significant, 0],
        'y': interface_coords[significant, 1],
        'local_mean': local_means[significant],
        'zscore': zscores[significant],
        'n_neighbors': local_counts[significant],
        'is_hotspot': zscores[significant] > 2,
        'is_coldspot': zscores[significant] < -2
    })


def calc_interface_width(
    cell_types: np.ndarray,
    coords: np.ndarray,
    tumor_types: Union[str, List[str]],
    stroma_types: Union[str, List[str]],
    n_perpendiculars: int = 100,
    percentile: float = 90
) -> Dict[str, float]:
    """
    Estimate the width of the tumor-stroma interface.

    Parameters
    ----------
    cell_types : np.ndarray
        Cell type labels.
    coords : np.ndarray
        Spatial coordinates.
    tumor_types : str or list
        Tumor cell types.
    stroma_types : str or list
        Stroma cell types.
    n_perpendiculars : int, default 100
        Number of perpendicular profiles to sample.
    percentile : float, default 90
        Percentile for interface width estimation.

    Returns
    -------
    dict
        Dictionary with 'mean_width', 'median_width', 'std_width'.
    """
    if isinstance(tumor_types, str):
        tumor_types = [tumor_types]
    if isinstance(stroma_types, str):
        stroma_types = [stroma_types]

    is_tumor = np.isin(cell_types, tumor_types)
    is_stroma = np.isin(cell_types, stroma_types)

    tumor_coords = coords[is_tumor]
    stroma_coords = coords[is_stroma]

    if len(tumor_coords) < 10 or len(stroma_coords) < 10:
        return {'mean_width': np.nan, 'median_width': np.nan, 'std_width': np.nan}

    from scipy.spatial import KDTree

    # For each tumor cell, find distance to nearest stroma
    stroma_tree = KDTree(stroma_coords)
    dist_to_stroma, _ = stroma_tree.query(tumor_coords)

    # For border tumor cells (close to stroma), estimate interface width
    border_threshold = np.percentile(dist_to_stroma, 25)
    border_tumor = tumor_coords[dist_to_stroma <= border_threshold]

    # Sample perpendicular profiles
    widths = []
    tumor_tree = KDTree(tumor_coords)

    for _ in range(n_perpendiculars):
        # Pick random border tumor cell
        idx = np.random.randint(len(border_tumor))
        center = border_tumor[idx]

        # Find direction to nearest stroma
        _, stroma_idx = stroma_tree.query(center)
        direction = stroma_coords[stroma_idx] - center
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Walk along direction, measure transition zone
        # Transition zone = where mixed tumor/stroma cells exist
        max_dist = np.linalg.norm(stroma_coords[stroma_idx] - center) * 2
        n_steps = 50
        step_size = max_dist / n_steps

        tumor_frac = []
        for s in range(n_steps):
            point = center + direction * s * step_size
            # Count tumor vs stroma in small radius
            nearby_tumor = tumor_tree.query_ball_point(point, r=step_size * 2)
            nearby_stroma = stroma_tree.query_ball_point(point, r=step_size * 2)
            total = len(nearby_tumor) + len(nearby_stroma)
            if total > 0:
                tumor_frac.append(len(nearby_tumor) / total)
            else:
                tumor_frac.append(np.nan)

        tumor_frac = np.array(tumor_frac)

        # Interface width = distance where tumor_frac goes from >0.8 to <0.2
        if not np.all(np.isnan(tumor_frac)):
            high_tumor = np.where(tumor_frac > 0.8)[0]
            low_tumor = np.where(tumor_frac < 0.2)[0]

            if len(high_tumor) > 0 and len(low_tumor) > 0:
                width = (low_tumor[0] - high_tumor[-1]) * step_size
                if width > 0:
                    widths.append(width)

    if len(widths) == 0:
        return {'mean_width': np.nan, 'median_width': np.nan, 'std_width': np.nan}

    return {
        'mean_width': np.mean(widths),
        'median_width': np.median(widths),
        'std_width': np.std(widths),
        'n_samples': len(widths)
    }
