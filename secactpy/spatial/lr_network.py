"""
Ligand-receptor network scoring for SecActPy.

Provides functions for scoring cell-cell communication through
ligand-receptor interactions in spatial transcriptomics data.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse


def load_lr_database(
    database: Literal["cellchatdb", "cellphonedb", "custom"] = "cellchatdb",
    species: Literal["human", "mouse"] = "human",
    custom_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ligand-receptor interaction database.

    Parameters
    ----------
    database : str, default "cellchatdb"
        Database to use: "cellchatdb", "cellphonedb", or "custom".
    species : str, default "human"
        Species: "human" or "mouse".
    custom_path : str, optional
        Path to custom L-R database (CSV with 'ligand' and 'receptor' columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'ligand', 'receptor', 'pathway' (optional),
        'annotation' (optional).

    Notes
    -----
    For full databases, install:
    - CellChatDB: pip install cellchat (or download from GitHub)
    - CellPhoneDB: pip install cellphonedb

    This function provides a minimal built-in set for testing.
    """
    if database == "custom" and custom_path:
        return pd.read_csv(custom_path)

    # Built-in minimal L-R pairs for common cytokines
    # In production, these would be loaded from full databases
    minimal_lr = pd.DataFrame([
        # Cytokine signaling
        ('IL6', 'IL6R', 'IL6 signaling'),
        ('IL6', 'IL6ST', 'IL6 signaling'),
        ('TNF', 'TNFRSF1A', 'TNF signaling'),
        ('TNF', 'TNFRSF1B', 'TNF signaling'),
        ('IFNG', 'IFNGR1', 'IFN signaling'),
        ('IFNG', 'IFNGR2', 'IFN signaling'),
        ('IL1B', 'IL1R1', 'IL1 signaling'),
        ('IL1B', 'IL1R2', 'IL1 signaling'),
        ('TGFB1', 'TGFBR1', 'TGF-beta signaling'),
        ('TGFB1', 'TGFBR2', 'TGF-beta signaling'),
        ('VEGFA', 'FLT1', 'VEGF signaling'),
        ('VEGFA', 'KDR', 'VEGF signaling'),
        ('CXCL12', 'CXCR4', 'Chemokine signaling'),
        ('CCL2', 'CCR2', 'Chemokine signaling'),
        ('CCL5', 'CCR5', 'Chemokine signaling'),
        ('CXCL8', 'CXCR1', 'Chemokine signaling'),
        ('CXCL8', 'CXCR2', 'Chemokine signaling'),
        # Growth factors
        ('EGF', 'EGFR', 'EGF signaling'),
        ('HGF', 'MET', 'HGF signaling'),
        ('PDGFB', 'PDGFRB', 'PDGF signaling'),
        ('FGF2', 'FGFR1', 'FGF signaling'),
        # Immune checkpoints
        ('CD274', 'PDCD1', 'Immune checkpoint'),
        ('CD80', 'CD28', 'Immune checkpoint'),
        ('CD80', 'CTLA4', 'Immune checkpoint'),
        # WNT signaling
        ('WNT3A', 'FZD1', 'WNT signaling'),
        ('WNT5A', 'FZD5', 'WNT signaling'),
        # Notch signaling
        ('DLL1', 'NOTCH1', 'Notch signaling'),
        ('JAG1', 'NOTCH1', 'Notch signaling'),
    ], columns=['ligand', 'receptor', 'pathway'])

    return minimal_lr


def score_lr_interactions(
    expression: pd.DataFrame,
    coords: np.ndarray,
    cell_types: np.ndarray,
    lr_pairs: pd.DataFrame,
    radius: float,
    method: Literal["product", "geometric", "min"] = "product",
    normalize: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Score ligand-receptor interactions between cell types.

    Computes interaction scores based on spatial proximity and
    expression levels of ligands and receptors.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression matrix (genes × spots).
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    cell_types : np.ndarray
        Cell type labels for each spot.
    lr_pairs : pd.DataFrame
        L-R pairs with 'ligand' and 'receptor' columns.
    radius : float
        Maximum distance for interaction.
    method : str, default "product"
        Scoring method:
        - "product": ligand_expr * receptor_expr
        - "geometric": sqrt(ligand_expr * receptor_expr)
        - "min": min(ligand_expr, receptor_expr)
    normalize : bool, default True
        Whether to normalize scores by cell type frequency.

    Returns
    -------
    dict
        Dictionary containing:
        - 'scores': Interaction scores (sender_type × receiver_type × lr_pair)
        - 'pvalues': P-values from permutation test (if computed)
        - 'lr_summary': Summary per L-R pair

    Examples
    --------
    >>> lr_pairs = load_lr_database()
    >>> result = score_lr_interactions(
    ...     expression, coords, cell_types, lr_pairs,
    ...     radius=100
    ... )
    >>> scores = result['scores']
    """
    from scipy.spatial import KDTree

    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    n_spots = len(cell_types)
    n_pairs = len(lr_pairs)

    # Build spatial index
    tree = KDTree(coords)
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    # Initialize score tensor: sender × receiver × lr_pair
    scores = np.zeros((n_types, n_types, n_pairs))
    counts = np.zeros((n_types, n_types))  # Number of interacting pairs

    # Process each L-R pair
    for pair_idx, row in lr_pairs.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']

        # Check if genes exist in expression data
        if ligand not in expression.index or receptor not in expression.index:
            continue

        ligand_expr = expression.loc[ligand].values
        receptor_expr = expression.loc[receptor].values

        # Score each cell-cell interaction
        for i, neighbors in enumerate(neighbors_list):
            sender_type = type_to_idx[cell_types[i]]
            l_expr = ligand_expr[i]

            for j in neighbors:
                if i == j:
                    continue

                receiver_type = type_to_idx[cell_types[j]]
                r_expr = receptor_expr[j]

                # Compute interaction score
                if method == "product":
                    score = l_expr * r_expr
                elif method == "geometric":
                    score = np.sqrt(l_expr * r_expr)
                elif method == "min":
                    score = min(l_expr, r_expr)
                else:
                    raise ValueError(f"Unknown method: {method}")

                scores[sender_type, receiver_type, pair_idx] += score
                if pair_idx == 0:  # Count once
                    counts[sender_type, receiver_type] += 1

    # Normalize by number of interacting cell pairs
    if normalize:
        for ti in range(n_types):
            for tj in range(n_types):
                if counts[ti, tj] > 0:
                    scores[ti, tj, :] /= counts[ti, tj]

    # Create output DataFrames
    score_dfs = {}
    for pair_idx, row in lr_pairs.iterrows():
        pair_name = f"{row['ligand']}_{row['receptor']}"
        score_dfs[pair_name] = pd.DataFrame(
            scores[:, :, pair_idx],
            index=unique_types,
            columns=unique_types
        )

    # Summary per L-R pair
    lr_summary = []
    for pair_idx, row in lr_pairs.iterrows():
        pair_scores = scores[:, :, pair_idx]
        lr_summary.append({
            'ligand': row['ligand'],
            'receptor': row['receptor'],
            'pathway': row.get('pathway', ''),
            'mean_score': np.mean(pair_scores),
            'max_score': np.max(pair_scores),
            'top_sender': unique_types[np.unravel_index(pair_scores.argmax(), pair_scores.shape)[0]],
            'top_receiver': unique_types[np.unravel_index(pair_scores.argmax(), pair_scores.shape)[1]]
        })

    return {
        'scores': score_dfs,
        'counts': pd.DataFrame(counts, index=unique_types, columns=unique_types),
        'lr_summary': pd.DataFrame(lr_summary)
    }


def score_lr_spatial(
    expression: pd.DataFrame,
    coords: np.ndarray,
    lr_pairs: pd.DataFrame,
    radius: float,
    method: Literal["product", "geometric", "min"] = "product"
) -> pd.DataFrame:
    """
    Score L-R interactions for each spatial location.

    Returns spot-level interaction scores rather than cell-type aggregates.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression matrix (genes × spots).
    coords : np.ndarray
        Spatial coordinates.
    lr_pairs : pd.DataFrame
        L-R pairs DataFrame.
    radius : float
        Interaction radius.
    method : str, default "product"
        Scoring method.

    Returns
    -------
    pd.DataFrame
        Interaction scores (lr_pairs × spots).

    Examples
    --------
    >>> scores = score_lr_spatial(expression, coords, lr_pairs, radius=100)
    >>> # Plot spatial distribution of IL6-IL6R interaction
    >>> plot_spatial_feature(coords, scores.loc['IL6_IL6R'], cmap='Reds')
    """
    from scipy.spatial import KDTree

    n_spots = coords.shape[0]
    n_pairs = len(lr_pairs)

    tree = KDTree(coords)
    neighbors_list = tree.query_ball_tree(tree, r=radius)

    # Initialize scores: lr_pairs × spots
    pair_names = []
    scores = np.zeros((n_pairs, n_spots))

    for pair_idx, row in lr_pairs.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        pair_names.append(f"{ligand}_{receptor}")

        if ligand not in expression.index or receptor not in expression.index:
            continue

        ligand_expr = expression.loc[ligand].values
        receptor_expr = expression.loc[receptor].values

        # For each spot, sum interaction scores with neighbors
        for i, neighbors in enumerate(neighbors_list):
            l_expr = ligand_expr[i]

            neighbor_scores = []
            for j in neighbors:
                if i == j:
                    continue

                r_expr = receptor_expr[j]

                if method == "product":
                    score = l_expr * r_expr
                elif method == "geometric":
                    score = np.sqrt(l_expr * r_expr)
                elif method == "min":
                    score = min(l_expr, r_expr)

                neighbor_scores.append(score)

            if neighbor_scores:
                scores[pair_idx, i] = np.mean(neighbor_scores)

    return pd.DataFrame(scores, index=pair_names, columns=expression.columns)


def calc_communication_probability(
    expression: pd.DataFrame,
    coords: np.ndarray,
    cell_types: np.ndarray,
    lr_pairs: pd.DataFrame,
    radius: float,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculate communication probability with permutation testing.

    Similar to CellChat's communication probability calculation
    but with spatial constraints.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression matrix.
    coords : np.ndarray
        Spatial coordinates.
    cell_types : np.ndarray
        Cell type labels.
    lr_pairs : pd.DataFrame
        L-R pairs DataFrame.
    radius : float
        Interaction radius.
    n_permutations : int, default 1000
        Number of permutations for p-value calculation.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with 'probability', 'pvalue' matrices.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get observed scores
    observed = score_lr_interactions(
        expression, coords, cell_types, lr_pairs, radius
    )

    unique_types = np.unique(cell_types)
    n_types = len(unique_types)
    n_pairs = len(lr_pairs)

    # Stack observed scores
    obs_scores = np.zeros((n_types, n_types, n_pairs))
    for pair_idx, pair_name in enumerate(observed['scores'].keys()):
        obs_scores[:, :, pair_idx] = observed['scores'][pair_name].values

    # Permutation testing
    perm_scores = np.zeros((n_permutations, n_types, n_types, n_pairs))

    for p in range(n_permutations):
        # Permute cell type labels
        perm_types = np.random.permutation(cell_types)

        perm_result = score_lr_interactions(
            expression, coords, perm_types, lr_pairs, radius,
            normalize=True
        )

        for pair_idx, pair_name in enumerate(perm_result['scores'].keys()):
            perm_scores[p, :, :, pair_idx] = perm_result['scores'][pair_name].values

    # Calculate p-values
    pvalues = np.zeros((n_types, n_types, n_pairs))
    for ti in range(n_types):
        for tj in range(n_types):
            for pk in range(n_pairs):
                obs = obs_scores[ti, tj, pk]
                perm = perm_scores[:, ti, tj, pk]
                pvalues[ti, tj, pk] = np.mean(perm >= obs)

    # Format output
    prob_dfs = {}
    pval_dfs = {}

    pair_names = list(observed['scores'].keys())
    for pair_idx, pair_name in enumerate(pair_names):
        prob_dfs[pair_name] = pd.DataFrame(
            obs_scores[:, :, pair_idx],
            index=unique_types,
            columns=unique_types
        )
        pval_dfs[pair_name] = pd.DataFrame(
            pvalues[:, :, pair_idx],
            index=unique_types,
            columns=unique_types
        )

    return {
        'probability': prob_dfs,
        'pvalue': pval_dfs,
        'lr_summary': observed['lr_summary']
    }


def aggregate_pathway_scores(
    lr_scores: Dict[str, pd.DataFrame],
    lr_pairs: pd.DataFrame,
    method: Literal["mean", "sum", "max"] = "mean"
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate L-R scores by signaling pathway.

    Parameters
    ----------
    lr_scores : dict
        Dictionary of L-R score matrices from score_lr_interactions.
    lr_pairs : pd.DataFrame
        L-R pairs with 'pathway' column.
    method : str, default "mean"
        Aggregation method.

    Returns
    -------
    dict
        Dictionary of pathway-level score matrices.
    """
    if 'pathway' not in lr_pairs.columns:
        raise ValueError("lr_pairs must have 'pathway' column")

    pathways = lr_pairs['pathway'].unique()
    pathway_scores = {}

    for pathway in pathways:
        if pd.isna(pathway) or pathway == '':
            continue

        # Get L-R pairs in this pathway
        pathway_pairs = lr_pairs[lr_pairs['pathway'] == pathway]
        pair_matrices = []

        for _, row in pathway_pairs.iterrows():
            pair_name = f"{row['ligand']}_{row['receptor']}"
            if pair_name in lr_scores:
                pair_matrices.append(lr_scores[pair_name].values)

        if pair_matrices:
            stacked = np.stack(pair_matrices)

            if method == "mean":
                agg = np.mean(stacked, axis=0)
            elif method == "sum":
                agg = np.sum(stacked, axis=0)
            elif method == "max":
                agg = np.max(stacked, axis=0)

            pathway_scores[pathway] = pd.DataFrame(
                agg,
                index=lr_scores[pair_name].index,
                columns=lr_scores[pair_name].columns
            )

    return pathway_scores


def identify_significant_interactions(
    lr_result: Dict[str, pd.DataFrame],
    pvalue_threshold: float = 0.05,
    score_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract significant L-R interactions from results.

    Parameters
    ----------
    lr_result : dict
        Output from calc_communication_probability.
    pvalue_threshold : float, default 0.05
        P-value significance threshold.
    score_threshold : float, optional
        Minimum score threshold.

    Returns
    -------
    pd.DataFrame
        Significant interactions with sender, receiver, lr_pair,
        score, pvalue columns.
    """
    significant = []

    prob_dfs = lr_result.get('probability', lr_result.get('scores', {}))
    pval_dfs = lr_result.get('pvalue', {})

    for pair_name, prob_df in prob_dfs.items():
        pval_df = pval_dfs.get(pair_name)

        for sender in prob_df.index:
            for receiver in prob_df.columns:
                score = prob_df.loc[sender, receiver]
                pval = pval_df.loc[sender, receiver] if pval_df is not None else np.nan

                if pd.notna(pval) and pval < pvalue_threshold:
                    if score_threshold is None or score > score_threshold:
                        ligand, receptor = pair_name.split('_', 1)
                        significant.append({
                            'sender': sender,
                            'receiver': receiver,
                            'lr_pair': pair_name,
                            'ligand': ligand,
                            'receptor': receptor,
                            'score': score,
                            'pvalue': pval
                        })

    df = pd.DataFrame(significant)
    if len(df) > 0:
        df = df.sort_values('pvalue')

    return df


def compare_lr_conditions(
    expression1: pd.DataFrame,
    expression2: pd.DataFrame,
    coords: np.ndarray,
    cell_types: np.ndarray,
    lr_pairs: pd.DataFrame,
    radius: float,
    condition_names: Tuple[str, str] = ("Condition1", "Condition2")
) -> pd.DataFrame:
    """
    Compare L-R interactions between two conditions.

    Parameters
    ----------
    expression1, expression2 : pd.DataFrame
        Expression matrices for two conditions.
    coords : np.ndarray
        Spatial coordinates (same for both).
    cell_types : np.ndarray
        Cell type labels.
    lr_pairs : pd.DataFrame
        L-R pairs DataFrame.
    radius : float
        Interaction radius.
    condition_names : tuple
        Names for the two conditions.

    Returns
    -------
    pd.DataFrame
        Comparison results with log2 fold changes.
    """
    result1 = score_lr_interactions(expression1, coords, cell_types, lr_pairs, radius)
    result2 = score_lr_interactions(expression2, coords, cell_types, lr_pairs, radius)

    comparisons = []

    for pair_name in result1['scores'].keys():
        df1 = result1['scores'][pair_name]
        df2 = result2['scores'][pair_name]

        for sender in df1.index:
            for receiver in df1.columns:
                score1 = df1.loc[sender, receiver]
                score2 = df2.loc[sender, receiver]

                # Log2 fold change with pseudocount
                log2fc = np.log2((score2 + 0.01) / (score1 + 0.01))

                comparisons.append({
                    'sender': sender,
                    'receiver': receiver,
                    'lr_pair': pair_name,
                    f'score_{condition_names[0]}': score1,
                    f'score_{condition_names[1]}': score2,
                    'log2fc': log2fc
                })

    df = pd.DataFrame(comparisons)
    df = df.sort_values('log2fc', key=abs, ascending=False)

    return df
