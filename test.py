import pandas as pd
import numpy as np

# Load R results
r_zscore = pd.read_csv("dataset/output/signature/CosMx/zscore.txt", sep=r'\s+', index_col=0)
r_se = pd.read_csv("dataset/output/signature/CosMx/se.txt", sep=r'\s+', index_col=0)

# After running Python inference
py_zscore = py_result['zscore']
py_se = py_result['se']

# Align
py_zscore_aligned = py_zscore.loc[r_zscore.index, r_zscore.columns]
py_se_aligned = py_se.loc[r_se.index, r_se.columns]

# Find max diff location for SE
se_diff = np.abs(py_se_aligned.values - r_se.values)
max_se_idx = np.unravel_index(np.nanargmax(se_diff), se_diff.shape)
print(f"Max SE diff at: {r_se.index[max_se_idx[0]]}, {r_se.columns[max_se_idx[1]]}")
print(f"  R SE: {r_se.iloc[max_se_idx]}")
print(f"  Py SE: {py_se_aligned.iloc[max_se_idx]}")

# Find max diff location for zscore
zscore_diff = np.abs(py_zscore_aligned.values - r_zscore.values)
max_z_idx = np.unravel_index(np.nanargmax(zscore_diff), zscore_diff.shape)
print(f"\nMax zscore diff at: {r_zscore.index[max_z_idx[0]]}, {r_zscore.columns[max_z_idx[1]]}")
print(f"  R zscore: {r_zscore.iloc[max_z_idx]}")
print(f"  Py zscore: {py_zscore_aligned.iloc[max_z_idx]}")

# Check correlation
print(f"\nZscore correlation: {np.corrcoef(py_zscore_aligned.values.flatten(), r_zscore.values.flatten())[0,1]:.10f}")
