#!/usr/bin/env python3
"""
SecActPy Command Line Interface

Secreted protein activity inference from gene expression data.

Usage:
    secactpy bulk -i <input> -o <output> [options]
    secactpy scrnaseq -i <input> -o <output> [options]
    secactpy visium -i <input> -o <output> [options]
    secactpy cosmx -i <input> -o <output> [options]

Examples:
    # Bulk RNA-seq (differential expression)
    secactpy bulk -i diff_expr.tsv -o results.h5ad --differential

    # scRNA-seq (h5ad or 10x)
    secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype

    # Visium spatial transcriptomics
    secactpy visium -i /path/to/visium/ -o results.h5ad

    # CosMx spatial transcriptomics
    secactpy cosmx -i data.h5ad -o results.h5ad --cell-type-col cell_type
"""

import argparse
import sys
import os
from pathlib import Path


# Detect CuPy/GPU availability at import time
def _detect_gpu():
    """Detect if CuPy and GPU are available."""
    try:
        import cupy as cp
        # Try to actually use the GPU
        cp.array([1, 2, 3])
        return True
    except Exception:
        return False

CUPY_AVAILABLE = _detect_gpu()
DEFAULT_BACKEND = "cupy" if CUPY_AVAILABLE else "numpy"


def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output H5AD file"
    )
    parser.add_argument(
        "-s", "--signature",
        default="secact",
        choices=["secact", "cytosig"],
        help="Signature matrix (default: secact)"
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=5e5,
        help="Ridge regularization parameter (default: 5e5)"
    )
    parser.add_argument(
        "-n", "--n-rand",
        type=int,
        default=1000,
        help="Number of permutations (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "cupy"],
        default=DEFAULT_BACKEND,
        help=f"Computation backend (default: {DEFAULT_BACKEND})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for large datasets (default: auto)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache permutation tables to disk"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )


def cmd_bulk(args: argparse.Namespace) -> int:
    """Run bulk RNA-seq inference."""
    from secactpy import secact_activity_inference
    from secactpy.io import save_results_to_h5ad
    import pandas as pd

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - Bulk RNA-seq Inference")
        print("=" * 60)
        print(f"Input:     {args.input}")
        print(f"Output:    {args.output}")
        print(f"Signature: {args.signature}")
        print(f"Lambda:    {args.lambda_}")
        print(f"N_rand:    {args.n_rand}")
        print(f"Backend:   {args.backend}")
        print("=" * 60)

    # Load input
    input_path = Path(args.input)
    if input_path.suffix in [".csv"]:
        expr = pd.read_csv(input_path, index_col=0)
    elif input_path.suffix in [".tsv", ".txt"]:
        expr = pd.read_csv(input_path, sep="\t", index_col=0)
    elif input_path.suffix in [".h5ad"]:
        import anndata
        adata = anndata.read_h5ad(input_path)
        expr = adata.to_df().T  # cells x genes -> genes x cells
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}", file=sys.stderr)
        return 1

    if verbose:
        print(f"Loaded expression: {expr.shape[0]} genes × {expr.shape[1]} samples")

    # Run inference
    result = secact_activity_inference(
        expr,
        is_differential=args.differential,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        verbose=verbose
    )

    # Save output
    save_results_to_h5ad(
        result,
        args.output,
        sample_names=list(expr.columns),
        verbose=verbose
    )

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_scrnaseq(args: argparse.Namespace) -> int:
    """Run scRNA-seq inference."""
    from secactpy import secact_activity_inference_scrnaseq

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - scRNA-seq Inference")
        print("=" * 60)
        print(f"Input:        {args.input}")
        print(f"Output:       {args.output}")
        print(f"Cell type:    {args.cell_type_col or 'None (all cells)'}")
        print(f"Single cell:  {args.single_cell}")
        print(f"Signature:    {args.signature}")
        print(f"Backend:      {args.backend}")
        print("=" * 60)

    # Run inference
    result = secact_activity_inference_scrnaseq(
        args.input,
        cell_type_col=args.cell_type_col,
        is_single_cell_level=args.single_cell,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_visium(args: argparse.Namespace) -> int:
    """Run Visium spatial transcriptomics inference."""
    from secactpy import secact_activity_inference_st

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - Visium Spatial Transcriptomics Inference")
        print("=" * 60)
        print(f"Input:       {args.input}")
        print(f"Output:      {args.output}")
        print(f"Cell type:   {args.cell_type_col or 'None (spot-level)'}")
        print(f"Spot level:  {args.spot_level}")
        print(f"Signature:   {args.signature}")
        print(f"Backend:     {args.backend}")
        print("=" * 60)

    # Run inference
    result = secact_activity_inference_st(
        args.input,
        cell_type_col=args.cell_type_col,
        is_spot_level=args.spot_level,
        min_genes=args.min_genes,
        scale_factor=args.scale_factor,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_cosmx(args: argparse.Namespace) -> int:
    """Run CosMx spatial transcriptomics inference."""
    from secactpy import secact_activity_inference_st

    verbose = not args.quiet and args.verbose

    if verbose:
        print("=" * 60)
        print("SecActPy - CosMx Spatial Transcriptomics Inference")
        print("=" * 60)
        print(f"Input:        {args.input}")
        print(f"Output:       {args.output}")
        print(f"Cell type:    {args.cell_type_col or 'None (cell-level)'}")
        print(f"Signature:    {args.signature}")
        print(f"Scale factor: {args.scale_factor}")
        print(f"Min genes:    {args.min_genes}")
        print(f"Sig filter:   {args.sig_filter}")
        print(f"Backend:      {args.backend}")
        print(f"Batch size:   {args.batch_size or 'auto'}")
        print("=" * 60)

    # Run inference (CosMx uses same ST function with different defaults)
    result = secact_activity_inference_st(
        args.input,
        cell_type_col=args.cell_type_col,
        is_spot_level=True,  # CosMx processes at cell level
        min_genes=args.min_genes,
        scale_factor=args.scale_factor,
        sig_matrix=args.signature,
        sig_filter=args.sig_filter,
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        verbose=verbose
    )

    # Save output
    from secactpy.io import save_results_to_h5ad
    save_results_to_h5ad(result, args.output, verbose=verbose)

    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two H5AD files for identity validation."""
    import numpy as np
    import pandas as pd
    
    print("=" * 60)
    print("SecActPy - H5AD Comparison")
    print("=" * 60)
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Sort indices: {args.sort}")
    print("=" * 60)
    
    # Try to load files
    try:
        import anndata as ad
    except ImportError:
        print("Error: anndata is required. Install with: pip install anndata")
        return 1
    
    try:
        adata1 = ad.read_h5ad(args.file1)
        print(f"\nFile 1 loaded: {adata1.shape}")
    except Exception as e:
        print(f"Error loading file 1: {e}")
        return 1
    
    try:
        adata2 = ad.read_h5ad(args.file2)
        print(f"File 2 loaded: {adata2.shape}")
    except Exception as e:
        print(f"Error loading file 2: {e}")
        return 1
    
    # Check shapes
    if adata1.shape != adata2.shape:
        print(f"\n❌ Shape mismatch: {adata1.shape} vs {adata2.shape}")
        return 1
    
    print(f"\n✓ Shapes match: {adata1.shape}")
    
    # If --sort, align data by indices
    if args.sort:
        print("\nSorting indices for alignment...")
        
        # Get common var_names and obs_names
        common_vars = sorted(set(adata1.var_names) & set(adata2.var_names))
        common_obs = sorted(set(adata1.obs_names) & set(adata2.obs_names))
        
        if len(common_vars) != len(adata1.var_names) or len(common_vars) != len(adata2.var_names):
            print(f"  Warning: var_names differ. Using {len(common_vars)} common features.")
        if len(common_obs) != len(adata1.obs_names) or len(common_obs) != len(adata2.obs_names):
            print(f"  Warning: obs_names differ. Using {len(common_obs)} common samples.")
        
        # Subset and reorder
        adata1 = adata1[common_obs, common_vars].copy()
        adata2 = adata2[common_obs, common_vars].copy()
        
        print(f"  Aligned shape: {adata1.shape}")
    
    # Helper function to extract matrix
    def get_matrix(adata, name):
        """Extract matrix from various locations in AnnData."""
        if name == 'beta':
            # Beta could be in X, layers['beta'], or obsm['beta']
            if 'beta' in adata.layers:
                return adata.layers['beta']
            elif 'beta' in adata.obsm:
                return adata.obsm['beta']
            else:
                return adata.X
        elif name in adata.layers:
            return adata.layers[name]
        elif name in adata.obsm:
            return adata.obsm[name]
        return None
    
    # Compare matrices
    matrices_to_compare = ['beta', 'se', 'zscore', 'pvalue']
    all_identical = True
    results = []
    
    for name in matrices_to_compare:
        mat1 = get_matrix(adata1, name)
        mat2 = get_matrix(adata2, name)
        
        if mat1 is None and mat2 is None:
            print(f"\n{name}: Not found in either file (skipping)")
            continue
        elif mat1 is None:
            print(f"\n❌ {name}: Only in file 2")
            all_identical = False
            continue
        elif mat2 is None:
            print(f"\n❌ {name}: Only in file 1")
            all_identical = False
            continue
        
        # Convert to dense if sparse
        if hasattr(mat1, 'toarray'):
            mat1 = mat1.toarray()
        if hasattr(mat2, 'toarray'):
            mat2 = mat2.toarray()
        
        mat1 = np.asarray(mat1, dtype=np.float64)
        mat2 = np.asarray(mat2, dtype=np.float64)
        
        if mat1.shape != mat2.shape:
            print(f"\n❌ {name}: Shape mismatch {mat1.shape} vs {mat2.shape}")
            all_identical = False
            continue
        
        # Compute differences
        diff = mat1 - mat2
        abs_diff = np.abs(diff)
        
        max_diff = np.nanmax(abs_diff)
        mean_diff = np.nanmean(abs_diff)
        
        # Correlation (flatten)
        valid = ~(np.isnan(mat1.flatten()) | np.isnan(mat2.flatten()))
        if valid.sum() > 1:
            corr = np.corrcoef(mat1.flatten()[valid], mat2.flatten()[valid])[0, 1]
        else:
            corr = np.nan
        
        # RMSE
        rmse = np.sqrt(np.nanmean(diff ** 2))
        
        # Check if identical within tolerance
        is_identical = max_diff <= args.tolerance
        
        status = "✓" if is_identical else "❌"
        print(f"\n{status} {name}:")
        print(f"    Shape:       {mat1.shape}")
        print(f"    Max diff:    {max_diff:.2e}")
        print(f"    Mean diff:   {mean_diff:.2e}")
        print(f"    RMSE:        {rmse:.2e}")
        print(f"    Correlation: {corr:.10f}")
        
        if not is_identical:
            all_identical = False
            # Show where differences occur
            max_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
            print(f"    Max diff at: {max_idx}")
            print(f"    File 1 val:  {mat1[max_idx]:.10f}")
            print(f"    File 2 val:  {mat2[max_idx]:.10f}")
            
            # Show sample/feature names at max diff location
            if args.verbose:
                obs_name = adata1.obs_names[max_idx[0]]
                var_name = adata1.var_names[max_idx[1]] if len(max_idx) > 1 else "N/A"
                print(f"    Location:    obs='{obs_name}', var='{var_name}'")
        
        results.append({
            'name': name,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rmse': rmse,
            'corr': corr,
            'identical': is_identical
        })
    
    # Compare indices (var_names, obs_names)
    print("\n" + "-" * 40)
    print("Index comparison:")
    
    # var_names (features/proteins)
    var1 = list(adata1.var_names)
    var2 = list(adata2.var_names)
    if var1 == var2:
        print(f"  ✓ var_names match ({len(var1)} features)")
    else:
        print(f"  ❌ var_names differ")
        if set(var1) == set(var2):
            print(f"    Same set, different order")
        else:
            only1 = set(var1) - set(var2)
            only2 = set(var2) - set(var1)
            if only1:
                print(f"    Only in file 1: {list(only1)[:5]}...")
            if only2:
                print(f"    Only in file 2: {list(only2)[:5]}...")
        if not args.sort:
            all_identical = False
    
    # obs_names (samples/cells)
    obs1 = list(adata1.obs_names)
    obs2 = list(adata2.obs_names)
    if obs1 == obs2:
        print(f"  ✓ obs_names match ({len(obs1)} samples)")
    else:
        print(f"  ❌ obs_names differ")
        if set(obs1) == set(obs2):
            print(f"    Same set, different order")
        if not args.sort:
            all_identical = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_identical:
        print("✓ Files are IDENTICAL (within tolerance)")
        return 0
    else:
        print("❌ Files are DIFFERENT")
        return 1


def main(argv=None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="secactpy",
        description="SecActPy: Secreted protein activity inference from gene expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bulk RNA-seq (differential expression)
  secactpy bulk -i diff_expr.tsv -o results.h5ad --differential -v

  # Bulk RNA-seq (raw counts, compute differential vs mean)
  secactpy bulk -i counts.tsv -o results.h5ad -v

  # scRNA-seq with cell type aggregation
  secactpy scrnaseq -i data.h5ad -o results.h5ad --cell-type-col celltype -v

  # scRNA-seq at single cell level
  secactpy scrnaseq -i data.h5ad -o results.h5ad --single-cell -v

  # Visium spatial transcriptomics (10x format)
  secactpy visium -i /path/to/visium/ -o results.h5ad -v

  # Visium with cell type deconvolution
  secactpy visium -i data.h5ad -o results.h5ad --cell-type-col cell_type -v

  # CosMx (single-cell spatial)
  secactpy cosmx -i cosmx.h5ad -o results.h5ad --batch-size 50000 -v

  # Use GPU acceleration
  secactpy bulk -i data.tsv -o results.h5ad --backend cupy -v

  # Use CytoSig signature
  secactpy bulk -i data.tsv -o results.h5ad --signature cytosig -v

  # Compare R and Python outputs
  secactpy compare r_output.h5ad python_output.h5ad -t 1e-10
  
  # Compare with different index ordering
  secactpy compare file1.h5ad file2.h5ad --sort -v

For more information, visit: https://github.com/psychemistz/SecActPy
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.2"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>"
    )

    # Bulk RNA-seq
    bulk_parser = subparsers.add_parser(
        "bulk",
        help="Bulk RNA-seq inference",
        description="Infer secreted protein activity from bulk RNA-seq data"
    )
    setup_common_args(bulk_parser)
    bulk_parser.add_argument(
        "-d", "--differential",
        action="store_true",
        help="Input is already differential expression (log2FC)"
    )
    bulk_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    bulk_parser.set_defaults(func=cmd_bulk)

    # scRNA-seq
    scrna_parser = subparsers.add_parser(
        "scrnaseq",
        help="scRNA-seq inference",
        description="Infer secreted protein activity from scRNA-seq data"
    )
    setup_common_args(scrna_parser)
    scrna_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type annotations"
    )
    scrna_parser.add_argument(
        "--single-cell",
        action="store_true",
        help="Compute at single cell level (default: aggregate by cell type)"
    )
    scrna_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    scrna_parser.set_defaults(func=cmd_scrnaseq)

    # Visium
    visium_parser = subparsers.add_parser(
        "visium",
        help="Visium spatial transcriptomics inference",
        description="Infer secreted protein activity from Visium data"
    )
    setup_common_args(visium_parser)
    visium_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type deconvolution results"
    )
    visium_parser.add_argument(
        "--spot-level",
        action="store_true",
        default=True,
        help="Compute at spot level (default: True)"
    )
    visium_parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="Minimum genes per spot (default: 200)"
    )
    visium_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1e5,
        help="Normalization scale factor (default: 1e5)"
    )
    visium_parser.add_argument(
        "--sig-filter",
        action="store_true",
        help="Filter signatures by available genes"
    )
    visium_parser.set_defaults(func=cmd_visium)

    # CosMx
    cosmx_parser = subparsers.add_parser(
        "cosmx",
        help="CosMx spatial transcriptomics inference",
        description="Infer secreted protein activity from CosMx data"
    )
    setup_common_args(cosmx_parser)
    cosmx_parser.add_argument(
        "-c", "--cell-type-col",
        default=None,
        help="Column name for cell type annotations"
    )
    cosmx_parser.add_argument(
        "--min-genes",
        type=int,
        default=50,
        help="Minimum genes per cell (default: 50)"
    )
    cosmx_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000,
        help="Normalization scale factor (default: 1000 for CosMx)"
    )
    cosmx_parser.add_argument(
        "--sig-filter",
        action="store_true",
        default=True,
        help="Filter signatures by available genes (default: True for CosMx)"
    )
    cosmx_parser.add_argument(
        "--no-sig-filter",
        action="store_false",
        dest="sig_filter",
        help="Disable signature filtering"
    )
    cosmx_parser.set_defaults(func=cmd_cosmx)

    # Compare H5AD files
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two H5AD files for validation",
        description="Compare two H5AD files to validate R vs Python output identity"
    )
    compare_parser.add_argument(
        "file1",
        help="First H5AD file (e.g., R output)"
    )
    compare_parser.add_argument(
        "file2",
        help="Second H5AD file (e.g., Python output)"
    )
    compare_parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=1e-10,
        help="Tolerance for numerical comparison (default: 1e-10)"
    )
    compare_parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort rows/columns before comparison (for different ordering)"
    )
    compare_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
