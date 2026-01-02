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
        lambda_=args.lambda_,
        n_rand=args.n_rand,
        seed=args.seed,
        backend=args.backend,
        use_cache=args.use_cache,
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
        print(f"Input:       {args.input}")
        print(f"Output:      {args.output}")
        print(f"Cell type:   {args.cell_type_col or 'None (cell-level)'}")
        print(f"Signature:   {args.signature}")
        print(f"Backend:     {args.backend}")
        print(f"Batch size:  {args.batch_size or 'auto'}")
        print("=" * 60)

    # Run inference (CosMx uses same ST function with different defaults)
    result = secact_activity_inference_st(
        args.input,
        cell_type_col=args.cell_type_col,
        is_spot_level=False,  # CosMx is single-cell resolution
        min_genes=args.min_genes,
        scale_factor=args.scale_factor,
        sig_matrix=args.signature,
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
        default=20,
        help="Minimum genes per cell (default: 20)"
    )
    cosmx_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1e5,
        help="Normalization scale factor (default: 1e5)"
    )
    cosmx_parser.set_defaults(func=cmd_cosmx)

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
