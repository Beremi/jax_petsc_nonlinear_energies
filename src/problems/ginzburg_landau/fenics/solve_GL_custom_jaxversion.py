#!/usr/bin/env python3
"""
Ginzburg-Landau 2D solver — custom Newton (JAX-version) CLI entry point.

Solver logic is in ``src/problems/ginzburg_landau/fenics/solver_custom_newton.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Ginzburg-Landau 2D custom Newton benchmark"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9],
        help="Mesh levels to run (default: 5 6 7 8 9)",
    )
    parser.add_argument("--json", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    parser.add_argument("--ksp-type", type=str, default="gmres", help="PETSc KSP type")
    parser.add_argument(
        "--pc-type",
        type=str,
        default="hypre",
        help="Preconditioner type (default: hypre)",
    )
    parser.add_argument(
        "--ksp-rtol",
        type=float,
        default=1e-3,
        help="KSP relative tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--ksp-max-it",
        type=int,
        default=200,
        help="KSP maximum iterations (default: 200)",
    )
    parser.add_argument(
        "--linesearch-tol",
        type=float,
        default=1e-3,
        help="Line-search tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--state-out",
        type=str,
        default="",
        help="Optional NPZ path for exporting the final scalar state (single level only)",
    )
    args = parser.parse_args()

    if args.state_out and len(args.levels) != 1:
        parser.error("--state-out requires exactly one mesh level")

    import dolfinx
    from mpi4py import MPI

    from src.problems.ginzburg_landau.fenics.solver_custom_newton import run_level

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.size

    if rank == 0:
        sys.stdout.write(
            f"Ginzburg-Landau 2D Custom Newton | {nprocs} MPI process(es)\n"
        )
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.flush()

    all_results = []
    for mesh_lvl in args.levels:
        if rank == 0:
            sys.stdout.write(f"  --- Mesh level {mesh_lvl} ---\n")
            sys.stdout.flush()

        result = run_level(
            mesh_lvl,
            verbose=(not args.quiet),
            ksp_type=args.ksp_type,
            pc_type=args.pc_type,
            ksp_rtol=args.ksp_rtol,
            ksp_max_it=args.ksp_max_it,
            linesearch_tol=args.linesearch_tol,
            state_out=args.state_out,
        )
        all_results.append(result)

        if rank == 0:
            sys.stdout.write(
                f"  RESULT mesh_level={result['mesh_level']} "
                f"dofs={result['total_dofs']} "
                f"setup={result['setup_time']:.3f}s "
                f"solve={result['solve_time']:.3f}s "
                f"iters={result['iters']} ksp={result['total_ksp_its']} "
                f"asm={result['asm_time_cumulative']:.3f}s "
                f"J(u)={result['energy']:.6f} [{result['message']}]\n"
            )
            sys.stdout.flush()
        comm.Barrier()

    if rank == 0:
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.write("Done.\n")
        sys.stdout.flush()

        if args.json:
            metadata = {
                "solver": "custom_jaxversion",
                "description": (
                    "Custom Newton (JAX-version algorithm): "
                    f"golden-section line search [-0.5, 2], {args.ksp_type} + {args.pc_type}"
                ),
                "dolfinx_version": dolfinx.__version__,
                "nprocs": nprocs,
                "linear_solver": {
                    "ksp_type": args.ksp_type,
                    "pc_type": args.pc_type,
                    "ksp_rtol": args.ksp_rtol,
                    "ksp_max_it": args.ksp_max_it,
                },
                "newton_params": {
                    "tolf": 1e-6,
                    "tolg": 1e-5,
                    "linesearch_interval": [-0.5, 2.0],
                    "linesearch_tol": args.linesearch_tol,
                    "maxit": 100,
                },
                "eps": 0.01,
            }
            output = {"metadata": metadata, "results": all_results}
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                json.dump(output, fp, indent=2)
            sys.stdout.write(f"Results saved to {args.json}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
