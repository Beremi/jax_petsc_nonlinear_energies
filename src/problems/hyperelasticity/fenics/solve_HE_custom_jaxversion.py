#!/usr/bin/env python3
"""
Hyperelasticity custom-Newton CLI entry point.

This canonical CLI now defaults to the maintained trust-region campaign
configuration while preserving the legacy long-option spellings as aliases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _add_legacy_store_false(
    parser: argparse.ArgumentParser,
    option: str,
    dest: str,
    *,
    help_text: str,
) -> None:
    parser.add_argument(option, action="store_false", dest=dest, help=help_text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1, help="Mesh level (1-4)")
    parser.add_argument("--steps", type=int, default=1, help="Number of time steps")
    parser.add_argument(
        "--start-step",
        "--start_step",
        dest="start_step",
        type=int,
        default=1,
        help="Global step index to start from",
    )
    parser.add_argument("--maxit", type=int, default=100, help="Maximum Newton iterations")
    parser.add_argument(
        "--init-npz",
        "--init_npz",
        dest="init_npz",
        type=str,
        default="",
        help="NPZ from JAX test data with coords and u_full_steps",
    )
    parser.add_argument(
        "--init-step",
        "--init_step",
        dest="init_step",
        type=int,
        default=0,
        help="Step index in init_npz to use as initial guess",
    )
    parser.add_argument(
        "--linesearch-a",
        "--linesearch_a",
        dest="linesearch_a",
        type=float,
        default=-0.5,
        help="Line-search interval lower bound",
    )
    parser.add_argument(
        "--linesearch-b",
        "--linesearch_b",
        dest="linesearch_b",
        type=float,
        default=2.0,
        help="Line-search interval upper bound",
    )
    parser.add_argument(
        "--linesearch-tol",
        "--linesearch_tol",
        dest="linesearch_tol",
        type=float,
        default=1e-1,
        help="Line-search tolerance",
    )
    parser.add_argument("--use-abs-det", "--use_abs_det", dest="use_abs_det", action="store_true")
    parser.add_argument(
        "--ksp-type",
        "--ksp_type",
        dest="ksp_type",
        type=str,
        default="stcg",
        help="PETSc KSP type",
    )
    parser.add_argument(
        "--pc-type",
        "--pc_type",
        dest="pc_type",
        type=str,
        default="gamg",
        help="PETSc PC type",
    )
    parser.add_argument(
        "--ksp-rtol",
        "--ksp_rtol",
        dest="ksp_rtol",
        type=float,
        default=1e-1,
        help="KSP relative tolerance",
    )
    parser.add_argument(
        "--ksp-max-it",
        "--ksp_max_it",
        dest="ksp_max_it",
        type=int,
        default=30,
        help="KSP maximum iterations per Newton step",
    )
    parser.add_argument("--tolf", type=float, default=1e-4, help="Newton energy-change tolerance")
    parser.add_argument("--tolg", type=float, default=1e-3, help="Newton gradient-norm tolerance")
    parser.add_argument(
        "--tolg-rel",
        "--tolg_rel",
        dest="tolg_rel",
        type=float,
        default=1e-3,
        help="Newton relative gradient tolerance (scaled by initial gradient)",
    )
    parser.add_argument(
        "--tolx-rel",
        "--tolx_rel",
        dest="tolx_rel",
        type=float,
        default=1e-3,
        help="Newton relative step-size tolerance",
    )
    parser.add_argument(
        "--tolx-abs",
        "--tolx_abs",
        dest="tolx_abs",
        type=float,
        default=1e-10,
        help="Newton absolute step-size tolerance",
    )
    parser.add_argument(
        "--require-all-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require energy, step, and gradient convergence together",
    )
    parser.add_argument(
        "--use-near-nullspace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach the elasticity near-nullspace to the Hessian",
    )
    _add_legacy_store_false(
        parser,
        "--no_near_nullspace",
        "use_near_nullspace",
        help_text="Disable elasticity near-nullspace on Hessian",
    )
    parser.add_argument(
        "--hypre-nodal-coarsen",
        "--hypre_nodal_coarsen",
        dest="hypre_nodal_coarsen",
        type=int,
        default=6,
        help="BoomerAMG nodal coarsen (-1 to skip setting)",
    )
    parser.add_argument(
        "--hypre-vec-interp-variant",
        "--hypre_vec_interp_variant",
        dest="hypre_vec_interp_variant",
        type=int,
        default=3,
        help="BoomerAMG vector interpolation variant (-1 to skip setting)",
    )
    parser.add_argument(
        "--hypre-strong-threshold",
        "--hypre_strong_threshold",
        dest="hypre_strong_threshold",
        type=float,
        default=None,
        help="BoomerAMG strong threshold",
    )
    parser.add_argument(
        "--hypre-coarsen-type",
        "--hypre_coarsen_type",
        dest="hypre_coarsen_type",
        type=str,
        default="",
        help="BoomerAMG coarsen type (e.g. HMIS, PMIS)",
    )
    parser.add_argument(
        "--save-history",
        "--save_history",
        dest="save_history",
        action="store_true",
        help="Include per-iteration Newton profile in output JSON",
    )
    parser.add_argument(
        "--save-linear-timing",
        "--save_linear_timing",
        dest="save_linear_timing",
        action="store_true",
        help="Include per-Newton linear timing breakdown in output JSON",
    )
    parser.add_argument(
        "--pc-setup-on-ksp-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run KSP/PC setup when previous linear solve hit ksp_max_it",
    )
    parser.add_argument("--pc_setup_on_ksp_cap", action="store_true", dest="pc_setup_on_ksp_cap")
    parser.add_argument("--out", type=str, default="", help="Output JSON file")
    parser.add_argument(
        "--total-steps",
        "--total_steps",
        dest="total_steps",
        type=int,
        default=24,
        help="Total steps that span the full 4x2pi rotation (controls step size)",
    )
    parser.add_argument(
        "--gamg-threshold",
        "--gamg_threshold",
        dest="gamg_threshold",
        type=float,
        default=0.05,
        help="GAMG threshold for filtering graph",
    )
    parser.add_argument(
        "--gamg-agg-nsmooths",
        "--gamg_agg_nsmooths",
        dest="gamg_agg_nsmooths",
        type=int,
        default=1,
        help="GAMG number of smoothing steps for SA prolongation",
    )
    parser.add_argument(
        "--gamg-set-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PCSetCoordinates for GAMG",
    )
    _add_legacy_store_false(
        parser,
        "--no_gamg_coordinates",
        "gamg_set_coordinates",
        help_text="Disable PCSetCoordinates for GAMG",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the trajectory early on Newton non-convergence",
    )
    _add_legacy_store_false(
        parser,
        "--no_fail_fast",
        "fail_fast",
        help_text="Do not stop trajectory early on Newton non-convergence",
    )
    parser.add_argument(
        "--retry-on-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair non-finite states with tighter settings",
    )
    _add_legacy_store_false(
        parser,
        "--no_retry_on_nonfinite",
        "retry_on_nonfinite",
        help_text="Disable non-finite repair attempt with tighter settings",
    )
    parser.add_argument(
        "--nonfinite-retry-rtol-factor",
        "--nonfinite_retry_rtol_factor",
        dest="nonfinite_retry_rtol_factor",
        type=float,
        default=0.1,
        help="Multiplier for KSP rtol in non-finite repair attempt",
    )
    parser.add_argument(
        "--nonfinite-retry-linesearch-b",
        "--nonfinite_retry_linesearch_b",
        dest="nonfinite_retry_linesearch_b",
        type=float,
        default=1.0,
        help="Upper bound of line-search interval in non-finite repair attempt",
    )
    parser.add_argument(
        "--retry-on-maxit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair attempts when Newton reaches max iterations",
    )
    _add_legacy_store_false(
        parser,
        "--no_retry_on_maxit",
        "retry_on_maxit",
        help_text="Disable repair attempt when Newton reaches max iterations",
    )
    parser.add_argument(
        "--retry-ksp-max-it-factor",
        "--retry_ksp_max_it_factor",
        dest="retry_ksp_max_it_factor",
        type=float,
        default=2.0,
        help="Multiplier for KSP max_it in repair attempt",
    )
    parser.add_argument(
        "--use-trust-region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trust-region globalization",
    )
    parser.add_argument(
        "--trust-radius-init",
        "--trust_radius_init",
        dest="trust_radius_init",
        type=float,
        default=0.5,
        help="Initial trust radius",
    )
    parser.add_argument(
        "--trust-radius-min",
        "--trust_radius_min",
        dest="trust_radius_min",
        type=float,
        default=1e-8,
        help="Minimum trust radius",
    )
    parser.add_argument(
        "--trust-radius-max",
        "--trust_radius_max",
        dest="trust_radius_max",
        type=float,
        default=1e6,
        help="Maximum trust radius",
    )
    parser.add_argument(
        "--trust-shrink",
        "--trust_shrink",
        dest="trust_shrink",
        type=float,
        default=0.5,
        help="Trust radius shrink factor",
    )
    parser.add_argument(
        "--trust-expand",
        "--trust_expand",
        dest="trust_expand",
        type=float,
        default=1.5,
        help="Trust radius expand factor",
    )
    parser.add_argument(
        "--trust-eta-shrink",
        "--trust_eta_shrink",
        dest="trust_eta_shrink",
        type=float,
        default=0.05,
        help="Trust ratio threshold for shrinking",
    )
    parser.add_argument(
        "--trust-eta-expand",
        "--trust_eta_expand",
        dest="trust_eta_expand",
        type=float,
        default=0.75,
        help="Trust ratio threshold for expanding",
    )
    parser.add_argument(
        "--trust-max-reject",
        "--trust_max_reject",
        dest="trust_max_reject",
        type=int,
        default=6,
        help="Maximum number of trust-region step rejections",
    )
    parser.add_argument(
        "--trust-subproblem-line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a post-KSP line search when using PETSc trust-region KSPs",
    )
    parser.add_argument(
        "--trust_subproblem_line_search",
        action="store_true",
        dest="trust_subproblem_line_search",
    )
    parser.add_argument(
        "--step-time-limit-s",
        "--step_time_limit_s",
        dest="step_time_limit_s",
        type=float,
        default=None,
        help="Optional per-step wall-time limit",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from mpi4py import MPI

    from src.problems.hyperelasticity.fenics.solver_custom_newton import run_level

    result = run_level(
        args.level,
        num_steps=args.steps,
        verbose=not args.quiet,
        maxit=args.maxit,
        start_step=args.start_step,
        init_npz=args.init_npz,
        init_step=args.init_step,
        linesearch_interval=(args.linesearch_a, args.linesearch_b),
        linesearch_tol=args.linesearch_tol,
        use_abs_det=args.use_abs_det,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        require_all_convergence=args.require_all_convergence,
        use_near_nullspace=args.use_near_nullspace,
        total_steps=args.total_steps,
        hypre_nodal_coarsen=args.hypre_nodal_coarsen,
        hypre_vec_interp_variant=args.hypre_vec_interp_variant,
        hypre_strong_threshold=args.hypre_strong_threshold,
        hypre_coarsen_type=args.hypre_coarsen_type,
        save_history=args.save_history,
        save_linear_timing=args.save_linear_timing,
        pc_setup_on_ksp_cap=args.pc_setup_on_ksp_cap,
        gamg_threshold=args.gamg_threshold,
        gamg_agg_nsmooths=args.gamg_agg_nsmooths,
        gamg_set_coordinates=args.gamg_set_coordinates,
        fail_fast=args.fail_fast,
        retry_on_nonfinite=args.retry_on_nonfinite,
        retry_on_maxit=args.retry_on_maxit,
        nonfinite_retry_rtol_factor=args.nonfinite_retry_rtol_factor,
        nonfinite_retry_linesearch_b=args.nonfinite_retry_linesearch_b,
        retry_ksp_max_it_factor=args.retry_ksp_max_it_factor,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        step_time_limit_s=args.step_time_limit_s,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
