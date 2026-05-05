#!/usr/bin/env python3
"""HyperElasticity 3D solver — DOF-partitioned JAX + PETSc (CLI entry point).

Solver logic is in ``src/problems/hyperelasticity/jax_petsc/solver.py``.
"""

import argparse
import json
from pathlib import Path

from mpi4py import MPI
from src.core.cli.threading import configure_jax_cpu_threading


def _build_parser(profile_defaults):
    parser = argparse.ArgumentParser(
        description="HyperElasticity3D DOF-partitioned JAX + PETSc solver"
    )

    parser.add_argument("--level", type=int, default=1, help="Mesh level (1-4)")
    parser.add_argument("--steps", type=int, default=1, help="Number of load steps")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total steps corresponding to full 4*2*pi rotation (default: --steps)",
    )
    parser.add_argument("--start_step", type=int, default=1, help="Starting load-step index")

    parser.add_argument(
        "--profile",
        choices=sorted(profile_defaults.keys()),
        default="reference",
        help="Linear solver profile",
    )

    parser.add_argument("--ksp_type", type=str, default=None, help="PETSc KSP type")
    parser.add_argument("--pc_type", type=str, default=None, help="PETSc PC type")
    parser.add_argument("--ksp_rtol", type=float, default=None, help="KSP relative tolerance")
    parser.add_argument("--ksp_max_it", type=int, default=None, help="KSP maximum iterations")
    parser.add_argument(
        "--pc_setup_on_ksp_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse PC setup and refresh only when previous KSP hit ksp_max_it",
    )
    parser.add_argument(
        "--gamg_threshold",
        type=float,
        default=None,
        help="GAMG threshold (critical for HE; performance profile uses 0.05)",
    )
    parser.add_argument(
        "--gamg_agg_nsmooths",
        type=int,
        default=None,
        help="GAMG agg_nsmooths option",
    )
    parser.add_argument(
        "--gamg_set_coordinates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set PC coordinates (after setOperators) when using GAMG",
    )
    parser.add_argument(
        "--use_near_nullspace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Attach 6 rigid-body near-nullspace vectors",
    )
    parser.add_argument(
        "--reorder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RCM DOF reordering in partition",
    )

    parser.add_argument(
        "--local_coloring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-rank local coloring assembler",
    )
    parser.add_argument(
        "--hvp_eval_mode",
        choices=("batched", "sequential"),
        default="sequential",
        help="Local-coloring HVP evaluation mode",
    )
    parser.add_argument(
        "--coloring_trials",
        type=int,
        default=10,
        help="Coloring trials per rank (global-coloring mode)",
    )
    parser.add_argument(
        "--assembly_mode",
        choices=("sfd", "element"),
        default="sfd",
        help="Hessian assembly mode: 'sfd' (graph coloring + HVP) or "
             "'element' (analytical element Hessians via jax.hessian)",
    )
    parser.add_argument(
        "--element_reorder_mode",
        choices=("none", "block_rcm", "block_xyz", "block_metis"),
        default=None,
        help="Element mode only: reorder free DOFs before PETSc ownership split "
             "(default: block_xyz)",
    )
    parser.add_argument(
        "--local_hessian_mode",
        choices=("element", "sfd_local", "sfd_local_vmap"),
        default=None,
        help="Element mode only: local Hessian assembly kernel "
             "('element', 'sfd_local', or 'sfd_local_vmap')",
    )
    parser.add_argument(
        "--problem_build_mode",
        choices=("rank_local", "replicated"),
        default=None,
        help="Element mode only: build rank-local overlap data or the legacy replicated mesh",
    )
    parser.add_argument(
        "--mesh_source",
        choices=("procedural", "hdf5"),
        default=None,
        help="Element rank-local mode only: build the structured HE mesh procedurally "
             "or read rank-local rows from HDF5",
    )
    parser.add_argument(
        "--distribution_strategy",
        choices=("overlap_p2p", "overlap_allgather"),
        default=None,
        help="Element mode only: overlap value exchange strategy",
    )
    parser.add_argument(
        "--assembly_backend",
        choices=("coo_local", "coo"),
        default=None,
        help="Element mode only: PETSc COO preallocation/value insertion backend",
    )
    parser.add_argument(
        "--he_pmg_coarsest_level",
        type=str,
        default="1",
        help="HE PCMG coarsest level, or 'auto' to choose from MPI rank count",
    )
    parser.add_argument(
        "--he_pmg_auto_min_dofs_per_rank",
        type=int,
        default=128,
        help="Minimum coarse-grid free DOFs per rank used by --he_pmg_coarsest_level auto",
    )
    parser.add_argument("--he_pmg_smoother_ksp_type", type=str, default="chebyshev")
    parser.add_argument("--he_pmg_smoother_pc_type", type=str, default="jacobi")
    parser.add_argument("--he_pmg_smoother_steps", type=int, default=2)
    parser.add_argument("--he_pmg_coarse_ksp_type", type=str, default="")
    parser.add_argument("--he_pmg_coarse_pc_type", type=str, default="hypre")
    parser.add_argument("--he_pmg_coarse_redundant_number", type=int, default=0)
    parser.add_argument("--he_pmg_coarse_telescope_reduction_factor", type=int, default=0)
    parser.add_argument("--he_pmg_coarse_factor_solver_type", type=str, default="")
    parser.add_argument("--he_pmg_coarse_hypre_nodal_coarsen", type=int, default=6)
    parser.add_argument("--he_pmg_coarse_hypre_vec_interp_variant", type=int, default=3)
    parser.add_argument("--he_pmg_coarse_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--he_pmg_coarse_hypre_coarsen_type", type=str, default="")
    parser.add_argument("--he_pmg_coarse_hypre_max_iter", type=int, default=2)
    parser.add_argument("--he_pmg_coarse_hypre_tol", type=float, default=0.0)
    parser.add_argument(
        "--he_pmg_coarse_hypre_relax_type_all",
        type=str,
        default="symmetric-SOR/Jacobi",
    )
    parser.add_argument(
        "--he_pmg_galerkin",
        choices=("both", "pmat", "mat"),
        default="both",
        help="PETSc PCMG Galerkin operator policy",
    )

    parser.add_argument("--tolf", type=float, default=1e-4, help="Energy-change tolerance")
    parser.add_argument("--tolg", type=float, default=1e-3, help="Gradient-norm tolerance")
    parser.add_argument(
        "--tolg_rel",
        type=float,
        default=1e-3,
        help="Relative gradient tolerance (scaled by initial norm)",
    )
    parser.add_argument("--tolx_rel", type=float, default=1e-3, help="Relative step tolerance")
    parser.add_argument("--tolx_abs", type=float, default=1e-10, help="Absolute step tolerance")
    parser.add_argument("--maxit", type=int, default=100, help="Maximum Newton iterations")
    parser.add_argument(
        "--step_time_limit_s",
        type=float,
        default=None,
        help="Optional per-step wall-time limit passed into the Newton loop",
    )

    parser.add_argument("--linesearch_a", type=float, default=-0.5, help="Line-search lower bound")
    parser.add_argument("--linesearch_b", type=float, default=2.0, help="Line-search upper bound")
    parser.add_argument(
        "--linesearch_tol",
        type=float,
        default=1e-3,
        help="Golden-section line-search tolerance",
    )
    parser.add_argument(
        "--trust_subproblem_line_search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using PETSc trust-region KSPs (stcg/nash/gltr), apply a post-KSP line search",
    )
    parser.add_argument(
        "--use_trust_region",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable rho-based trust-region globalization in the nonlinear solver",
    )
    parser.add_argument("--trust_radius_init", type=float, default=1.0)
    parser.add_argument("--trust_radius_min", type=float, default=1e-8)
    parser.add_argument("--trust_radius_max", type=float, default=1e6)
    parser.add_argument("--trust_shrink", type=float, default=0.5)
    parser.add_argument("--trust_expand", type=float, default=1.5)
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05)
    parser.add_argument("--trust_eta_expand", type=float, default=0.75)
    parser.add_argument("--trust_max_reject", type=int, default=6)

    parser.add_argument(
        "--retry_on_failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry once on non-finite/maxit with tighter linear settings",
    )
    parser.add_argument("--stop_on_fail", action="store_true", help="Stop load stepping on first failure")

    parser.add_argument(
        "--use_abs_det",
        action="store_true",
        help="Use abs(det(F)) in energy (debug compatibility option)",
    )
    parser.add_argument("--nproc", type=int, default=1, help="OMP thread count per rank")
    parser.add_argument("--save_history", action="store_true", help="Save per-iteration Newton history")
    parser.add_argument(
        "--save_linear_timing",
        action="store_true",
        help="Save per-Newton linear timing breakdown",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    parser.add_argument("--out", type=str, default="", help="Write JSON output to this file")
    parser.add_argument(
        "--state-out",
        type=str,
        default="",
        help="Optional NPZ path for exporting the final deformed state",
    )

    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nproc", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()
    configure_jax_cpu_threading(pre_args.nproc)

    from src.problems.hyperelasticity.jax_petsc.solver import PROFILE_DEFAULTS, run

    parser = _build_parser(PROFILE_DEFAULTS)
    args = parser.parse_args()

    if args.total_steps is None:
        args.total_steps = args.steps

    result = run(args)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
