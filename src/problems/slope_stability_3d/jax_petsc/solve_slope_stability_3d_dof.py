#!/usr/bin/env python3
"""CLI for the 3D heterogeneous slope-stability JAX + PETSc solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from src.core.cli.threading import configure_jax_cpu_threading
from src.problems.slope_stability_3d.support.mesh import DEFAULT_MESH_NAME


def _build_parser(profile_defaults):
    parser = argparse.ArgumentParser(
        description="3D heterogeneous slope-stability DOF-partitioned JAX + PETSc solver"
    )
    parser.add_argument("--mesh_name", type=str, default=DEFAULT_MESH_NAME)
    parser.add_argument("--elem_degree", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--lambda-target", dest="lambda_target", type=float, default=None)
    parser.add_argument("--initial-state", type=str, default="")
    parser.add_argument(
        "--profile",
        choices=sorted(profile_defaults.keys()),
        default="performance",
    )

    parser.add_argument("--ksp_type", type=str, default=None)
    parser.add_argument("--pc_type", type=str, default=None)
    parser.add_argument("--ksp_rtol", type=float, default=None)
    parser.add_argument("--ksp_max_it", type=int, default=None)
    parser.add_argument("--ksp_accept_true_rel", type=float, default=None)
    parser.add_argument(
        "--pc_setup_on_ksp_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--gamg_threshold", type=float, default=None)
    parser.add_argument("--gamg_agg_nsmooths", type=int, default=None)
    parser.add_argument("--hypre_nodal_coarsen", type=int, default=None)
    parser.add_argument("--hypre_vec_interp_variant", type=int, default=None)
    parser.add_argument("--hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--hypre_coarsen_type", type=str, default=None)
    parser.add_argument("--hypre_max_iter", type=int, default=None)
    parser.add_argument("--hypre_tol", type=float, default=None)
    parser.add_argument("--hypre_relax_type_all", type=str, default=None)
    parser.add_argument(
        "--gamg_set_coordinates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use_near_nullspace",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--distribution_strategy",
        choices=("overlap_allgather", "overlap_p2p"),
        default="overlap_p2p",
    )
    parser.add_argument(
        "--problem_build_mode",
        choices=("replicated", "root_bcast", "rank_local"),
        default="root_bcast",
    )
    parser.add_argument(
        "--mg_level_build_mode",
        choices=("replicated", "root_bcast", "rank_local"),
        default="root_bcast",
    )
    parser.add_argument(
        "--mg_transfer_build_mode",
        choices=("replicated", "root_bcast", "owned_rows"),
        default="owned_rows",
    )
    parser.add_argument(
        "--element_reorder_mode",
        choices=("none", "block_rcm", "block_xyz"),
        default="block_xyz",
    )
    parser.add_argument(
        "--local_hessian_mode",
        choices=("element", "sfd_local", "sfd_local_vmap"),
        default="element",
    )
    parser.add_argument(
        "--autodiff_tangent_mode",
        choices=("element", "constitutive"),
        default="element",
    )
    parser.add_argument(
        "--reuse_hessian_value_buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--p4_hessian_chunk_size", type=str, default="32")
    parser.add_argument(
        "--p4_chunk_scatter_cache",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--p4_chunk_scatter_cache_max_gib", type=float, default=0.5)
    parser.add_argument("--p4_chunk_autotune_candidates", type=str, default="32,64,128,256")
    parser.add_argument("--p4_chunk_autotune_rss_budget_gib", type=float, default=64.0)
    parser.add_argument(
        "--assembly_backend",
        choices=("coo", "coo_local", "blocked_local"),
        default="coo",
    )
    parser.add_argument("--petsc_log_view_path", type=str, default="")
    parser.add_argument(
        "--enable_petsc_log_events",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--jax_trace_dir", type=str, default="")

    parser.add_argument(
        "--mg_strategy",
        choices=(
            "auto",
            "same_mesh_p2_p1",
            "uniform_refined_p2_p1_chain",
            "same_mesh_p4_p2_p1",
            "uniform_refined_p4_p2_p1_p1",
            "uniform_refined_p1_chain",
        ),
        default="auto",
    )
    parser.add_argument("--mg_p1_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p1_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p1_smoother_steps", type=int, default=None)
    parser.add_argument("--mg_p2_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p2_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p2_smoother_steps", type=int, default=None)
    parser.add_argument("--mg_p4_smoother_ksp_type", type=str, default=None)
    parser.add_argument("--mg_p4_smoother_pc_type", type=str, default=None)
    parser.add_argument("--mg_p4_smoother_steps", type=int, default=None)
    parser.add_argument(
        "--mg_coarse_backend",
        choices=("hypre", "lu", "jacobi"),
        default="hypre",
    )
    parser.add_argument("--mg_coarse_ksp_type", type=str, default=None)
    parser.add_argument("--mg_coarse_pc_type", type=str, default=None)
    parser.add_argument("--mg_coarse_hypre_nodal_coarsen", type=int, default=6)
    parser.add_argument("--mg_coarse_hypre_vec_interp_variant", type=int, default=3)
    parser.add_argument("--mg_coarse_hypre_strong_threshold", type=float, default=0.5)
    parser.add_argument("--mg_coarse_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--mg_coarse_hypre_max_iter", type=int, default=2)
    parser.add_argument("--mg_coarse_hypre_tol", type=float, default=0.0)
    parser.add_argument(
        "--mg_coarse_hypre_relax_type_all",
        type=str,
        default="symmetric-SOR/Jacobi",
    )

    parser.add_argument("--tolf", type=float, default=1.0e-4)
    parser.add_argument("--tolg", type=float, default=1.0e-3)
    parser.add_argument("--tolg_rel", type=float, default=1.0e-3)
    parser.add_argument("--tolx_rel", type=float, default=1.0e-3)
    parser.add_argument("--tolx_abs", type=float, default=1.0e-10)
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument(
        "--line_search",
        choices=("golden_fixed", "armijo", "residual_bisection", "residual_bisection_tol"),
        default="residual_bisection",
    )
    parser.add_argument("--armijo_alpha0", type=float, default=1.0)
    parser.add_argument("--armijo_c1", type=float, default=1.0e-4)
    parser.add_argument("--armijo_shrink", type=float, default=0.5)
    parser.add_argument("--armijo_max_ls", type=int, default=40)
    parser.add_argument("--linesearch_a", type=float, default=-0.5)
    parser.add_argument("--linesearch_b", type=float, default=2.0)
    parser.add_argument("--linesearch_tol", type=float, default=1.0e-1)
    parser.add_argument(
        "--trust_subproblem_line_search",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use_trust_region",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--elastic_initial_guess",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--regularized_newton_tangent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--newton_r_min", type=float, default=1.0e-4)
    parser.add_argument("--newton_r_initial", type=float, default=1.0)
    parser.add_argument("--newton_r_max", type=float, default=2.0)
    parser.add_argument("--newton_r_fail_growth", type=float, default=2.0)
    parser.add_argument("--newton_r_small_alpha_growth", type=float, default=2.0 ** 0.25)
    parser.add_argument("--newton_r_decay", type=float, default=2.0 ** 0.5)
    parser.add_argument("--newton_r_retry_max", type=int, default=16)
    parser.add_argument("--trust_radius_init", type=float, default=0.5)
    parser.add_argument("--trust_radius_min", type=float, default=1.0e-8)
    parser.add_argument("--trust_radius_max", type=float, default=1.0e6)
    parser.add_argument("--trust_shrink", type=float, default=0.5)
    parser.add_argument("--trust_expand", type=float, default=1.5)
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05)
    parser.add_argument("--trust_eta_expand", type=float, default=0.75)
    parser.add_argument("--trust_max_reject", type=int, default=6)
    parser.add_argument("--step_time_limit_s", type=float, default=None)
    parser.add_argument(
        "--accept_ksp_maxit_direction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guard_ksp_maxit_direction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ksp_maxit_direction_true_rel_cap",
        type=float,
        default=6.0e-2,
    )

    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--save_history", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug_setup", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    parser.add_argument("--progress-out", type=str, default="")
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nproc", type=int, default=1)
    pre_args, _ = pre_parser.parse_known_args()
    configure_jax_cpu_threading(pre_args.nproc)

    from src.problems.slope_stability_3d.jax_petsc.solver import PROFILE_DEFAULTS, run

    parser = _build_parser(PROFILE_DEFAULTS)
    args = parser.parse_args()
    result = run(args)

    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(result, indent=2))
        if str(args.out):
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
