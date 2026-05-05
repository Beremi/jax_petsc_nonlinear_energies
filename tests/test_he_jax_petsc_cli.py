from __future__ import annotations

from src.problems.hyperelasticity.jax_petsc import solve_HE_dof


def test_he_jax_petsc_direct_cli_accepts_trust_region_flags():
    parser = solve_HE_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(
        [
            "--use_trust_region",
            "--trust_radius_init",
            "0.5",
            "--trust_shrink",
            "0.5",
            "--trust_expand",
            "1.5",
            "--trust_eta_shrink",
            "0.05",
            "--trust_eta_expand",
            "0.75",
            "--trust_max_reject",
            "6",
            "--trust_subproblem_line_search",
        ]
    )

    assert args.use_trust_region is True
    assert args.trust_radius_init == 0.5
    assert args.trust_shrink == 0.5
    assert args.trust_expand == 1.5
    assert args.trust_eta_shrink == 0.05
    assert args.trust_eta_expand == 0.75
    assert args.trust_max_reject == 6
    assert args.trust_subproblem_line_search is True


def test_he_jax_petsc_direct_cli_accepts_state_out():
    parser = solve_HE_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(["--state-out", "sample_state.npz"])

    assert args.state_out == "sample_state.npz"


def test_he_jax_petsc_direct_cli_accepts_distributed_element_options():
    parser = solve_HE_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(
        [
            "--problem_build_mode",
            "rank_local",
            "--distribution_strategy",
            "overlap_p2p",
            "--assembly_backend",
            "coo_local",
        ]
    )

    assert args.problem_build_mode == "rank_local"
    assert args.distribution_strategy == "overlap_p2p"
    assert args.assembly_backend == "coo_local"


def test_he_jax_petsc_direct_cli_accepts_pmg_options():
    parser = solve_HE_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(
        [
            "--pc_type",
            "mg",
            "--he_pmg_coarsest_level",
            "auto",
            "--he_pmg_smoother_ksp_type",
            "chebyshev",
            "--he_pmg_smoother_pc_type",
            "jacobi",
            "--he_pmg_coarse_pc_type",
            "redundant",
            "--he_pmg_coarse_redundant_number",
            "16",
            "--he_pmg_coarse_telescope_reduction_factor",
            "16",
        ]
    )

    assert args.pc_type == "mg"
    assert args.he_pmg_coarsest_level == "auto"
    assert args.he_pmg_smoother_ksp_type == "chebyshev"
    assert args.he_pmg_coarse_pc_type == "redundant"
    assert args.he_pmg_coarse_redundant_number == 16
    assert args.he_pmg_coarse_telescope_reduction_factor == 16
