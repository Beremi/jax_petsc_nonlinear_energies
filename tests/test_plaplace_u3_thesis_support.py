from __future__ import annotations

import math

import numpy as np

from src.problems.plaplace_u3.thesis.assignment import attach_assignment_metadata, classify_gap
from src.problems.plaplace_u3.thesis.directions import DirectionContext
from src.problems.plaplace_u3.thesis.functionals import (
    compute_state_stats_free,
    rescale_free_to_solution,
)
from src.problems.plaplace_u3.thesis.tables import TABLE_5_2_DIRECTION_D, TABLE_5_3_DIRECTION_VH
from src.problems.plaplace_u3.thesis.solver_common import build_objective_bundle, build_problem
from src.problems.plaplace_u3.thesis.solver_mpa import run_mpa
from src.problems.plaplace_u3.thesis.solver_oa import run_oa
from src.problems.plaplace_u3.thesis.transfer import (
    nested_w1p_error,
    prolong_free_to_problem,
    same_mesh_w1p_error,
)


def test_thesis_I_is_scale_invariant():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    stats = compute_state_stats_free(problem.params, problem.u_init)
    scaled = 3.7 * np.asarray(problem.u_init, dtype=np.float64)
    scaled_stats = compute_state_stats_free(problem.params, scaled)
    assert math.isclose(stats.I, scaled_stats.I, rel_tol=1e-10, abs_tol=1e-10)


def test_thesis_u_tilde_balances_a_and_b():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.5,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    scaled_free, _, stats = rescale_free_to_solution(problem.params, problem.u_init)
    assert np.linalg.norm(scaled_free) > 0.0
    assert math.isclose(stats.a, stats.b, rel_tol=1e-10, abs_tol=1e-10)


def test_nested_transfer_preserves_prolonged_target_state():
    coarse = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    fine = build_problem(
        dimension=2,
        level=3,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    coarse_scaled, _, _ = rescale_free_to_solution(coarse.params, coarse.u_init)
    coarse_on_fine = prolong_free_to_problem(coarse.params, coarse_scaled, fine.params)
    err = nested_w1p_error(coarse.params, coarse_scaled, fine.params, coarse_on_fine)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_same_mesh_error_is_zero_for_identical_state():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    scaled_free, _, _ = rescale_free_to_solution(problem.params, problem.u_init)
    err = same_mesh_w1p_error(problem.params, scaled_free, scaled_free)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_all_thesis_directions_are_descents():
    problem = build_problem(
        dimension=2,
        level=2,
        p=2.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    objective = build_objective_bundle(problem, "J")
    ctx = DirectionContext(problem, objective)

    for direction in ("d_vh", "d_rn", "d"):
        result = ctx.compute(np.asarray(problem.u_init, dtype=np.float64), direction)
        assert np.isfinite(result.stop_measure)
        assert result.descent_value <= 1.0e-10


def test_1d_tables_leave_p_9_over_6_unpublished_and_align_following_rows():
    assert (9.0 / 6.0) not in TABLE_5_2_DIRECTION_D
    assert (9.0 / 6.0) not in TABLE_5_3_DIRECTION_VH
    assert math.isclose(TABLE_5_2_DIRECTION_D[10.0 / 6.0]["J"], 0.76, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(TABLE_5_3_DIRECTION_VH[10.0 / 6.0]["J"], 0.76, rel_tol=0.0, abs_tol=1.0e-12)


def test_unpublished_rows_are_not_labeled_as_low_p_mismatches():
    row = {
        "table": "table_5_8",
        "method": "rmpa",
        "status": "maxit",
        "level": 7,
        "p": 1.5,
        "thesis_J": None,
        "thesis_I": None,
        "thesis_iterations": None,
        "thesis_direction_iterations": None,
        "assignment_acceptance_pass": None,
    }
    assert classify_gap(row) == "Secondary / unpublished thesis row"


def test_table_5_13_mismatch_is_labeled_as_low_impact_when_energy_matches():
    row = attach_assignment_metadata(
        {
        "table": "table_5_13",
        "method": "rmpa",
        "direction": "d",
        "status": "completed",
        "level": 6,
        "p": 3.0,
        "J": 4.194002180491161,
        "outer_iterations": 31,
        "thesis_direction_iterations": 19,
        "delta_direction_iterations": 12,
        }
    )
    assert row["assignment_acceptance_pass"] is True
    assert row["assignment_verdict"] == "low impact"
    assert (
        classify_gap(row)
        == "Low-impact direction-count discrepancy with matched principal-branch energy"
    )


def test_square_oa2_skew_stays_on_thesis_multibranch_solution():
    problem = build_problem(
        dimension=2,
        level=7,
        p=2.0,
        geometry="square_pi",
        init_mode="skew",
        seed=0,
    )
    result = run_oa(
        problem,
        variant="oa2",
        direction="d_vh",
        epsilon=1.0e-5,
        maxit=500,
    )

    assert result["status"] == "completed"
    assert math.isclose(result["I"], 2.98, rel_tol=0.0, abs_tol=0.03)
    assert math.isclose(result["J"], 19.80, rel_tol=0.0, abs_tol=0.25)


def test_square_mpa_low_p_row_returns_consistent_maxit_trace():
    problem = build_problem(
        dimension=2,
        level=6,
        p=11.0 / 6.0,
        geometry="square_pi",
        init_mode="sine",
        seed=0,
    )
    result = run_mpa(
        problem,
        direction="d_vh",
        epsilon=1.0e-3,
        maxit=20,
    )

    assert result["status"] == "maxit"
    assert len(result["history"]) == 20
    assert math.isfinite(result["J"])
    assert math.isfinite(result["I"])
    assert result["accepted_step_count"] == 20
    assert math.isclose(result["J"], result["history"][-1]["J"], rel_tol=0.0, abs_tol=1.0e-12)
    assert result["history"][-1]["stop_name"] == "(5.7)"
