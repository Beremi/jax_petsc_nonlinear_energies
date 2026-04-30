from __future__ import annotations

from experiments.runners import (
    run_gl_final_suite,
    run_he_final_suite_best,
    run_he_pure_jax_suite_best,
    run_plaplace_final_suite,
)
from src.problems.hyperelasticity.jax.solve_HE_jax_newton import _result_from_step_message


def _single_step_payload(backend: str) -> dict:
    return {
        "case": {"backend": backend},
        "result": {
            "solve_time_total": 1.25,
            "steps": [
                {
                    "step": 1,
                    "nit": 3,
                    "time": 1.25,
                    "energy": -0.5,
                    "message": "Converged",
                    "history": [{"t_ls": 0.02}],
                    "linear_timing": [
                        {"ksp_its": 4, "assemble_total_time": 0.1, "pc_setup_time": 0.2, "solve_time": 0.3}
                    ],
                }
            ],
        },
    }


def _load_step_payload(backend: str) -> dict:
    return {
        "case": {"backend": backend},
        "result": {
            "solve_time_total": 2.5,
            "steps": [
                {
                    "step": 1,
                    "nit": 2,
                    "time": 1.0,
                    "energy": 5.0,
                    "message": "Converged",
                    "history": [{"t_ls": 0.1, "trust_rejects": 0}],
                    "linear_timing": [
                        {"ksp_its": 3, "assemble_time": 0.2, "pc_setup_time": 0.1, "solve_time": 0.2}
                    ],
                },
                {
                    "step": 2,
                    "nit": 3,
                    "time": 1.5,
                    "energy": 4.0,
                    "message": "Converged",
                    "history": [{"t_ls": 0.2, "trust_rejects": 1}],
                    "linear_timing": [
                        {"ksp_its": 4, "assemble_time": 0.3, "pc_setup_time": 0.2, "solve_time": 0.4}
                    ],
                },
            ],
        },
    }


def test_plaplace_summary_contract_keys_are_stable():
    row = run_plaplace_final_suite._summarize_case("fenics_custom", 5, 2, _single_step_payload("fenics"))
    assert set(row) == {
        "solver",
        "backend",
        "level",
        "nprocs",
        "completed_steps",
        "first_failed_step",
        "failure_mode",
        "failure_time_s",
        "total_newton_iters",
        "total_linear_iters",
        "total_time_s",
        "mean_step_time_s",
        "max_step_time_s",
        "assembly_time_s",
        "pc_init_time_s",
        "ksp_solve_time_s",
        "line_search_time_s",
        "final_energy",
        "result",
    }


def test_gl_summary_contract_keys_are_stable():
    row = run_gl_final_suite._summarize_case("fenics_custom", 5, 2, _single_step_payload("fenics"))
    assert set(row) == {
        "solver",
        "backend",
        "level",
        "nprocs",
        "completed_steps",
        "first_failed_step",
        "failure_mode",
        "failure_time_s",
        "total_newton_iters",
        "total_linear_iters",
        "total_time_s",
        "mean_step_time_s",
        "max_step_time_s",
        "assembly_time_s",
        "pc_init_time_s",
        "ksp_solve_time_s",
        "line_search_time_s",
        "final_energy",
        "result",
    }


def test_he_summary_contract_keys_are_stable():
    row = run_he_final_suite_best._summarize_case("fenics_custom", 24, 1, 2, _load_step_payload("fenics"))
    assert set(row) == {
        "solver",
        "backend",
        "total_steps",
        "level",
        "nprocs",
        "completed_steps",
        "first_failed_step",
        "failure_mode",
        "failure_time_s",
        "total_newton_iters",
        "total_linear_iters",
        "total_time_s",
        "mean_step_time_s",
        "max_step_time_s",
        "assembly_time_s",
        "pc_init_time_s",
        "ksp_solve_time_s",
        "line_search_time_s",
        "trust_rejects",
        "final_energy",
        "result",
    }


def test_pure_jax_he_summary_contract_keys_are_stable():
    row = run_he_pure_jax_suite_best._summarize_payload(
        {
            "solver": "pure_jax",
            "level": 1,
            "total_steps": 24,
            "total_dofs": 10,
            "free_dofs": 9,
            "time": 1.0,
            "total_newton_iters": 2,
            "total_linear_iters": 3,
            "result": "completed",
            "steps": [{"time": 1.0}],
        }
    )
    assert set(row) == {
        "solver",
        "level",
        "total_steps",
        "total_dofs",
        "free_dofs",
        "time",
        "total_newton_iters",
        "total_linear_iters",
        "max_step_time",
        "result",
    }


def test_pure_jax_he_status_accepts_lowercase_converged_message():
    assert _result_from_step_message("Trust-region step converged") == "completed"
    assert _result_from_step_message("Trust-region tolerances satisfied") == "completed"
    assert _result_from_step_message("Line search failed") == "failed"
