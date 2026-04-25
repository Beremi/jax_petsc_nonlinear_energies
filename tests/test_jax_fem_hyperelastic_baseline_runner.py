from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.runners import run_jax_fem_hyperelastic_baseline as runner


def _write_state(path: Path, scale: float) -> None:
    coords_ref = np.asarray(
        [
            [0.0, -0.005, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, -0.005, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, -0.005, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    displacement = np.zeros_like(coords_ref)
    displacement[:, 0] = scale * coords_ref[:, 0]
    coords_final = coords_ref + displacement
    tetrahedra = np.asarray([[0, 1, 3, 2], [2, 3, 5, 4]], dtype=np.int32)
    np.savez(
        path,
        coords_ref=coords_ref,
        coords_final=coords_final,
        displacement=displacement,
        tetrahedra=tetrahedra,
    )


def test_build_comparison_summary_filters_nan_warmups_and_checks_fairness(tmp_path: Path) -> None:
    repo_state = tmp_path / "repo_state.npz"
    jax_state = tmp_path / "jax_state.npz"
    _write_state(repo_state, 0.02)
    _write_state(jax_state, 0.020002)

    maintained = {
        "implementation": "repo_serial_direct",
        "level": 1,
        "mesh_path": "/tmp/mesh.h5",
        "schedule": [0.0025, 0.005, 0.0075, 0.01],
        "state_npz": str(repo_state),
        "total_wall_time_s": 0.25,
        "case_contract": {"constitutive_law": "compressible_neo_hookean"},
        "rows": [
            {"step": 1, "displacement_x": 0.0025, "energy": 1.0, "u_max": 0.0025, "wall_time_s": 0.05},
            {"step": 2, "displacement_x": 0.0050, "energy": 1.5, "u_max": 0.0050, "wall_time_s": 0.05},
            {"step": 3, "displacement_x": 0.0075, "energy": 2.0, "u_max": 0.0075, "wall_time_s": 0.05},
            {"step": 4, "displacement_x": 0.0100, "energy": 2.5, "u_max": 0.0100, "wall_time_s": 0.05},
        ],
    }
    jax_fem = {
        "implementation": "jax_fem_umfpack_serial",
        "level": 1,
        "mesh_path": "/tmp/mesh.h5",
        "schedule": [0.0025, 0.005, 0.0075, 0.01],
        "state_npz": str(jax_state),
        "total_wall_time_s": 2.5,
        "case_contract": {"constitutive_law": "compressible_neo_hookean"},
        "rows": [
            {"step": 1, "displacement_x": 0.0025, "energy": 1.00001, "u_max": 0.0025, "wall_time_s": 0.5},
            {"step": 2, "displacement_x": 0.0050, "energy": 1.50001, "u_max": 0.0050, "wall_time_s": 0.5},
            {"step": 3, "displacement_x": 0.0075, "energy": 2.00001, "u_max": 0.0075, "wall_time_s": 0.5},
            {"step": 4, "displacement_x": 0.0100, "energy": 2.50001, "u_max": 0.0100, "wall_time_s": 0.5},
        ],
    }
    timing_rows = [
        {"implementation": "repo_serial_direct", "phase": "warmup", "repeat": 1, "wall_time_s": float("nan")},
        {"implementation": "jax_fem_umfpack_serial", "phase": "warmup", "repeat": 1, "wall_time_s": float("nan")},
        {"implementation": "repo_serial_direct", "phase": "measured", "repeat": 1, "wall_time_s": 0.20},
        {"implementation": "repo_serial_direct", "phase": "measured", "repeat": 2, "wall_time_s": 0.18},
        {"implementation": "repo_serial_direct", "phase": "measured", "repeat": 3, "wall_time_s": 0.22},
        {"implementation": "jax_fem_umfpack_serial", "phase": "measured", "repeat": 1, "wall_time_s": 2.50},
        {"implementation": "jax_fem_umfpack_serial", "phase": "measured", "repeat": 2, "wall_time_s": 2.40},
        {"implementation": "jax_fem_umfpack_serial", "phase": "measured", "repeat": 3, "wall_time_s": 2.60},
    ]

    summary = runner._build_comparison_summary(maintained, jax_fem, timing_rows)

    assert summary["timing_medians_s"] == {
        "repo_serial_direct": 0.2,
        "jax_fem_umfpack_serial": 2.5,
    }
    assert summary["fairness_gate"]["passed"] is True
    assert summary["final_metrics"]["energy_rel_diff"] < 5.0e-2
    assert summary["final_metrics"]["field_relative_l2"] < 5.0e-2
    assert len(summary["step_rows"]) == 4
