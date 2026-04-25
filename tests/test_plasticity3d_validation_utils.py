from __future__ import annotations

import numpy as np

from experiments.analysis import plasticity3d_validation_utils as utils


def test_parse_markdown_pipe_table_extracts_rows() -> None:
    text = """
# Demo

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | 1.0 | 2.0 |
| Accepted steps | 3 | 4 |
"""
    rows = utils.parse_markdown_pipe_table(text, "## Summary")
    assert rows == [
        {"Metric": "Runtime [s]", "MATLAB": "1.0", "PETSc": "2.0"},
        {"Metric": "Accepted steps", "MATLAB": "3", "PETSc": "4"},
    ]


def test_curve_relative_l2_interpolates_candidate_grid() -> None:
    ref_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ref_y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cand_x = np.array([1.0, 3.0], dtype=np.float64)
    cand_y = np.array([1.0, 3.0], dtype=np.float64)
    assert utils.curve_relative_l2(ref_x, ref_y, cand_x, cand_y) == 0.0


def test_compute_boundary_profile_returns_sorted_trace() -> None:
    coords_ref = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    coords_final = coords_ref + np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    profile = utils.compute_boundary_profile(coords_ref, coords_final, y_band_fraction=0.2, z_quantile=0.5)
    assert np.all(np.diff(profile["x"]) >= 0.0)
    assert profile["u_mag"].shape[0] >= 3


def test_acceptance_flags_match_thresholds() -> None:
    flags = utils.acceptance_flags(
        critical_lambda_rel_diff=0.02,
        umax_curve_rel_l2=0.04,
        endpoint_disp_rel_l2=0.09,
    )
    assert flags == {
        "critical_lambda_pass": True,
        "umax_curve_pass": True,
        "endpoint_disp_pass": True,
    }
