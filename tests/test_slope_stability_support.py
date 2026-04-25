from __future__ import annotations

import numpy as np

from src.problems.slope_stability.support import (
    DEFAULT_CASE,
    DEFAULT_LEVEL,
    build_case_data,
    build_refined_p1_case_data,
    build_same_mesh_lagrange_case_data,
    case_name_for_level,
    ensure_all_level_case_hdf5,
    h_for_level,
    supported_levels,
)


def test_default_case_matches_external_mesh_counts():
    case = build_case_data(DEFAULT_CASE)
    assert case.level == DEFAULT_LEVEL
    assert case.nodes.shape == (2721, 2)
    assert case.elems_scalar.shape == (1300, 6)
    assert case.elems.shape == (1300, 12)
    assert case.freedofs.shape == (5220,)
    assert int(case.q_mask[:, 0].sum()) == 2580
    assert int(case.q_mask[:, 1].sum()) == 2640
    assert case.adjacency.shape == (5220, 5220)


def test_default_case_quadrature_and_force_shapes():
    case = build_case_data(DEFAULT_CASE)
    assert case.elem_B.shape == (1300, 7, 3, 12)
    assert case.quad_weight.shape == (1300, 7)
    assert case.force.shape == (2 * case.nodes.shape[0],)
    assert case.eps_p_old.shape == (1300, 7, 3)
    assert np.all(case.quad_weight > 0.0)


def test_supported_level_counts_and_h_are_exact():
    expected = {
        1: {"h": 4.0, "nodes": 157, "elements": 64, "free_dofs": 260},
        2: {"h": 2.0, "nodes": 701, "elements": 320, "free_dofs": 1290},
        3: {"h": 1.0, "nodes": 2721, "elements": 1300, "free_dofs": 5220},
        4: {"h": 0.5, "nodes": 10641, "elements": 5200, "free_dofs": 20840},
        5: {"h": 0.25, "nodes": 42081, "elements": 20800, "free_dofs": 83280},
        6: {"h": 0.125, "nodes": 167361, "elements": 83200, "free_dofs": 332960},
        7: {"h": 0.0625, "nodes": 667521, "elements": 332800, "free_dofs": 1331520},
    }
    assert supported_levels() == (1, 2, 3, 4, 5, 6, 7)
    for level, counts in expected.items():
        case = build_case_data(case_name_for_level(level))
        assert h_for_level(level) == counts["h"]
        assert case.level == level
        assert case.h == counts["h"]
        assert case.nodes.shape == (counts["nodes"], 2)
        assert case.elems_scalar.shape == (counts["elements"], 6)
        assert case.elems.shape == (counts["elements"], 12)
        assert case.freedofs.shape == (counts["free_dofs"],)


def test_all_level_hdf5_snapshots_are_materialized():
    paths = ensure_all_level_case_hdf5()
    assert len(paths) == 8
    for path in paths:
        assert path.exists()


def test_refined_p1_case_reuses_nodes_and_refines_each_p2_triangle():
    p2_case = build_case_data(DEFAULT_CASE)
    p1_case = build_refined_p1_case_data(DEFAULT_CASE)
    assert p1_case.nodes.shape == p2_case.nodes.shape
    assert p1_case.freedofs.shape == p2_case.freedofs.shape
    assert p1_case.elems_scalar.shape == (4 * p2_case.elems_scalar.shape[0], 3)
    assert p1_case.elems.shape == (4 * p2_case.elems.shape[0], 6)
    assert p1_case.elem_B.shape == (4 * p2_case.elems_scalar.shape[0], 1, 3, 6)
    assert p1_case.quad_weight.shape == (4 * p2_case.elems_scalar.shape[0], 1)


def test_same_mesh_lagrange_cases_cover_p1_p2_p4_counts():
    expected = {
        1: {"nodes": 2721, "elements": 5200, "free_dofs": 5220},
        2: {"nodes": 10641, "elements": 5200, "free_dofs": 20840},
        4: {"nodes": 42081, "elements": 5200, "free_dofs": 83280},
    }
    for degree, counts in expected.items():
        case = build_same_mesh_lagrange_case_data(case_name_for_level(4), degree=degree)
        assert case.level == 4
        assert case.nodes.shape == (counts["nodes"], 2)
        assert case.elems_scalar.shape == (counts["elements"], {1: 3, 2: 6, 4: 15}[degree])
        assert case.freedofs.shape == (counts["free_dofs"],)
        assert case.elem_B.shape[0] == counts["elements"]
        assert case.elem_B.shape[-1] == 2 * case.elems_scalar.shape[1]
