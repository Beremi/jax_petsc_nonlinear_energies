from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from src.problems.slope_stability_3d.jax.jax_energy_3d import (
    chunked_vmapped_element_constitutive_hessian_3d,
    chunked_vmapped_element_hessian_3d,
    element_constitutive_hessian_3d,
    element_energy_3d,
    element_hessian_3d,
    element_residual_3d,
    mc_potential_density_3d,
    principal_values_from_sym6,
)
from src.problems.slope_stability_3d.jax_petsc import multigrid
from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)
from src.problems.slope_stability_3d.support.mesh import (
    PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
    base_mesh_name_for_name,
    build_same_mesh_lagrange_case_data,
    ensure_same_mesh_case_hdf5,
    load_case_hdf5,
    load_same_mesh_case_hdf5_rank_local_light,
    ownership_block_size_3d,
    same_mesh_case_hdf5_path,
    select_reordered_perm_3d,
    uniform_refinement_steps_for_name,
)
from src.problems.slope_stability_3d.support.reduction import davis_b_reduction_qp
from src.problems.slope_stability_3d.support.simplex_lagrange import (
    evaluate_tetra_lagrange_basis,
    tetra_reference_nodes,
)
from experiments.runners import run_plasticity3d_backend_mix_case as plasticity3d_mix_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = "src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py"


def _numpy_davis_b(
    c0: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    c01 = np.asarray(c0, dtype=np.float64) / float(lam)
    phi1 = np.arctan(np.tan(np.asarray(phi, dtype=np.float64)) / float(lam))
    psi1 = np.arctan(np.tan(np.asarray(psi, dtype=np.float64)) / float(lam))
    beta = np.cos(phi1) * np.cos(psi1) / (1.0 - np.sin(phi1) * np.sin(psi1))
    c0_lambda = beta * c01
    phi_lambda = np.arctan(beta * np.tan(phi1))
    return 2.0 * c0_lambda * np.cos(phi_lambda), np.sin(phi_lambda)


def _numpy_potential_density_3d(
    eps6: np.ndarray,
    c_bar: float,
    sin_phi: float,
    shear: float,
    bulk: float,
    lame: float,
) -> float:
    e11, e22, e33, g12, g23, g13 = np.asarray(eps6, dtype=np.float64)
    # The source 3D benchmark computes invariants from IDENT * E_trial, so the
    # engineering shears must be halved before forming the symmetric tensor.
    e12 = 0.5 * g12
    e23 = 0.5 * g23
    e13 = 0.5 * g13
    I1 = e11 + e22 + e33
    I2 = e11 * e22 + e11 * e33 + e22 * e33 - e12 * e12 - e13 * e13 - e23 * e23
    I3 = (
        e11 * e22 * e33
        - e33 * e12 * e12
        - e22 * e13 * e13
        - e11 * e23 * e23
        + 2.0 * e12 * e13 * e23
    )
    Q = max(0.0, (I1 * I1 - 3.0 * I2) / 9.0)
    R = (-2.0 * I1**3 + 9.0 * I1 * I2 - 27.0 * I3) / 54.0
    theta = 0.0
    if Q > 0.0:
        theta = np.arccos(np.clip(R / max(1.0e-15, np.sqrt(Q**3)), -1.0, 1.0)) / 3.0
    sqrtQ = np.sqrt(Q)
    eig_1 = -2.0 * sqrtQ * np.cos(theta + 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_2 = -2.0 * sqrtQ * np.cos(theta - 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_3 = -2.0 * sqrtQ * np.cos(theta) + I1 / 3.0

    f_tr = (
        2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    )
    gamma_sl = (eig_1 - eig_2) / max(1.0e-15, 1.0 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / max(1.0e-15, 1.0 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / max(1.0e-15, 3.0 - sin_phi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / max(1.0e-15, 3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denom_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / np.sign(denom_s + 1.0e-15) / max(1.0e-15, abs(denom_s))
    lambda_l = (
        shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / np.sign(denom_l + 1.0e-15) / max(1.0e-15, abs(denom_l))
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / np.sign(denom_r + 1.0e-15) / max(1.0e-15, abs(denom_r))
    lambda_a = (
        2.0 * bulk * sin_phi * I1 - c_bar
    ) / np.sign(denom_a + 1.0e-15) / max(1.0e-15, abs(denom_a))

    elastic_quadratic = eig_1 * eig_1 + eig_2 * eig_2 + eig_3 * eig_3
    psi_el = 0.5 * lame * I1**2 + shear * elastic_quadratic
    psi_s = psi_el - 0.5 * denom_s * lambda_s**2
    psi_l = (
        0.5 * lame * I1**2
        + shear * (eig_3**2 + 0.5 * (eig_1 + eig_2) ** 2)
        - 0.5 * denom_l * lambda_l**2
    )
    psi_r = (
        0.5 * lame * I1**2
        + shear * (eig_1**2 + 0.5 * (eig_2 + eig_3) ** 2)
        - 0.5 * denom_r * lambda_r**2
    )
    psi_a = 0.5 * bulk * I1**2 - 0.5 * denom_a * lambda_a**2

    if f_tr <= 0.0:
        return float(psi_el)
    if lambda_s <= min(gamma_sl, gamma_sr):
        return float(psi_s)
    if gamma_sl < gamma_sr and gamma_sl <= lambda_l <= gamma_la:
        return float(psi_l)
    if gamma_sl > gamma_sr and gamma_sr <= lambda_r <= gamma_ra:
        return float(psi_r)
    return float(psi_a)


def _element_args_for_degree(degree: int):
    case = load_case_hdf5(ensure_same_mesh_case_hdf5("hetero_ssr_L1", int(degree)))
    c_bar_q, sin_phi_q = davis_b_reduction_qp(case.c0_q, case.phi_q, case.psi_q, 1.0)
    return case, c_bar_q, sin_phi_q


def _run_json(command: list[str], output_path: Path) -> dict[str, object]:
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _solver_command(
    output_path: Path,
    *,
    nprocs: int = 1,
    elem_degree: int = 2,
    pc_type: str = "gamg",
    ksp_type: str = "gmres",
    problem_build_mode: str = "replicated",
    mg_level_build_mode: str = "replicated",
    mg_transfer_build_mode: str = "owned_rows",
    mg_strategy: str = "same_mesh_p2_p1",
    distribution_strategy: str = "overlap_allgather",
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(nprocs),
        str(PYTHON),
        "-u",
        SOLVER,
        "--mesh_name",
        "hetero_ssr_L1",
        "--elem_degree",
        str(elem_degree),
        "--pc_type",
        str(pc_type),
        "--ksp_type",
        str(ksp_type),
        "--ksp_max_it",
        "100",
        "--maxit",
        "1",
        "--problem_build_mode",
        str(problem_build_mode),
        "--mg_level_build_mode",
        str(mg_level_build_mode),
        "--mg_transfer_build_mode",
        str(mg_transfer_build_mode),
        "--mg_strategy",
        str(mg_strategy),
        "--distribution_strategy",
        str(distribution_strategy),
        "--element_reorder_mode",
        "block_xyz",
        "--no-use_trust_region",
        "--quiet",
        "--out",
        str(output_path),
    ]


def _reference_same_mesh_transfer_entries_dict(
    coarse,
    fine,
    *,
    build_mode: str,
    tolerance: float = 1.0e-12,
):
    coarse_elem = np.asarray(coarse.params["elems_scalar"], dtype=np.int64)
    fine_elem = np.asarray(fine.params["elems_scalar"], dtype=np.int64)
    fine_ref = tetra_reference_nodes(int(fine.degree))
    coarse_hatp = np.asarray(
        evaluate_tetra_lagrange_basis(int(coarse.degree), fine_ref)[0],
        dtype=np.float64,
    )
    entries: dict[tuple[int, int], float] = {}
    for elem_id in range(int(fine_elem.shape[0])):
        coarse_nodes = np.asarray(coarse_elem[elem_id], dtype=np.int64)
        fine_nodes = np.asarray(fine_elem[elem_id], dtype=np.int64)
        for fine_local_idx, fine_node in enumerate(fine_nodes.tolist()):
            weights = np.asarray(coarse_hatp[:, fine_local_idx], dtype=np.float64)
            nonzero = np.flatnonzero(np.abs(weights) > float(tolerance))
            for comp in range(3):
                fine_total = 3 * int(fine_node) + comp
                fine_free_orig = int(fine.total_to_free_orig[fine_total])
                if fine_free_orig < 0:
                    continue
                fine_row = int(fine.iperm[fine_free_orig])
                if str(build_mode) == "owned_rows" and not (
                    int(fine.lo) <= fine_row < int(fine.hi)
                ):
                    continue
                for coarse_local_idx in nonzero.tolist():
                    coarse_total = 3 * int(coarse_nodes[coarse_local_idx]) + comp
                    coarse_free_orig = int(coarse.total_to_free_orig[coarse_total])
                    if coarse_free_orig < 0:
                        continue
                    coarse_col = int(coarse.iperm[coarse_free_orig])
                    value = float(weights[coarse_local_idx])
                    key = (fine_row, coarse_col)
                    previous = entries.get(key)
                    if previous is None:
                        entries[key] = value
                    else:
                        assert abs(previous - value) <= float(tolerance)
    if not entries:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    keys = np.asarray(list(entries.keys()), dtype=np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    rows = np.asarray(keys[order, 0], dtype=np.int64)
    cols = np.asarray(keys[order, 1], dtype=np.int64)
    data = np.asarray([entries[tuple(keys[idx])] for idx in order], dtype=np.float64)
    return rows, cols, data


def _reference_parent_transfer_entries_dict(
    coarse,
    fine,
    *,
    build_mode: str,
    tolerance: float = 1.0e-12,
):
    coarse_elem = np.asarray(coarse.params["elems_scalar"], dtype=np.int64)
    coarse_nodes = np.asarray(coarse.params["nodes"], dtype=np.float64)
    fine_elem = np.asarray(fine.params["elems_scalar"], dtype=np.int64)
    fine_nodes = np.asarray(fine.params["nodes"], dtype=np.float64)
    fine_parent = np.asarray(fine.params["macro_parent"], dtype=np.int64).ravel()
    entries: dict[tuple[int, int], float] = {}
    node_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for elem_id in range(int(fine_elem.shape[0])):
        parent_id = int(fine_parent[elem_id])
        coarse_elem_nodes = np.asarray(coarse_elem[parent_id], dtype=np.int64)
        coarse_vertex_nodes = coarse_elem_nodes[:4]
        parent_vertices = np.asarray(coarse_nodes[coarse_vertex_nodes], dtype=np.float64)
        base = np.asarray(parent_vertices[0], dtype=np.float64)
        jac = np.column_stack(
            (
                np.asarray(parent_vertices[1] - base, dtype=np.float64),
                np.asarray(parent_vertices[2] - base, dtype=np.float64),
                np.asarray(parent_vertices[3] - base, dtype=np.float64),
            )
        )
        for fine_node in np.asarray(fine_elem[elem_id], dtype=np.int64).tolist():
            cached = node_cache.get(int(fine_node))
            if cached is None:
                xi = np.linalg.solve(
                    jac,
                    np.asarray(fine_nodes[int(fine_node)] - base, dtype=np.float64),
                ).reshape(3, 1)
                weights = np.asarray(
                    evaluate_tetra_lagrange_basis(int(coarse.degree), xi)[0][:, 0],
                    dtype=np.float64,
                )
                nonzero = np.flatnonzero(np.abs(weights) > float(tolerance))
                cached = (
                    np.asarray(coarse_elem_nodes[nonzero], dtype=np.int64),
                    np.asarray(weights[nonzero], dtype=np.float64),
                )
                node_cache[int(fine_node)] = cached
            coarse_global_nodes, interp_weights = cached
            for comp in range(3):
                fine_total = 3 * int(fine_node) + comp
                fine_free_orig = int(fine.total_to_free_orig[fine_total])
                if fine_free_orig < 0:
                    continue
                fine_row = int(fine.iperm[fine_free_orig])
                if str(build_mode) == "owned_rows" and not (
                    int(fine.lo) <= fine_row < int(fine.hi)
                ):
                    continue
                for coarse_global_node, value in zip(
                    coarse_global_nodes, interp_weights, strict=False
                ):
                    coarse_total = 3 * int(coarse_global_node) + comp
                    coarse_free_orig = int(coarse.total_to_free_orig[coarse_total])
                    if coarse_free_orig < 0:
                        continue
                    coarse_col = int(coarse.iperm[coarse_free_orig])
                    key = (fine_row, coarse_col)
                    value_f = float(value)
                    previous = entries.get(key)
                    if previous is None:
                        entries[key] = value_f
                    else:
                        assert abs(previous - value_f) <= float(tolerance)
    if not entries:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    keys = np.asarray(list(entries.keys()), dtype=np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    rows = np.asarray(keys[order, 0], dtype=np.int64)
    cols = np.asarray(keys[order, 1], dtype=np.int64)
    data = np.asarray([entries[tuple(keys[idx])] for idx in order], dtype=np.float64)
    return rows, cols, data


def test_davis_b_reduction_qp_matches_source_array_formula():
    c0 = np.array([[15.0, 10.0], [18.0, 15.0]], dtype=np.float64)
    phi = np.deg2rad(np.array([[30.0, 35.0], [32.0, 38.0]], dtype=np.float64))
    psi = np.deg2rad(np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64))

    c_bar, sin_phi = davis_b_reduction_qp(c0, phi, psi, 1.2)
    exp_c_bar, exp_sin_phi = _numpy_davis_b(c0, phi, psi, 1.2)

    np.testing.assert_allclose(c_bar, exp_c_bar)
    np.testing.assert_allclose(sin_phi, exp_sin_phi)


def test_mc_potential_density_3d_matches_numpy_reference():
    rng = np.random.default_rng(0)
    eps6 = np.array([0.15, -0.03, 0.07, 0.04, -0.02, 0.01], dtype=np.float64)
    c_bar = 21.0
    sin_phi = np.sin(np.deg2rad(31.0))
    shear = 9200.0
    bulk = 15000.0
    lame = bulk - 2.0 * shear / 3.0

    value_jax = float(
        mc_potential_density_3d(
            jax.numpy.asarray(eps6),
            c_bar,
            sin_phi,
            shear,
            bulk,
            lame,
        )
    )
    value_np = _numpy_potential_density_3d(eps6, c_bar, sin_phi, shear, bulk, lame)
    assert np.isfinite(value_jax)
    np.testing.assert_allclose(value_jax, value_np, rtol=1.0e-10, atol=1.0e-10)


def test_principal_values_from_sym6_use_tensor_shear_conversion():
    eps6 = np.array([0.08, -0.04, 0.03, 0.12, -0.09, 0.05], dtype=np.float64)
    eig_1, eig_2, eig_3, I1 = principal_values_from_sym6(jax.numpy.asarray(eps6))

    e11, e22, e33, g12, g23, g13 = eps6
    mat = np.array(
        [
            [e11, 0.5 * g12, 0.5 * g13],
            [0.5 * g12, e22, 0.5 * g23],
            [0.5 * g13, 0.5 * g23, e33],
        ],
        dtype=np.float64,
    )
    expected = np.linalg.eigvalsh(mat)

    np.testing.assert_allclose(
        np.array([float(eig_3), float(eig_2), float(eig_1)]),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(float(I1), np.trace(mat), rtol=0.0, atol=1.0e-15)


def test_element_gradient_and_hessian_directional_sanity():
    case, c_bar_q, sin_phi_q = _element_args_for_degree(1)
    rng = np.random.default_rng(1)
    u = 1.0e-5 * rng.standard_normal(case.elems.shape[1])
    direction = rng.standard_normal(case.elems.shape[1])
    direction /= np.linalg.norm(direction)
    h = 1.0e-7

    def energy(u_vec: np.ndarray) -> float:
        return float(
            element_energy_3d(
                jax.numpy.asarray(u_vec),
                jax.numpy.asarray(case.dphix[0]),
                jax.numpy.asarray(case.dphiy[0]),
                jax.numpy.asarray(case.dphiz[0]),
                jax.numpy.asarray(case.quad_weight[0]),
                jax.numpy.asarray(c_bar_q[0]),
                jax.numpy.asarray(sin_phi_q[0]),
                jax.numpy.asarray(case.shear_q[0]),
                jax.numpy.asarray(case.bulk_q[0]),
                jax.numpy.asarray(case.lame_q[0]),
            )
        )

    grad = np.asarray(
        element_residual_3d(
            jax.numpy.asarray(u),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )
    hess = np.asarray(
        element_hessian_3d(
            jax.numpy.asarray(u),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )

    fd_grad_dir = (energy(u + h * direction) - energy(u - h * direction)) / (2.0 * h)
    np.testing.assert_allclose(fd_grad_dir, grad @ direction, rtol=2.0e-4, atol=5.0e-6)

    grad_plus = np.asarray(
        element_residual_3d(
            jax.numpy.asarray(u + h * direction),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )
    grad_minus = np.asarray(
        element_residual_3d(
            jax.numpy.asarray(u - h * direction),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )
    fd_hv = (grad_plus - grad_minus) / (2.0 * h)
    np.testing.assert_allclose(fd_hv, hess @ direction, rtol=5.0e-3, atol=5.0e-4)


def test_constitutive_autodiff_element_hessian_matches_element_hessian():
    case, c_bar_q, sin_phi_q = _element_args_for_degree(1)
    rng = np.random.default_rng(7)
    u = 1.0e-5 * rng.standard_normal(case.elems.shape[1])

    hess_element = np.asarray(
        element_hessian_3d(
            jax.numpy.asarray(u),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )
    hess_constitutive = np.asarray(
        element_constitutive_hessian_3d(
            jax.numpy.asarray(u),
            jax.numpy.asarray(case.dphix[0]),
            jax.numpy.asarray(case.dphiy[0]),
            jax.numpy.asarray(case.dphiz[0]),
            jax.numpy.asarray(case.quad_weight[0]),
            jax.numpy.asarray(c_bar_q[0]),
            jax.numpy.asarray(sin_phi_q[0]),
            jax.numpy.asarray(case.shear_q[0]),
            jax.numpy.asarray(case.bulk_q[0]),
            jax.numpy.asarray(case.lame_q[0]),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        hess_constitutive,
        hess_element,
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_large_cohesion_recovers_elastic_limit():
    rng = np.random.default_rng(2)
    eps6 = rng.standard_normal(6) * 1.0e-3
    sin_phi = np.sin(np.deg2rad(30.0))
    shear = 9100.0
    bulk = 15000.0
    lame = bulk - 2.0 * shear / 3.0
    elastic = 0.5 * lame * (eps6[0] + eps6[1] + eps6[2]) ** 2 + shear * (
        eps6[0] ** 2
        + eps6[1] ** 2
        + eps6[2] ** 2
        + 0.5 * (eps6[3] ** 2 + eps6[4] ** 2 + eps6[5] ** 2)
    )
    value = float(
        mc_potential_density_3d(
            jax.numpy.asarray(eps6),
            1.0e12,
            sin_phi,
            shear,
            bulk,
            lame,
        )
    )
    np.testing.assert_allclose(value, elastic, rtol=1.0e-10, atol=1.0e-10)


def test_importer_round_trip_and_gravity_sign():
    for degree, expected_nodes_per_elem, expected_nq in ((1, 4, 1), (2, 10, 11), (4, 35, 24)):
        path = ensure_same_mesh_case_hdf5("hetero_ssr_L1", degree)
        case = load_case_hdf5(path)
        assert case.elems_scalar.shape[1] == expected_nodes_per_elem
        assert case.dphix.shape[1] == expected_nq
        assert case.gravity_axis == 1
        assert np.array_equal(np.unique(case.material_id), np.array([0, 1, 2, 3], dtype=np.int64))
        force = np.asarray(case.force, dtype=np.float64).reshape((-1, 3))
        assert np.allclose(force[:, 0], 0.0)
        assert np.allclose(force[:, 2], 0.0)
        assert float(np.sum(force[:, 1])) < 0.0


def test_material_id_to_property_mapping_matches_source_benchmark_order():
    case = load_case_hdf5(ensure_same_mesh_case_hdf5("hetero_ssr_L1", 2))
    expected = {
        0: (15.0, 30.0, 19.0),
        1: (15.0, 38.0, 22.0),
        2: (10.0, 35.0, 21.0),
        3: (18.0, 32.0, 20.0),
    }
    for mid, (c0_exp, phi_deg_exp, gamma_exp) in expected.items():
        mask = np.asarray(case.material_id, dtype=np.int64) == int(mid)
        assert np.any(mask)
        np.testing.assert_allclose(case.c0_q[mask], c0_exp, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(case.phi_q[mask], np.deg2rad(phi_deg_exp), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(case.gamma_q[mask], gamma_exp, rtol=0.0, atol=1.0e-12)


def test_importer_uses_source_internal_xyz_axis_order():
    path = ensure_same_mesh_case_hdf5("hetero_ssr_L1", 2)
    case = load_case_hdf5(path)
    mins = np.min(np.asarray(case.nodes, dtype=np.float64), axis=0)
    maxs = np.max(np.asarray(case.nodes, dtype=np.float64), axis=0)
    np.testing.assert_allclose(mins, np.array([-175.0, 0.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(
        maxs,
        np.array([30.0, 60.0, 86.60254037844386]),
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("degree", [2, 4])
def test_boundary_lifting_marks_all_degree_aware_face_nodes_constrained(degree: int):
    case = load_case_hdf5(ensure_same_mesh_case_hdf5("hetero_ssr_L1", int(degree)))
    labels = {"x": (1, 2), "y": (5,), "z": (3, 4)}
    axis_idx = {"x": 0, "y": 1, "z": 2}
    for axis_name, constrained in labels.items():
        mask = np.isin(case.boundary_label, np.asarray(constrained, dtype=np.int64))
        nodes = np.unique(np.asarray(case.surf[mask], dtype=np.int64).ravel())
        assert nodes.size > 0
        assert not np.any(case.q_mask[nodes, axis_idx[axis_name]])


def test_glued_bottom_constrains_all_components_at_y_zero_nodes():
    case = load_case_hdf5(
        ensure_same_mesh_case_hdf5(
            "hetero_ssr_L1",
            1,
            constraint_variant=PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
        )
    )
    y0_nodes = np.where(np.abs(np.asarray(case.nodes, dtype=np.float64)[:, 1]) <= 1.0e-12)[0]
    assert y0_nodes.size > 0
    assert not np.any(np.asarray(case.q_mask, dtype=bool)[y0_nodes, :])


def test_componentwise_and_glued_bottom_cases_coexist_and_differ():
    componentwise_path = ensure_same_mesh_case_hdf5(
        "hetero_ssr_L1",
        1,
        constraint_variant=PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    )
    glued_path = ensure_same_mesh_case_hdf5(
        "hetero_ssr_L1",
        1,
        constraint_variant=PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
    )
    assert componentwise_path != glued_path
    assert componentwise_path == same_mesh_case_hdf5_path(
        "hetero_ssr_L1",
        1,
        PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    )
    assert glued_path == same_mesh_case_hdf5_path(
        "hetero_ssr_L1",
        1,
        PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
    )

    componentwise = load_case_hdf5(componentwise_path)
    glued = load_case_hdf5(glued_path)
    assert str(componentwise.constraint_variant) == PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM
    assert str(glued.constraint_variant) == PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM
    assert int(glued.freedofs.size) < int(componentwise.freedofs.size)

    y0_nodes = np.where(np.abs(np.asarray(componentwise.nodes, dtype=np.float64)[:, 1]) <= 1.0e-12)[0]
    assert y0_nodes.size > 0
    assert np.any(np.any(np.asarray(componentwise.q_mask, dtype=bool)[y0_nodes, :], axis=1))
    assert not np.any(np.asarray(glued.q_mask, dtype=bool)[y0_nodes, :])


def test_same_mesh_transfer_sanity_p2_to_p1_and_p4_to_p2():
    comm = MPI.COMM_SELF

    finest_p2 = build_same_mesh_lagrange_case_data("hetero_ssr_L1", degree=2, build_mode="replicated", comm=comm)
    params_p2 = dict(finest_p2.__dict__)
    params_p2["elem_type"] = "P2"
    params_p2["element_degree"] = 2
    perm_p2 = select_reordered_perm_3d(
        "block_xyz",
        adjacency=finest_p2.adjacency,
        coords_all=params_p2["nodes"],
        freedofs=params_p2["freedofs"],
        n_parts=1,
    )
    hierarchy_p2 = multigrid.build_mixed_pmg_hierarchy(
        specs=multigrid.mixed_hierarchy_specs(
            mesh_name="hetero_ssr_L1",
            finest_degree=2,
            strategy="same_mesh_p2_p1",
        ),
        finest_params=params_p2,
        finest_adjacency=finest_p2.adjacency,
        finest_perm=perm_p2,
        reorder_mode="block_xyz",
        comm=comm,
        level_build_mode="replicated",
        transfer_build_mode="owned_rows",
    )
    coarse, fine = hierarchy_p2.levels
    prolong = hierarchy_p2.prolongations[0]
    def p1_bc_compatible_field(nodes: np.ndarray) -> np.ndarray:
        y = nodes[:, 1]
        return np.column_stack((np.zeros_like(y), 0.05 * y, np.zeros_like(y)))

    coarse_full = p1_bc_compatible_field(np.asarray(coarse.params["nodes"], dtype=np.float64)).reshape(-1)
    fine_full = p1_bc_compatible_field(np.asarray(fine.params["nodes"], dtype=np.float64)).reshape(-1)
    coarse_vec = PETSc.Vec().createMPI((coarse.n_free, coarse.n_free), comm=comm)
    fine_vec = PETSc.Vec().createMPI((fine.n_free, fine.n_free), comm=comm)
    coarse_vec.array[:] = coarse_full[np.asarray(coarse.params["freedofs"], dtype=np.int64)][coarse.perm]
    prolong.mult(coarse_vec, fine_vec)
    expected_fine = fine_full[np.asarray(fine.params["freedofs"], dtype=np.int64)][fine.perm]
    np.testing.assert_allclose(np.asarray(fine_vec.array[:], dtype=np.float64), expected_fine, atol=1.0e-10)
    hierarchy_p2.cleanup()

    finest_p4 = build_same_mesh_lagrange_case_data("hetero_ssr_L1", degree=4, build_mode="replicated", comm=comm)
    params_p4 = dict(finest_p4.__dict__)
    params_p4["elem_type"] = "P4"
    params_p4["element_degree"] = 4
    perm_p4 = select_reordered_perm_3d(
        "block_xyz",
        adjacency=finest_p4.adjacency,
        coords_all=params_p4["nodes"],
        freedofs=params_p4["freedofs"],
        n_parts=1,
    )
    hierarchy_p4 = multigrid.build_mixed_pmg_hierarchy(
        specs=multigrid.mixed_hierarchy_specs(
            mesh_name="hetero_ssr_L1",
            finest_degree=4,
            strategy="same_mesh_p4_p2_p1",
        ),
        finest_params=params_p4,
        finest_adjacency=finest_p4.adjacency,
        finest_perm=perm_p4,
        reorder_mode="block_xyz",
        comm=comm,
        level_build_mode="replicated",
        transfer_build_mode="owned_rows",
    )
    coarse_p2 = hierarchy_p4.levels[1]
    fine_p4 = hierarchy_p4.levels[2]
    prolong_p42 = hierarchy_p4.prolongations[1]
    def p2_bc_compatible_field(nodes: np.ndarray) -> np.ndarray:
        x = nodes[:, 0]
        y = nodes[:, 1]
        z = nodes[:, 2]
        return np.column_stack(
            (
                np.zeros_like(y),
                0.02 * y * (1.0 + 0.01 * x - 0.02 * z),
                np.zeros_like(y),
            )
        )

    coarse_full_p2 = p2_bc_compatible_field(np.asarray(coarse_p2.params["nodes"], dtype=np.float64)).reshape(-1)
    fine_full_p4 = p2_bc_compatible_field(np.asarray(fine_p4.params["nodes"], dtype=np.float64)).reshape(-1)
    coarse_vec_p2 = PETSc.Vec().createMPI((coarse_p2.n_free, coarse_p2.n_free), comm=comm)
    fine_vec_p4 = PETSc.Vec().createMPI((fine_p4.n_free, fine_p4.n_free), comm=comm)
    coarse_vec_p2.array[:] = coarse_full_p2[np.asarray(coarse_p2.params["freedofs"], dtype=np.int64)][coarse_p2.perm]
    prolong_p42.mult(coarse_vec_p2, fine_vec_p4)
    expected_p4 = fine_full_p4[np.asarray(fine_p4.params["freedofs"], dtype=np.int64)][fine_p4.perm]
    np.testing.assert_allclose(np.asarray(fine_vec_p4.array[:], dtype=np.float64), expected_p4, atol=1.0e-10)
    hierarchy_p4.cleanup()


def test_vectorized_transfer_entries_match_reference_builders():
    comm = MPI.COMM_SELF

    finest_p2 = build_same_mesh_lagrange_case_data(
        "hetero_ssr_L1", degree=2, build_mode="replicated", comm=comm
    )
    params_p2 = dict(finest_p2.__dict__)
    params_p2["elem_type"] = "P2"
    params_p2["element_degree"] = 2
    perm_p2 = select_reordered_perm_3d(
        "block_xyz",
        adjacency=finest_p2.adjacency,
        coords_all=params_p2["nodes"],
        freedofs=params_p2["freedofs"],
        n_parts=1,
    )
    hierarchy_p2 = multigrid.build_mixed_pmg_hierarchy(
        specs=multigrid.mixed_hierarchy_specs(
            mesh_name="hetero_ssr_L1",
            finest_degree=2,
            strategy="same_mesh_p2_p1",
        ),
        finest_params=params_p2,
        finest_adjacency=finest_p2.adjacency,
        finest_perm=perm_p2,
        reorder_mode="block_xyz",
        comm=comm,
        level_build_mode="replicated",
        transfer_build_mode="owned_rows",
    )
    coarse_p1, fine_p2 = hierarchy_p2.levels
    rows_new, cols_new, data_new = multigrid._adjacent_same_mesh_prolongation_entries(
        coarse_p1,
        fine_p2,
        build_mode="replicated",
    )
    rows_ref, cols_ref, data_ref = _reference_same_mesh_transfer_entries_dict(
        coarse_p1,
        fine_p2,
        build_mode="replicated",
    )
    np.testing.assert_array_equal(rows_new, rows_ref)
    np.testing.assert_array_equal(cols_new, cols_ref)
    np.testing.assert_allclose(data_new, data_ref, rtol=0.0, atol=1.0e-12)
    hierarchy_p2.cleanup()

    params_p2_local = load_same_mesh_case_hdf5_rank_local_light(
        "hetero_ssr_L1",
        2,
        reorder_mode="block_xyz",
        comm=comm,
    )
    hierarchy_p2_local = multigrid.build_mixed_pmg_hierarchy(
        specs=multigrid.mixed_hierarchy_specs(
            mesh_name="hetero_ssr_L1",
            finest_degree=2,
            strategy="same_mesh_p2_p1",
        ),
        finest_params=params_p2_local,
        finest_adjacency=None,
        finest_perm=np.asarray(params_p2_local["_distributed_perm"], dtype=np.int64),
        reorder_mode="block_xyz",
        comm=comm,
        level_build_mode="rank_local",
        transfer_build_mode="owned_rows",
    )
    coarse_p1_local, fine_p2_local = hierarchy_p2_local.levels
    rows_local, cols_local, data_local = multigrid._adjacent_same_mesh_prolongation_entries(
        coarse_p1_local,
        fine_p2_local,
        build_mode="owned_rows",
    )
    rows_ref, cols_ref, data_ref = _reference_same_mesh_transfer_entries_dict(
        coarse_p1_local,
        fine_p2_local,
        build_mode="owned_rows",
    )
    np.testing.assert_array_equal(rows_local, rows_ref)
    np.testing.assert_array_equal(cols_local, cols_ref)
    np.testing.assert_allclose(data_local, data_ref, rtol=0.0, atol=1.0e-12)
    hierarchy_p2_local.cleanup()

    fine_l12 = build_same_mesh_lagrange_case_data(
        "hetero_ssr_L1_2", degree=1, build_mode="replicated", comm=comm
    )
    coarse_l1 = build_same_mesh_lagrange_case_data(
        "hetero_ssr_L1", degree=1, build_mode="replicated", comm=comm
    )
    params_coarse = dict(coarse_l1.__dict__)
    params_coarse["elem_type"] = "P1"
    params_coarse["element_degree"] = 1
    params_fine = dict(fine_l12.__dict__)
    params_fine["elem_type"] = "P1"
    params_fine["element_degree"] = 1
    coarse_level = multigrid._build_level_space(
        mesh_name="hetero_ssr_L1",
        params=params_coarse,
        adjacency=coarse_l1.adjacency,
        reorder_mode="block_xyz",
        comm=comm,
    )
    fine_level = multigrid._build_level_space(
        mesh_name="hetero_ssr_L1_2",
        params=params_fine,
        adjacency=fine_l12.adjacency,
        reorder_mode="block_xyz",
        comm=comm,
    )
    rows_new, cols_new, data_new = multigrid._parent_mesh_prolongation_entries(
        coarse_level,
        fine_level,
        build_mode="replicated",
    )
    rows_ref, cols_ref, data_ref = _reference_parent_transfer_entries_dict(
        coarse_level,
        fine_level,
        build_mode="replicated",
    )
    np.testing.assert_array_equal(rows_new, rows_ref)
    np.testing.assert_array_equal(cols_new, cols_ref)
    np.testing.assert_allclose(data_new, data_ref, rtol=0.0, atol=1.0e-12)


def test_chained_uniform_refinement_aliases_are_parsed_consistently():
    assert base_mesh_name_for_name("hetero_ssr_L1") == "hetero_ssr_L1"
    assert uniform_refinement_steps_for_name("hetero_ssr_L1") == 0
    assert base_mesh_name_for_name("hetero_ssr_L1_2_3") == "hetero_ssr_L1"
    assert uniform_refinement_steps_for_name("hetero_ssr_L1_2_3") == 2
    with pytest.raises(ValueError, match="sequential suffixes"):
        base_mesh_name_for_name("hetero_ssr_L1_2_4")


def test_mixed_hierarchy_specs_support_uniform_refined_p1_chain():
    specs = multigrid.mixed_hierarchy_specs(
        mesh_name="hetero_ssr_L1_2_3",
        finest_degree=1,
        strategy="uniform_refined_p1_chain",
    )
    assert [(spec.mesh_name, spec.degree) for spec in specs] == [
        ("hetero_ssr_L1", 1),
        ("hetero_ssr_L1_2", 1),
        ("hetero_ssr_L1_2_3", 1),
    ]
    with pytest.raises(ValueError, match="requires finest degree 1"):
        multigrid.mixed_hierarchy_specs(
            mesh_name="hetero_ssr_L1_2_3",
            finest_degree=4,
            strategy="uniform_refined_p1_chain",
        )


def test_mixed_hierarchy_specs_allow_degenerate_uniform_refined_p1_chain():
    specs = multigrid.mixed_hierarchy_specs(
        mesh_name="hetero_ssr_L1",
        finest_degree=1,
        strategy="uniform_refined_p1_chain",
    )
    assert [(spec.mesh_name, spec.degree) for spec in specs] == [("hetero_ssr_L1", 1)]


def test_local_pmg_support_handles_single_level_p1_chain():
    support = plasticity3d_mix_runner._build_local_pmg_support(
        backend=None,
        mesh_name="hetero_ssr_L1",
        elem_degree=1,
        lambda_target=1.55,
        pmg_strategy="uniform_refined_p1_chain",
        ksp_rtol=1.0e-1,
        ksp_max_it=100,
        use_near_nullspace=True,
    )
    try:
        assert support.hierarchy is None
        assert support.realized_levels == 1
        assert support.pc_backend == "hypre"
    finally:
        support.close()


def test_local_pmg_support_builds_p2_on_refined_chain():
    ensure_same_mesh_case_hdf5("hetero_ssr_L1_2_3", 2)
    support = plasticity3d_mix_runner._build_local_pmg_support(
        backend=None,
        mesh_name="hetero_ssr_L1_2_3",
        elem_degree=2,
        lambda_target=1.55,
        pmg_strategy="same_mesh_p2_p1",
        ksp_rtol=1.0e-1,
        ksp_max_it=100,
        use_near_nullspace=True,
    )
    try:
        assert support.hierarchy is not None
        assert support.realized_levels == 2
        assert support.pc_backend == "mg"
    finally:
        support.close()


def test_coo_local_backend_matches_global_coo():
    comm = MPI.COMM_SELF
    case = build_same_mesh_lagrange_case_data(
        "hetero_ssr_L1", degree=1, build_mode="replicated", comm=comm
    )
    params = dict(case.__dict__)
    params["elem_type"] = "P1"
    params["element_degree"] = 1
    c_bar_q, sin_phi_q = davis_b_reduction_qp(
        np.asarray(params["c0_q"], dtype=np.float64),
        np.asarray(params["phi_q"], dtype=np.float64),
        np.asarray(params["psi_q"], dtype=np.float64),
        1.0,
    )
    params["c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
    params["sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)
    perm = select_reordered_perm_3d(
        "block_xyz",
        adjacency=case.adjacency,
        coords_all=params["nodes"],
        freedofs=params["freedofs"],
        n_parts=1,
    )
    common_kwargs = dict(
        params=params,
        comm=comm,
        adjacency=case.adjacency,
        ksp_type="cg",
        pc_type="hypre",
        ksp_max_it=20,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        perm_override=perm,
        block_size_override=ownership_block_size_3d(
            np.asarray(params["freedofs"], dtype=np.int64)
        ),
        distribution_strategy="overlap_allgather",
        use_near_nullspace=False,
    )
    assembler_coo = SlopeStability3DReorderedElementAssembler(
        **common_kwargs,
        assembly_backend="coo",
    )
    assembler_local = SlopeStability3DReorderedElementAssembler(
        **common_kwargs,
        assembly_backend="coo_local",
    )
    rng = np.random.default_rng(42)
    u_owned = 1.0e-6 * rng.standard_normal(assembler_coo.layout.hi - assembler_coo.layout.lo)
    assembler_coo.assemble_hessian(u_owned)
    assembler_local.assemble_hessian(u_owned)
    ia_coo, ja_coo, a_coo = assembler_coo.A.getValuesCSR()
    ia_local, ja_local, a_local = assembler_local.A.getValuesCSR()
    np.testing.assert_array_equal(np.asarray(ia_coo, dtype=np.int64), np.asarray(ia_local, dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(ja_coo, dtype=np.int64), np.asarray(ja_local, dtype=np.int64))
    np.testing.assert_allclose(np.asarray(a_coo, dtype=np.float64), np.asarray(a_local, dtype=np.float64), rtol=1.0e-12, atol=1.0e-12)
    assembler_coo.cleanup()
    assembler_local.cleanup()


def test_constitutive_autodiff_assembler_matches_element_autodiff():
    comm = MPI.COMM_SELF
    case = build_same_mesh_lagrange_case_data(
        "hetero_ssr_L1", degree=1, build_mode="replicated", comm=comm
    )
    params = dict(case.__dict__)
    params["elem_type"] = "P1"
    params["element_degree"] = 1
    c_bar_q, sin_phi_q = davis_b_reduction_qp(
        np.asarray(params["c0_q"], dtype=np.float64),
        np.asarray(params["phi_q"], dtype=np.float64),
        np.asarray(params["psi_q"], dtype=np.float64),
        1.0,
    )
    params["c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
    params["sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)
    perm = select_reordered_perm_3d(
        "block_xyz",
        adjacency=case.adjacency,
        coords_all=params["nodes"],
        freedofs=params["freedofs"],
        n_parts=1,
    )
    common_kwargs = dict(
        params=params,
        comm=comm,
        adjacency=case.adjacency,
        ksp_type="cg",
        pc_type="hypre",
        ksp_max_it=20,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        perm_override=perm,
        block_size_override=ownership_block_size_3d(
            np.asarray(params["freedofs"], dtype=np.int64)
        ),
        distribution_strategy="overlap_allgather",
        use_near_nullspace=False,
        assembly_backend="coo",
    )
    assembler_element = SlopeStability3DReorderedElementAssembler(
        **common_kwargs,
        autodiff_tangent_mode="element",
    )
    assembler_constitutive = SlopeStability3DReorderedElementAssembler(
        **common_kwargs,
        autodiff_tangent_mode="constitutive",
    )
    rng = np.random.default_rng(123)
    u_owned = 1.0e-6 * rng.standard_normal(
        assembler_element.layout.hi - assembler_element.layout.lo
    )
    assembler_element.assemble_hessian(u_owned)
    assembler_constitutive.assemble_hessian(u_owned)
    ia_elem, ja_elem, a_elem = assembler_element.A.getValuesCSR()
    ia_const, ja_const, a_const = assembler_constitutive.A.getValuesCSR()
    np.testing.assert_array_equal(
        np.asarray(ia_elem, dtype=np.int64), np.asarray(ia_const, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        np.asarray(ja_elem, dtype=np.int64), np.asarray(ja_const, dtype=np.int64)
    )
    np.testing.assert_allclose(
        np.asarray(a_const, dtype=np.float64),
        np.asarray(a_elem, dtype=np.float64),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assembler_element.cleanup()
    assembler_constitutive.cleanup()


def test_chunked_p4_hessian_smoke():
    case, c_bar_q, sin_phi_q = _element_args_for_degree(4)
    u_batch = np.zeros((2, case.elems.shape[1]), dtype=np.float64)
    hess = chunked_vmapped_element_hessian_3d(
        jax.numpy.asarray(u_batch),
        jax.numpy.asarray(case.dphix[:2]),
        jax.numpy.asarray(case.dphiy[:2]),
        jax.numpy.asarray(case.dphiz[:2]),
        jax.numpy.asarray(case.quad_weight[:2]),
        jax.numpy.asarray(c_bar_q[:2]),
        jax.numpy.asarray(sin_phi_q[:2]),
        jax.numpy.asarray(case.shear_q[:2]),
        jax.numpy.asarray(case.bulk_q[:2]),
        jax.numpy.asarray(case.lame_q[:2]),
        chunk_size=1,
    )
    arr = np.asarray(hess, dtype=np.float64)
    assert arr.shape == (2, case.elems.shape[1], case.elems.shape[1])
    assert np.all(np.isfinite(arr))


def test_chunked_p4_constitutive_hessian_matches_element_autodiff():
    case, c_bar_q, sin_phi_q = _element_args_for_degree(4)
    rng = np.random.default_rng(9)
    u_batch = 1.0e-6 * rng.standard_normal((1, case.elems.shape[1]))
    hess_element = chunked_vmapped_element_hessian_3d(
        jax.numpy.asarray(u_batch),
        jax.numpy.asarray(case.dphix[:1]),
        jax.numpy.asarray(case.dphiy[:1]),
        jax.numpy.asarray(case.dphiz[:1]),
        jax.numpy.asarray(case.quad_weight[:1]),
        jax.numpy.asarray(c_bar_q[:1]),
        jax.numpy.asarray(sin_phi_q[:1]),
        jax.numpy.asarray(case.shear_q[:1]),
        jax.numpy.asarray(case.bulk_q[:1]),
        jax.numpy.asarray(case.lame_q[:1]),
        chunk_size=1,
    )
    hess_constitutive = chunked_vmapped_element_constitutive_hessian_3d(
        jax.numpy.asarray(u_batch),
        jax.numpy.asarray(case.dphix[:1]),
        jax.numpy.asarray(case.dphiy[:1]),
        jax.numpy.asarray(case.dphiz[:1]),
        jax.numpy.asarray(case.quad_weight[:1]),
        jax.numpy.asarray(c_bar_q[:1]),
        jax.numpy.asarray(sin_phi_q[:1]),
        jax.numpy.asarray(case.shear_q[:1]),
        jax.numpy.asarray(case.bulk_q[:1]),
        jax.numpy.asarray(case.lame_q[:1]),
        chunk_size=1,
    )
    np.testing.assert_allclose(
        np.asarray(hess_constitutive, dtype=np.float64),
        np.asarray(hess_element, dtype=np.float64),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_p2_gamg_smoke(tmp_path: Path):
    output_path = tmp_path / "p2_gamg.json"
    payload = _run_json(
        _solver_command(
            output_path,
            elem_degree=2,
            pc_type="gamg",
            ksp_type="gmres",
            problem_build_mode="replicated",
            distribution_strategy="overlap_allgather",
        ),
        output_path,
    )
    assert payload["elem_degree"] == 2
    assert payload["pc_type"] == "gamg"
    assert int(payload["nit"]) == 1
    assert float(payload["energy"]) < 0.0
    assert int(payload["linear_iterations_last"]) > 0


def test_distributed_p2_and_pmg_smoke(tmp_path: Path):
    output_path = tmp_path / "p2_mg_ranklocal.json"
    payload = _run_json(
        _solver_command(
            output_path,
            nprocs=2,
            elem_degree=2,
            pc_type="mg",
            ksp_type="gmres",
            problem_build_mode="rank_local",
            mg_level_build_mode="rank_local",
            mg_transfer_build_mode="owned_rows",
            mg_strategy="same_mesh_p2_p1",
            distribution_strategy="overlap_p2p",
        ),
        output_path,
    )
    assert payload["elem_degree"] == 2
    assert payload["pc_type"] == "mg"
    assert int(payload["nit"]) == 1
    assert int(payload["linear_iterations_last"]) > 0
    assert "mg_hierarchy" in payload
    assert int(payload["mg_hierarchy"]["level_records"][0]["degree"]) == 1
