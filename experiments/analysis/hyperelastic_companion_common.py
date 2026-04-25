from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.problem_data.hdf5 import load_problem_hdf5, mesh_data_path


DEFAULT_LEVEL = 1
DEFAULT_DISPLACEMENT_SCHEDULE = (0.0025, 0.0050, 0.0075, 0.0100)


def hyperelastic_mesh_path(level: int = DEFAULT_LEVEL) -> Path:
    return mesh_data_path("HyperElasticity", f"HyperElasticity_level{int(level)}.h5")


def load_hyperelastic_case(level: int = DEFAULT_LEVEL) -> dict[str, object]:
    params, adjacency = load_problem_hdf5(hyperelastic_mesh_path(level))
    coords_ref = np.asarray(params["nodes2coord"], dtype=np.float64)
    tetrahedra = np.asarray(params["elems2nodes"], dtype=np.int32)
    freedofs = np.asarray(params["dofsMinim"], dtype=np.int64).reshape((-1,))
    u0_reference = np.asarray(params["u0"], dtype=np.float64).reshape((-1,))
    case = {
        "level": int(level),
        "mesh_path": str(hyperelastic_mesh_path(level)),
        "params": params,
        "adjacency": adjacency,
        "coords_ref": coords_ref,
        "tetrahedra": tetrahedra,
        "freedofs": freedofs,
        "u0_reference": u0_reference,
        "C1": float(params["C1"]),
        "D1": float(params["D1"]),
    }
    case.update(face_nodes(coords_ref))
    return case


def face_nodes(coords_ref: np.ndarray, *, atol: float = 1.0e-12) -> dict[str, object]:
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    x_min = float(np.min(coords_ref[:, 0]))
    x_max = float(np.max(coords_ref[:, 0]))
    left_nodes = np.flatnonzero(np.isclose(coords_ref[:, 0], x_min, atol=atol))
    right_nodes = np.flatnonzero(np.isclose(coords_ref[:, 0], x_max, atol=atol))
    return {
        "x_min": x_min,
        "x_max": x_max,
        "left_nodes": left_nodes.astype(np.int64),
        "right_nodes": right_nodes.astype(np.int64),
    }


def prescribe_right_face_translation(
    u0_reference: np.ndarray,
    coords_ref: np.ndarray,
    displacement_x: float,
    *,
    right_nodes: np.ndarray | None = None,
) -> np.ndarray:
    u0_step = np.asarray(u0_reference, dtype=np.float64).reshape((-1,)).copy()
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    if right_nodes is None:
        right_nodes = face_nodes(coords_ref)["right_nodes"]
    right_nodes = np.asarray(right_nodes, dtype=np.int64).reshape((-1,))
    u0_step[3 * right_nodes + 0] = coords_ref[right_nodes, 0] + float(displacement_x)
    u0_step[3 * right_nodes + 1] = coords_ref[right_nodes, 1]
    u0_step[3 * right_nodes + 2] = coords_ref[right_nodes, 2]
    return u0_step


def full_coordinates_from_free_vector(
    u_free: np.ndarray,
    u0_step: np.ndarray,
    freedofs: np.ndarray,
) -> np.ndarray:
    x_full = np.asarray(u0_step, dtype=np.float64).reshape((-1,)).copy()
    x_full[np.asarray(freedofs, dtype=np.int64)] = np.asarray(u_free, dtype=np.float64).reshape((-1,))
    return x_full


def compute_energy_from_full_coordinates(x_full: np.ndarray, params: dict[str, object]) -> float:
    x_full = np.asarray(x_full, dtype=np.float64).reshape((-1,))
    elems = np.asarray(params["elems2nodes"], dtype=np.int32)
    dvx = np.asarray(params["dphix"], dtype=np.float64)
    dvy = np.asarray(params["dphiy"], dtype=np.float64)
    dvz = np.asarray(params["dphiz"], dtype=np.float64)
    vol = np.asarray(params["vol"], dtype=np.float64).reshape((-1,))
    c1 = float(params["C1"])
    d1 = float(params["D1"])

    vx_elem = x_full[0::3][elems]
    vy_elem = x_full[1::3][elems]
    vz_elem = x_full[2::3][elems]

    f11 = np.sum(vx_elem * dvx, axis=1)
    f12 = np.sum(vx_elem * dvy, axis=1)
    f13 = np.sum(vx_elem * dvz, axis=1)
    f21 = np.sum(vy_elem * dvx, axis=1)
    f22 = np.sum(vy_elem * dvy, axis=1)
    f23 = np.sum(vy_elem * dvz, axis=1)
    f31 = np.sum(vz_elem * dvx, axis=1)
    f32 = np.sum(vz_elem * dvy, axis=1)
    f33 = np.sum(vz_elem * dvz, axis=1)

    i1 = (
        f11**2
        + f12**2
        + f13**2
        + f21**2
        + f22**2
        + f23**2
        + f31**2
        + f32**2
        + f33**2
    )
    detf = np.abs(
        f11 * f22 * f33
        - f11 * f23 * f32
        - f12 * f21 * f33
        + f12 * f23 * f31
        + f13 * f21 * f32
        - f13 * f22 * f31
    )
    detf = np.maximum(detf, 1.0e-12)
    w = c1 * (i1 - 3.0 - 2.0 * np.log(detf)) + d1 * (detf - 1.0) ** 2
    return float(np.sum(w * vol))


def displacement_from_full_coordinates(x_full: np.ndarray, coords_ref: np.ndarray) -> np.ndarray:
    x_full = np.asarray(x_full, dtype=np.float64).reshape((-1, 3))
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    return x_full - coords_ref


def max_displacement_norm(displacement: np.ndarray) -> float:
    displacement = np.asarray(displacement, dtype=np.float64).reshape((-1, 3))
    return float(np.max(np.linalg.norm(displacement, axis=1)))


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    denom = float(np.linalg.norm(ref.reshape(-1)))
    if denom <= 0.0:
        return 0.0 if float(np.linalg.norm(cand.reshape(-1))) <= 0.0 else float("inf")
    return float(np.linalg.norm((cand - ref).reshape(-1)) / denom)


def centerline_profile(
    coords_ref: np.ndarray,
    displacement: np.ndarray,
    *,
    y_target: float = 0.0,
    z_target: float = 0.0,
    atol: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    coords_ref = np.asarray(coords_ref, dtype=np.float64).reshape((-1, 3))
    displacement = np.asarray(displacement, dtype=np.float64).reshape((-1, 3))
    mask = np.isclose(coords_ref[:, 1], float(y_target), atol=atol) & np.isclose(
        coords_ref[:, 2], float(z_target), atol=atol
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("No centerline nodes found for the requested target and tolerance.")
    order = np.argsort(coords_ref[indices, 0])
    node_ids = indices[order]
    return {
        "node_ids": node_ids.astype(np.int64),
        "x": coords_ref[node_ids, 0].astype(np.float64),
        "ux": displacement[node_ids, 0].astype(np.float64),
        "uy": displacement[node_ids, 1].astype(np.float64),
        "uz": displacement[node_ids, 2].astype(np.float64),
        "umag": np.linalg.norm(displacement[node_ids], axis=1).astype(np.float64),
    }


def step_schedule(values: tuple[float, ...] | list[float] | np.ndarray | None = None) -> list[float]:
    if values is None:
        values = DEFAULT_DISPLACEMENT_SCHEDULE
    return [float(value) for value in values]
