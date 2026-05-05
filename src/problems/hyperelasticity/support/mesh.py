from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from mpi4py import MPI

from src.core.petsc.dof_partition import _rank_of_dof_vec, petsc_ownership_range
from src.core.problem_data.hdf5 import load_problem_hdf5, mesh_data_path


LENGTH_X = 0.4
HALF_WIDTH = 0.005
C1 = 38461538.461538464
D1 = 83333333.33333333

# Cube node order:
# n000, n100, n010, n110, n001, n101, n011, n111.
TET_TEMPLATE = np.array(
    [
        [0, 1, 2, 5],
        [0, 2, 4, 5],
        [2, 4, 5, 6],
        [1, 3, 2, 5],
        [3, 5, 7, 2],
        [2, 5, 7, 6],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class HyperElasticityGrid:
    nx: int
    ny: int
    nz: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def nx1(self) -> int:
        return int(self.nx) + 1

    @property
    def ny1(self) -> int:
        return int(self.ny) + 1

    @property
    def nz1(self) -> int:
        return int(self.nz) + 1

    @property
    def n_nodes(self) -> int:
        return self.nx1 * self.ny1 * self.nz1

    @property
    def n_total_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_free_nodes(self) -> int:
        return (int(self.nx) - 1) * self.ny1 * self.nz1

    @property
    def n_free_dofs(self) -> int:
        return 3 * self.n_free_nodes


def expand_tet_connectivity_to_dofs(elems2nodes):
    """Expand scalar-node tetra connectivity to flat 3-DOF connectivity.

    scalar tet: [n0, n1, n2, ...]
    dof tet:    [3*n0,3*n0+1,3*n0+2, ..., 3*nk+2]
    """
    elems2nodes = np.asarray(elems2nodes, dtype=np.int64)
    dof_offsets = np.arange(3, dtype=np.int64)
    return (3 * elems2nodes[:, :, None] + dof_offsets[None, None, :]).reshape(
        elems2nodes.shape[0], 3 * elems2nodes.shape[1]
    )


def dimensions_for_level(level: int) -> tuple[int, int, int]:
    if int(level) < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    return 80 * (2 ** (int(level) - 1)), 2 ** int(level), 2 ** int(level)


def grid_for_level(level: int) -> HyperElasticityGrid:
    nx, ny, nz = dimensions_for_level(int(level))
    return HyperElasticityGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        x_min=0.0,
        x_max=LENGTH_X,
        y_min=-HALF_WIDTH,
        y_max=HALF_WIDTH,
        z_min=-HALF_WIDTH,
        z_max=HALF_WIDTH,
    )


def _read_grid_metadata(path: str, level: int) -> HyperElasticityGrid:
    nx, ny, nz = dimensions_for_level(int(level))
    with h5py.File(path, "r") as handle:
        n_nodes = int(handle["nodes2coord"].shape[0])
        n_elems = int(handle["elems2nodes"].shape[0])
        expected_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        expected_elems = 6 * nx * ny * nz
        if n_nodes != expected_nodes or n_elems != expected_elems:
            raise ValueError(
                f"HyperElasticity level {level} HDF5 shape mismatch: "
                f"nodes={n_nodes} expected={expected_nodes}, "
                f"elems={n_elems} expected={expected_elems}"
            )
        coords = handle["nodes2coord"]
        x_min = float(coords[0, 0])
        x_max = float(coords[nx, 0])
        y_min = float(coords[0, 1])
        y_max = float(coords[(ny * (nx + 1)), 1])
        z_min = float(coords[0, 2])
        z_max = float(coords[(nz * (nx + 1) * (ny + 1)), 2])
    return HyperElasticityGrid(nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max)


def _cube_offsets(grid: HyperElasticityGrid) -> np.ndarray:
    nx1 = int(grid.nx1)
    ny1 = int(grid.ny1)
    return np.array(
        [
            0,
            1,
            nx1,
            nx1 + 1,
            nx1 * ny1,
            nx1 * ny1 + 1,
            nx1 * ny1 + nx1,
            nx1 * ny1 + nx1 + 1,
        ],
        dtype=np.int64,
    )


def generate_structured_nodes(grid: HyperElasticityGrid) -> np.ndarray:
    x = np.linspace(float(grid.x_min), float(grid.x_max), int(grid.nx1), dtype=np.float64)
    y = np.linspace(float(grid.y_min), float(grid.y_max), int(grid.ny1), dtype=np.float64)
    z = np.linspace(float(grid.z_min), float(grid.z_max), int(grid.nz1), dtype=np.float64)

    coords = np.empty((int(grid.n_nodes), 3), dtype=np.float64)
    coords[:, 0] = np.tile(x, int(grid.ny1) * int(grid.nz1))
    coords[:, 1] = np.tile(np.repeat(y, int(grid.nx1)), int(grid.nz1))
    coords[:, 2] = np.repeat(z, int(grid.nx1) * int(grid.ny1))
    return coords


def generate_structured_elements_for_indices(
    elem_indices: np.ndarray,
    grid: HyperElasticityGrid,
) -> np.ndarray:
    elem = np.asarray(elem_indices, dtype=np.int64).ravel()
    if elem.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    cell = elem // int(TET_TEMPLATE.shape[0])
    tet = elem % int(TET_TEMPLATE.shape[0])
    cx = cell % int(grid.nx)
    plane = cell // int(grid.nx)
    cy = plane % int(grid.ny)
    cz = plane // int(grid.ny)
    base = (cz * int(grid.ny1) + cy) * int(grid.nx1) + cx
    cube = base[:, None] + _cube_offsets(grid)[None, :]
    return cube[np.arange(elem.size)[:, None], TET_TEMPLATE[tet]]


def _reference_gradient_patterns(
    grid: HyperElasticityGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = (float(grid.x_max) - float(grid.x_min)) / float(grid.nx)
    dy = (float(grid.y_max) - float(grid.y_min)) / float(grid.ny)
    dz = (float(grid.z_max) - float(grid.z_min)) / float(grid.nz)
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [dx, 0.0, 0.0],
            [0.0, dy, 0.0],
            [dx, dy, 0.0],
            [0.0, 0.0, dz],
            [dx, 0.0, dz],
            [0.0, dy, dz],
            [dx, dy, dz],
        ],
        dtype=np.float64,
    )

    gradients = np.empty((int(TET_TEMPLATE.shape[0]), 4, 3), dtype=np.float64)
    volumes = np.empty(int(TET_TEMPLATE.shape[0]), dtype=np.float64)
    for idx, tet_nodes in enumerate(TET_TEMPLATE):
        vertices = cube[tet_nodes]
        matrix = np.ones((4, 4), dtype=np.float64)
        matrix[:, 1:] = vertices
        gradients[idx] = np.linalg.inv(matrix)[1:, :].T
        jac = np.column_stack(
            [
                vertices[1] - vertices[0],
                vertices[2] - vertices[0],
                vertices[3] - vertices[0],
            ]
        )
        volumes[idx] = abs(np.linalg.det(jac)) / 6.0

    gradients[np.abs(gradients) < 1e-12] = 0.0
    return gradients[:, :, 0], gradients[:, :, 1], gradients[:, :, 2], volumes


def generate_structured_element_data_for_indices(
    elem_indices: np.ndarray,
    grid: HyperElasticityGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elem = np.asarray(elem_indices, dtype=np.int64).ravel()
    if elem.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    tet = elem % int(TET_TEMPLATE.shape[0])
    dphix_pattern, dphiy_pattern, dphiz_pattern, vol_pattern = (
        _reference_gradient_patterns(grid)
    )
    return (
        np.asarray(dphix_pattern[tet], dtype=np.float64),
        np.asarray(dphiy_pattern[tet], dtype=np.float64),
        np.asarray(dphiz_pattern[tet], dtype=np.float64),
        np.asarray(vol_pattern[tet], dtype=np.float64),
    )


def _require_supported_element_degree(element_degree: int) -> int:
    degree = int(element_degree)
    if degree not in {1, 4}:
        raise ValueError(
            f"HyperElasticity element_degree={degree!r} is not supported; "
            "expected 1 or 4"
        )
    return degree


def _degree_node_shape(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> tuple[int, int, int]:
    degree = _require_supported_element_degree(int(element_degree))
    return (
        degree * int(grid.nx) + 1,
        degree * int(grid.ny) + 1,
        degree * int(grid.nz) + 1,
    )


def n_nodes_for_element_degree(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> int:
    qnx1, qny1, qnz1 = _degree_node_shape(grid, int(element_degree))
    return int(qnx1) * int(qny1) * int(qnz1)


def n_free_nodes_for_element_degree(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> int:
    degree = _require_supported_element_degree(int(element_degree))
    return (degree * int(grid.nx) - 1) * (
        degree * int(grid.ny) + 1
    ) * (degree * int(grid.nz) + 1)


def n_total_dofs_for_element_degree(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> int:
    return 3 * n_nodes_for_element_degree(grid, int(element_degree))


def n_free_dofs_for_element_degree(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> int:
    return 3 * n_free_nodes_for_element_degree(grid, int(element_degree))


def _degree_node_ijk(
    node_ids: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.asarray(node_ids, dtype=np.int64)
    qnx1, qny1, _ = _degree_node_shape(grid, int(element_degree))
    ix = node % int(qnx1)
    plane = node // int(qnx1)
    iy = plane % int(qny1)
    iz = plane // int(qny1)
    return ix, iy, iz


def _degree_node_coordinates(
    node_ids: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    ix, iy, iz = _degree_node_ijk(np.asarray(node_ids, dtype=np.int64), grid, degree)
    coords = np.empty((ix.size, 3), dtype=np.float64)
    coords[:, 0] = float(grid.x_min) + (float(grid.x_max) - float(grid.x_min)) * (
        ix.astype(np.float64) / float(degree * int(grid.nx))
    )
    coords[:, 1] = float(grid.y_min) + (float(grid.y_max) - float(grid.y_min)) * (
        iy.astype(np.float64) / float(degree * int(grid.ny))
    )
    coords[:, 2] = float(grid.z_min) + (float(grid.z_max) - float(grid.z_min)) * (
        iz.astype(np.float64) / float(degree * int(grid.nz))
    )
    return coords


def _tetra_lagrange_node_tuples(element_degree: int) -> tuple[tuple[int, int, int, int], ...]:
    from src.problems.slope_stability_3d.support.simplex_lagrange import (
        tetra_lagrange_node_tuples,
    )

    return tetra_lagrange_node_tuples(int(element_degree))


def generate_structured_lagrange_elements_for_indices(
    elem_indices: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    if degree == 1:
        return generate_structured_elements_for_indices(elem_indices, grid)

    elem = np.asarray(elem_indices, dtype=np.int64).ravel()
    if elem.size == 0:
        return np.zeros((0, len(_tetra_lagrange_node_tuples(degree))), dtype=np.int64)

    cell = elem // int(TET_TEMPLATE.shape[0])
    tet = elem % int(TET_TEMPLATE.shape[0])
    cx = cell % int(grid.nx)
    plane = cell // int(grid.nx)
    cy = plane % int(grid.ny)
    cz = plane // int(grid.ny)

    cube_offsets_ijk = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    base_ijk = np.stack((cx, cy, cz), axis=1)
    cube_ijk = base_ijk[:, None, :] + cube_offsets_ijk[None, :, :]
    tet_ijk = cube_ijk[np.arange(elem.size)[:, None], TET_TEMPLATE[tet]]

    tuples = np.asarray(_tetra_lagrange_node_tuples(degree), dtype=np.int64)
    qijk = np.einsum("ap,epd->ead", tuples, tet_ijk)
    qnx1, qny1, _ = _degree_node_shape(grid, degree)
    return (
        (qijk[:, :, 2] * int(qny1) + qijk[:, :, 1]) * int(qnx1)
        + qijk[:, :, 0]
    ).astype(np.int64, copy=False)


def _quadrature_volume_3d(element_degree: int) -> tuple[np.ndarray, np.ndarray]:
    from src.problems.slope_stability_3d.support.mesh import _quadrature_volume_3d as _quad

    return _quad(int(element_degree))


def _reference_lagrange_element_patterns(
    grid: HyperElasticityGrid,
    element_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.problems.slope_stability_3d.support.simplex_lagrange import (
        evaluate_tetra_lagrange_basis,
    )

    degree = _require_supported_element_degree(int(element_degree))
    if degree == 1:
        dphix, dphiy, dphiz, vol = _reference_gradient_patterns(grid)
        return dphix[:, None, :], dphiy[:, None, :], dphiz[:, None, :], vol[:, None]

    dx = (float(grid.x_max) - float(grid.x_min)) / float(grid.nx)
    dy = (float(grid.y_max) - float(grid.y_min)) / float(grid.ny)
    dz = (float(grid.z_max) - float(grid.z_min)) / float(grid.nz)
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [dx, 0.0, 0.0],
            [0.0, dy, 0.0],
            [dx, dy, 0.0],
            [0.0, 0.0, dz],
            [dx, 0.0, dz],
            [0.0, dy, dz],
            [dx, dy, dz],
        ],
        dtype=np.float64,
    )
    tuples = np.asarray(_tetra_lagrange_node_tuples(degree), dtype=np.float64)
    xi, weights = _quadrature_volume_3d(degree)
    _, dhat1, dhat2, dhat3 = evaluate_tetra_lagrange_basis(degree, xi)
    n_tets = int(TET_TEMPLATE.shape[0])
    n_q = int(xi.shape[1])
    n_p = int(tuples.shape[0])

    dphix = np.empty((n_tets, n_q, n_p), dtype=np.float64)
    dphiy = np.empty((n_tets, n_q, n_p), dtype=np.float64)
    dphiz = np.empty((n_tets, n_q, n_p), dtype=np.float64)
    quad_weight = np.empty((n_tets, n_q), dtype=np.float64)

    for tet_id, tet_nodes in enumerate(TET_TEMPLATE):
        vertices = cube[np.asarray(tet_nodes, dtype=np.int64)]
        elem_coords = (tuples @ vertices) / float(degree)
        xcoord = elem_coords[:, 0]
        ycoord = elem_coords[:, 1]
        zcoord = elem_coords[:, 2]
        for q in range(n_q):
            dh1 = np.asarray(dhat1[:, q], dtype=np.float64)
            dh2 = np.asarray(dhat2[:, q], dtype=np.float64)
            dh3 = np.asarray(dhat3[:, q], dtype=np.float64)

            j11 = float(xcoord @ dh1)
            j12 = float(ycoord @ dh1)
            j13 = float(zcoord @ dh1)
            j21 = float(xcoord @ dh2)
            j22 = float(ycoord @ dh2)
            j23 = float(zcoord @ dh2)
            j31 = float(xcoord @ dh3)
            j32 = float(ycoord @ dh3)
            j33 = float(zcoord @ dh3)
            det_j = (
                j11 * (j22 * j33 - j23 * j32)
                - j12 * (j21 * j33 - j23 * j31)
                + j13 * (j21 * j32 - j22 * j31)
            )
            inv_det = 1.0 / det_j
            dphix[tet_id, q, :] = (
                ((j22 * j33 - j23 * j32) * dh1)
                - ((j12 * j33 - j13 * j32) * dh2)
                + ((j12 * j23 - j13 * j22) * dh3)
            ) * inv_det
            dphiy[tet_id, q, :] = (
                (-(j21 * j33 - j23 * j31) * dh1)
                + ((j11 * j33 - j13 * j31) * dh2)
                - ((j11 * j23 - j13 * j21) * dh3)
            ) * inv_det
            dphiz[tet_id, q, :] = (
                ((j21 * j32 - j22 * j31) * dh1)
                - ((j11 * j32 - j12 * j31) * dh2)
                + ((j11 * j22 - j12 * j21) * dh3)
            ) * inv_det
            quad_weight[tet_id, q] = abs(det_j) * float(weights[q])

    dphix[np.abs(dphix) < 1e-12] = 0.0
    dphiy[np.abs(dphiy) < 1e-12] = 0.0
    dphiz[np.abs(dphiz) < 1e-12] = 0.0
    return dphix, dphiy, dphiz, quad_weight


def generate_structured_lagrange_element_data_for_indices(
    elem_indices: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    degree = _require_supported_element_degree(int(element_degree))
    if degree == 1:
        return generate_structured_element_data_for_indices(elem_indices, grid)

    elem = np.asarray(elem_indices, dtype=np.int64).ravel()
    n_p = len(_tetra_lagrange_node_tuples(degree))
    if elem.size == 0:
        xi, _ = _quadrature_volume_3d(degree)
        n_q = int(xi.shape[1])
        return (
            np.zeros((0, n_q, n_p), dtype=np.float64),
            np.zeros((0, n_q, n_p), dtype=np.float64),
            np.zeros((0, n_q, n_p), dtype=np.float64),
            np.zeros((0, n_q), dtype=np.float64),
        )
    tet = elem % int(TET_TEMPLATE.shape[0])
    dphix_pattern, dphiy_pattern, dphiz_pattern, weight_pattern = (
        _reference_lagrange_element_patterns(grid, degree)
    )
    return (
        np.asarray(dphix_pattern[tet], dtype=np.float64),
        np.asarray(dphiy_pattern[tet], dtype=np.float64),
        np.asarray(dphiz_pattern[tet], dtype=np.float64),
        np.asarray(weight_pattern[tet], dtype=np.float64),
    )


def _index_dtype(n_free: int) -> type[np.integer]:
    return np.int32 if int(n_free) <= int(np.iinfo(np.int32).max) else np.int64


def _node_ijk(node_ids: np.ndarray, grid: HyperElasticityGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.asarray(node_ids, dtype=np.int64)
    ix = node % int(grid.nx1)
    plane = node // int(grid.nx1)
    iy = plane % int(grid.ny1)
    iz = plane // int(grid.ny1)
    return ix, iy, iz


def _node_coordinates(node_ids: np.ndarray, grid: HyperElasticityGrid) -> np.ndarray:
    ix, iy, iz = _node_ijk(np.asarray(node_ids, dtype=np.int64), grid)
    coords = np.empty((ix.size, 3), dtype=np.float64)
    coords[:, 0] = float(grid.x_min) + (float(grid.x_max) - float(grid.x_min)) * (
        ix.astype(np.float64) / float(grid.nx)
    )
    coords[:, 1] = float(grid.y_min) + (float(grid.y_max) - float(grid.y_min)) * (
        iy.astype(np.float64) / float(grid.ny)
    )
    coords[:, 2] = float(grid.z_min) + (float(grid.z_max) - float(grid.z_min)) * (
        iz.astype(np.float64) / float(grid.nz)
    )
    return coords


def _free_block_to_node_ids(
    block_ids: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
    element_degree: int = 1,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    if degree != 1:
        return _free_block_to_degree_node_ids(block_ids, grid, reorder_mode, degree)

    block = np.asarray(block_ids, dtype=np.int64)
    mode = str(reorder_mode)
    if mode == "none":
        x_inner = block % (int(grid.nx) - 1)
        plane = block // (int(grid.nx) - 1)
        iy = plane % int(grid.ny1)
        iz = plane // int(grid.ny1)
    elif mode == "block_xyz":
        iz = block % int(grid.nz1)
        tmp = block // int(grid.nz1)
        iy = tmp % int(grid.ny1)
        x_inner = tmp // int(grid.ny1)
    else:
        raise ValueError(
            f"rank-local HyperElasticity supports element_reorder_mode='none' "
            f"or 'block_xyz', got {mode!r}"
        )
    ix = x_inner + 1
    return (iz * int(grid.ny1) + iy) * int(grid.nx1) + ix


def _free_block_to_degree_node_ids(
    block_ids: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
    element_degree: int,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    block = np.asarray(block_ids, dtype=np.int64)
    qnx1, qny1, qnz1 = _degree_node_shape(grid, degree)
    qnx = int(qnx1) - 1
    mode = str(reorder_mode)
    if mode == "none":
        x_inner = block % (qnx - 1)
        plane = block // (qnx - 1)
        iy = plane % int(qny1)
        iz = plane // int(qny1)
    elif mode == "block_xyz":
        iz = block % int(qnz1)
        tmp = block // int(qnz1)
        iy = tmp % int(qny1)
        x_inner = tmp // int(qny1)
    else:
        raise ValueError(
            f"rank-local HyperElasticity supports element_reorder_mode='none' "
            f"or 'block_xyz', got {mode!r}"
        )
    ix = x_inner + 1
    return (iz * int(qny1) + iy) * int(qnx1) + ix


def reordered_free_to_total_dofs(
    reord_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
    element_degree: int = 1,
) -> np.ndarray:
    reord = np.asarray(reord_dofs, dtype=np.int64)
    block = reord // 3
    comp = reord % 3
    nodes = _free_block_to_node_ids(block, grid, str(reorder_mode), int(element_degree))
    return 3 * nodes + comp


def total_dofs_to_reordered_free(
    total_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
    element_degree: int = 1,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    total = np.asarray(total_dofs, dtype=np.int64)
    node = total // 3
    comp = total % 3
    if degree == 1:
        ix, iy, iz = _node_ijk(node, grid)
        qnx = int(grid.nx)
        qny1 = int(grid.ny1)
        qnz1 = int(grid.nz1)
    else:
        ix, iy, iz = _degree_node_ijk(node, grid, degree)
        qnx1, qny1, qnz1 = _degree_node_shape(grid, degree)
        qnx = int(qnx1) - 1
    free = (ix > 0) & (ix < qnx)
    out = np.full(total.shape, -1, dtype=np.int64)
    if not np.any(free):
        return out
    x_inner = ix[free] - 1
    mode = str(reorder_mode)
    if mode == "none":
        block = (iz[free] * int(qny1) + iy[free]) * (qnx - 1) + x_inner
    elif mode == "block_xyz":
        block = (x_inner * int(qny1) + iy[free]) * int(qnz1) + iz[free]
    else:
        raise ValueError(
            f"rank-local HyperElasticity supports element_reorder_mode='none' "
            f"or 'block_xyz', got {mode!r}"
        )
    out[free] = 3 * block + comp[free]
    return out


def _structured_freedofs(grid: HyperElasticityGrid) -> np.ndarray:
    free_x = np.arange(1, int(grid.nx), dtype=np.int64)
    yz_planes = np.arange(int(grid.ny1) * int(grid.nz1), dtype=np.int64)
    free_nodes = (yz_planes[:, None] * int(grid.nx1) + free_x[None, :]).ravel()
    offsets = np.arange(3, dtype=np.int64)
    return (3 * free_nodes[:, None] + offsets[None, :]).ravel()


def _local_candidate_element_indices(
    owned_node_ids: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int = 1,
) -> np.ndarray:
    if owned_node_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    degree = _require_supported_element_degree(int(element_degree))
    if degree == 1:
        ix, iy, iz = _node_ijk(owned_node_ids, grid)
        cell_x = ix
        cell_y = iy
        cell_z = iz
    else:
        ix, iy, iz = _degree_node_ijk(owned_node_ids, grid, degree)
        cell_x = ix // degree
        cell_y = iy // degree
        cell_z = iz // degree
    cell_batches = []
    for dx in (-1, 0):
        cx = cell_x + dx
        valid_x = (cx >= 0) & (cx < int(grid.nx))
        for dy in (-1, 0):
            cy = cell_y + dy
            valid_xy = valid_x & (cy >= 0) & (cy < int(grid.ny))
            for dz in (-1, 0):
                cz = cell_z + dz
                valid = valid_xy & (cz >= 0) & (cz < int(grid.nz))
                if np.any(valid):
                    cell_batches.append(
                        (cz[valid] * int(grid.ny) + cy[valid]) * int(grid.nx)
                        + cx[valid]
                    )
    if not cell_batches:
        return np.zeros(0, dtype=np.int64)
    cell_ids = np.unique(np.concatenate(cell_batches, axis=0))
    return (cell_ids[:, None] * 6 + np.arange(6, dtype=np.int64)[None, :]).ravel()


def _read_rows_grouped(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
    if np.any(rows[:-1] > rows[1:]):
        raise ValueError("HDF5 row reads require sorted row indices")
    breaks = np.where(np.diff(rows) != 1)[0] + 1
    groups = np.split(rows, breaks)
    chunks = [dataset[int(group[0]) : int(group[-1]) + 1] for group in groups]
    if len(chunks) == 1:
        return np.asarray(chunks[0])
    return np.concatenate(chunks, axis=0)


def _values_for_total_dofs(
    total_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    angle: float,
    element_degree: int = 1,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    total = np.asarray(total_dofs, dtype=np.int64)
    node = total // 3
    comp = total % 3
    if degree == 1:
        ix, _, _ = _node_ijk(node, grid)
        right_ix = int(grid.nx)
        coords = _node_coordinates(node, grid)
    else:
        ix, _, _ = _degree_node_ijk(node, grid, degree)
        right_ix = degree * int(grid.nx)
        coords = _degree_node_coordinates(node, grid, degree)
    values = coords[np.arange(total.size), comp].copy()
    right = ix == int(right_ix)
    if np.any(right):
        y_rot = np.cos(float(angle)) * coords[right, 1] + np.sin(float(angle)) * coords[right, 2]
        z_rot = -np.sin(float(angle)) * coords[right, 1] + np.cos(float(angle)) * coords[right, 2]
        right_indices = np.nonzero(right)[0]
        comp_right = comp[right]
        y_mask = comp_right == 1
        z_mask = comp_right == 2
        values[right_indices[y_mask]] = y_rot[y_mask]
        values[right_indices[z_mask]] = z_rot[z_mask]
    return values


def local_dirichlet_values_from_reference(params: dict[str, object], angle: float) -> np.ndarray:
    grid = params["_he_grid"]
    return _values_for_total_dofs(
        np.asarray(params["_distributed_local_total_nodes"], dtype=np.int64),
        grid,
        float(angle),
        int(params.get("element_degree", 1)),
    )


def _owned_nullspace(
    owned_total_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    element_degree: int = 1,
) -> np.ndarray:
    degree = _require_supported_element_degree(int(element_degree))
    node = np.asarray(owned_total_dofs, dtype=np.int64) // 3
    comp = np.asarray(owned_total_dofs, dtype=np.int64) % 3
    coords = (
        _node_coordinates(node, grid)
        if degree == 1
        else _degree_node_coordinates(node, grid, degree)
    )
    kernel = np.zeros((owned_total_dofs.size, 6), dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    mask_x = comp == 0
    mask_y = comp == 1
    mask_z = comp == 2
    kernel[mask_x, 0] = 1.0
    kernel[mask_y, 1] = 1.0
    kernel[mask_z, 2] = 1.0
    kernel[mask_y, 3] = -z[mask_y]
    kernel[mask_z, 3] = y[mask_z]
    kernel[mask_x, 4] = z[mask_x]
    kernel[mask_z, 4] = -x[mask_z]
    kernel[mask_x, 5] = -y[mask_x]
    kernel[mask_y, 5] = x[mask_y]
    return kernel


def build_procedural_hyperelasticity_export_params(
    mesh_level: int,
) -> dict[str, object]:
    """Build full structured HE metadata needed for state export/validation."""
    grid = grid_for_level(int(mesh_level))
    nodes2coord = generate_structured_nodes(grid)
    elem_idx = np.arange(6 * int(grid.nx) * int(grid.ny) * int(grid.nz), dtype=np.int64)
    elems_scalar = generate_structured_elements_for_indices(elem_idx, grid)
    u0_ref = nodes2coord.ravel()
    freedofs = _structured_freedofs(grid)
    right_nodes = np.where(
        np.isclose(nodes2coord[:, 0], float(grid.x_max))
    )[0].astype(np.int64)
    return {
        "u_0": u0_ref.copy(),
        "u_0_ref": u0_ref.copy(),
        "freedofs": freedofs,
        "elems_scalar": elems_scalar,
        "nodes2coord": nodes2coord,
        "right_nodes": right_nodes,
        "C1": C1,
        "D1": D1,
    }


def load_rank_local_hyperelasticity(
    mesh_level: int,
    *,
    comm: MPI.Comm,
    reorder_mode: str = "block_xyz",
    mesh_source: str = "procedural",
    element_degree: int = 1,
) -> tuple[dict[str, object], None, np.ndarray]:
    """Build only this rank's HyperElasticity overlap domain."""
    degree = _require_supported_element_degree(int(element_degree))
    mode = str(reorder_mode)
    if mode not in {"none", "block_xyz"}:
        raise ValueError(
            f"rank-local HyperElasticity supports element_reorder_mode='none' "
            f"or 'block_xyz', got {mode!r}"
        )
    source = str(mesh_source)
    if source not in {"procedural", "hdf5"}:
        raise ValueError(
            "rank-local HyperElasticity mesh_source must be 'procedural' or "
            f"'hdf5', got {source!r}"
        )
    if degree != 1 and source != "procedural":
        raise ValueError(
            "rank-local HyperElasticity element_degree=4 currently requires "
            "mesh_source='procedural'"
        )

    filename = mesh_data_path("HyperElasticity", f"HyperElasticity_level{int(mesh_level)}.h5")
    grid = (
        grid_for_level(int(mesh_level))
        if source == "procedural"
        else _read_grid_metadata(filename, int(mesh_level))
    )
    n_free = int(n_free_dofs_for_element_degree(grid, degree))
    n_total = int(n_total_dofs_for_element_degree(grid, degree))
    dtype = _index_dtype(n_free)
    lo, hi = petsc_ownership_range(n_free, int(comm.rank), int(comm.size), block_size=3)
    owned_reord = np.arange(lo, hi, dtype=np.int64)
    owned_total_dofs = reordered_free_to_total_dofs(owned_reord, grid, mode, degree)
    owned_node_ids = np.unique(owned_total_dofs // 3)
    local_elem_idx = _local_candidate_element_indices(owned_node_ids, grid, degree)

    if source == "procedural":
        elems_scalar = generate_structured_lagrange_elements_for_indices(
            local_elem_idx,
            grid,
            degree,
        )
        dphix, dphiy, dphiz, vol = generate_structured_lagrange_element_data_for_indices(
            local_elem_idx,
            grid,
            degree,
        )
        c1 = C1
        d1 = D1
    else:
        with h5py.File(filename, "r") as handle:
            elems_scalar = np.asarray(
                _read_rows_grouped(handle["elems2nodes"], local_elem_idx),
                dtype=np.int64,
            )
            dphix = np.asarray(_read_rows_grouped(handle["dphix"], local_elem_idx), dtype=np.float64)
            dphiy = np.asarray(_read_rows_grouped(handle["dphiy"], local_elem_idx), dtype=np.float64)
            dphiz = np.asarray(_read_rows_grouped(handle["dphiz"], local_elem_idx), dtype=np.float64)
            vol = np.asarray(_read_rows_grouped(handle["vol"], local_elem_idx), dtype=np.float64)
            c1 = float(handle["C1"][()])
            d1 = float(handle["D1"][()])

    elems_total = expand_tet_connectivity_to_dofs(elems_scalar)
    elems_reordered = total_dofs_to_reordered_free(elems_total, grid, mode, degree)
    exact_local = np.any((elems_reordered >= int(lo)) & (elems_reordered < int(hi)), axis=1)
    if np.any(~exact_local):
        local_elem_idx = local_elem_idx[exact_local]
        elems_scalar = elems_scalar[exact_local]
        elems_total = elems_total[exact_local]
        elems_reordered = elems_reordered[exact_local]
        dphix = dphix[exact_local]
        dphiy = dphiy[exact_local]
        dphiz = dphiz[exact_local]
        vol = vol[exact_local]

    if elems_total.size == 0:
        local_total_dofs = np.zeros(0, dtype=np.int64)
        elems_local_np = np.zeros((0, 12), dtype=np.int32)
        local_scalar_nodes = np.zeros(0, dtype=np.int64)
        elems_scalar_np = np.zeros((0, 4), dtype=np.int32)
    else:
        local_total_dofs, inverse = np.unique(elems_total.ravel(), return_inverse=True)
        elems_local_np = inverse.reshape(elems_total.shape).astype(np.int32)
        local_scalar_nodes, scalar_inverse = np.unique(
            elems_scalar.ravel(), return_inverse=True
        )
        elems_scalar_np = scalar_inverse.reshape(elems_scalar.shape).astype(np.int32)

    local_total_to_free = total_dofs_to_reordered_free(local_total_dofs, grid, mode, degree)
    masked = np.where(elems_reordered >= 0, elems_reordered, np.int64(n_free))
    elem_min = np.min(masked, axis=1) if elems_reordered.size else np.zeros(0, dtype=np.int64)
    valid = elem_min < int(n_free)
    local_elem_owner = np.full(local_elem_idx.size, -1, dtype=np.int64)
    if np.any(valid):
        local_elem_owner[valid] = _rank_of_dof_vec(
            elem_min[valid],
            int(n_free),
            int(comm.size),
            block_size=3,
        )
    local_energy_weights = (local_elem_owner == int(comm.rank)).astype(np.float64)
    owned_block_ids = np.arange(lo // 3, hi // 3, dtype=np.int64)
    owned_block_nodes = _free_block_to_node_ids(owned_block_ids, grid, mode, degree)
    owned_block_coordinates = (
        _node_coordinates(owned_block_nodes, grid)
        if degree == 1
        else _degree_node_coordinates(owned_block_nodes, grid, degree)
    )
    params: dict[str, object] = {
        "freedofs": np.zeros(0, dtype=dtype),
        "C1": c1,
        "D1": d1,
        "element_degree": int(degree),
        "_he_grid": grid,
        "_distributed_mesh_source": source,
        "_distributed_formula_layout": True,
        "_distributed_local_data": True,
        "_distributed_reorder_mode": mode,
        "_distributed_n_free": int(n_free),
        "_distributed_n_total": int(n_total),
        "_distributed_lo": int(lo),
        "_distributed_hi": int(hi),
        "_distributed_ownership_block_size": 3,
        "_distributed_local_elem_idx": np.asarray(local_elem_idx, dtype=dtype),
        "_distributed_local_elems_total": np.asarray(elems_total, dtype=dtype),
        "_distributed_local_elems_reordered": np.asarray(elems_reordered, dtype=dtype),
        "_distributed_local_total_nodes": np.asarray(local_total_dofs, dtype=dtype),
        "_distributed_elems_local_np": np.asarray(elems_local_np, dtype=np.int32),
        "_distributed_local_elems_scalar_np": np.asarray(elems_scalar_np, dtype=np.int32),
        "_distributed_local_total_to_free_reord": np.asarray(local_total_to_free, dtype=dtype),
        "_distributed_energy_weights": local_energy_weights,
        "_distributed_dphix": dphix,
        "_distributed_dphiy": dphiy,
        "_distributed_dphiz": dphiz,
        "_distributed_vol": vol,
        "_distributed_dirichlet_ref_local": _values_for_total_dofs(
            local_total_dofs, grid, 0.0, degree
        ),
        "_distributed_u_init_owned": _values_for_total_dofs(
            owned_total_dofs, grid, 0.0, degree
        ),
        "_distributed_owned_block_coordinates": owned_block_coordinates,
        "_distributed_owned_nullspace": _owned_nullspace(owned_total_dofs, grid, degree),
        "_distributed_total_dofs": int(n_total),
    }
    return params, None, np.asarray(params["_distributed_u_init_owned"], dtype=np.float64)


class MeshHyperElasticity3D:
    """Load HyperElasticity 3D mesh/problem data from HDF5."""

    def __init__(self, mesh_level):
        self.mesh_level = int(mesh_level)
        self.filename = mesh_data_path(
            "HyperElasticity", f"HyperElasticity_level{self.mesh_level}.h5"
        )
        self._load_data(self.filename)
        self._build_problem_data()

    def _load_data(self, filename):
        self._raw, self.adjacency = load_problem_hdf5(filename)
        if self.adjacency is None:
            raise RuntimeError("Mesh file is missing required 'adjacency' group")

    def _build_problem_data(self):
        nodes2coord = np.asarray(self._raw["nodes2coord"], dtype=np.float64)
        elems_scalar = np.asarray(self._raw["elems2nodes"], dtype=np.int64)
        elems_dof = expand_tet_connectivity_to_dofs(elems_scalar)

        u0_ref = np.asarray(self._raw["u0"], dtype=np.float64).ravel()
        freedofs = np.asarray(self._raw["dofsMinim"], dtype=np.int64).ravel()

        right_x = np.max(nodes2coord[:, 0])
        right_nodes = np.where(np.isclose(nodes2coord[:, 0], right_x))[0].astype(np.int64)

        # Rigid-body near-nullspace in full DOF space, then restricted to free DOFs.
        n_nodes = nodes2coord.shape[0]
        rigid_modes = np.zeros((3 * n_nodes, 6), dtype=np.float64)
        rigid_modes[0::3, 0] = 1.0
        rigid_modes[1::3, 1] = 1.0
        rigid_modes[2::3, 2] = 1.0

        rigid_modes[1::3, 3] = -nodes2coord[:, 2]
        rigid_modes[2::3, 3] = nodes2coord[:, 1]

        rigid_modes[0::3, 4] = nodes2coord[:, 2]
        rigid_modes[2::3, 4] = -nodes2coord[:, 0]

        rigid_modes[0::3, 5] = -nodes2coord[:, 1]
        rigid_modes[1::3, 5] = nodes2coord[:, 0]

        elastic_kernel = rigid_modes[freedofs, :]

        self.params = {
            "u_0": u0_ref.copy(),
            "u_0_ref": u0_ref.copy(),
            "freedofs": freedofs,
            "elems": elems_dof,
            "elems_scalar": elems_scalar,
            "dphix": np.asarray(self._raw["dphix"], dtype=np.float64),
            "dphiy": np.asarray(self._raw["dphiy"], dtype=np.float64),
            "dphiz": np.asarray(self._raw["dphiz"], dtype=np.float64),
            "vol": np.asarray(self._raw["vol"], dtype=np.float64),
            "C1": float(self._raw["C1"]),
            "D1": float(self._raw["D1"]),
            "nodes2coord": nodes2coord,
            "right_nodes": right_nodes,
            "elastic_kernel": elastic_kernel,
        }

        self.u_init = u0_ref[freedofs].copy()

    def get_data(self):
        return self.params, self.adjacency, self.u_init.copy()
