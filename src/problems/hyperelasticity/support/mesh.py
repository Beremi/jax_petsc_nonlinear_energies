from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from mpi4py import MPI

from src.core.petsc.dof_partition import _rank_of_dof_vec, petsc_ownership_range
from src.core.problem_data.hdf5 import load_problem_hdf5, mesh_data_path


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

    scalar tet: [n0, n1, n2, n3]
    dof tet:    [3*n0,3*n0+1,3*n0+2, ..., 3*n3+2]
    """
    elems2nodes = np.asarray(elems2nodes, dtype=np.int64)
    dof_offsets = np.arange(3, dtype=np.int64)
    return (3 * elems2nodes[:, :, None] + dof_offsets[None, None, :]).reshape(
        elems2nodes.shape[0], 12
    )


def dimensions_for_level(level: int) -> tuple[int, int, int]:
    if int(level) < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    return 80 * (2 ** (int(level) - 1)), 2 ** int(level), 2 ** int(level)


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
    return HyperElasticityGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
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
) -> np.ndarray:
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


def reordered_free_to_total_dofs(
    reord_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
) -> np.ndarray:
    reord = np.asarray(reord_dofs, dtype=np.int64)
    block = reord // 3
    comp = reord % 3
    nodes = _free_block_to_node_ids(block, grid, str(reorder_mode))
    return 3 * nodes + comp


def total_dofs_to_reordered_free(
    total_dofs: np.ndarray,
    grid: HyperElasticityGrid,
    reorder_mode: str,
) -> np.ndarray:
    total = np.asarray(total_dofs, dtype=np.int64)
    node = total // 3
    comp = total % 3
    ix, iy, iz = _node_ijk(node, grid)
    free = (ix > 0) & (ix < int(grid.nx))
    out = np.full(total.shape, -1, dtype=np.int64)
    if not np.any(free):
        return out
    x_inner = ix[free] - 1
    mode = str(reorder_mode)
    if mode == "none":
        block = (iz[free] * int(grid.ny1) + iy[free]) * (int(grid.nx) - 1) + x_inner
    elif mode == "block_xyz":
        block = (x_inner * int(grid.ny1) + iy[free]) * int(grid.nz1) + iz[free]
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
) -> np.ndarray:
    if owned_node_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    ix, iy, iz = _node_ijk(owned_node_ids, grid)
    cell_batches = []
    for dx in (-1, 0):
        cx = ix + dx
        valid_x = (cx >= 0) & (cx < int(grid.nx))
        for dy in (-1, 0):
            cy = iy + dy
            valid_xy = valid_x & (cy >= 0) & (cy < int(grid.ny))
            for dz in (-1, 0):
                cz = iz + dz
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
) -> np.ndarray:
    total = np.asarray(total_dofs, dtype=np.int64)
    node = total // 3
    comp = total % 3
    ix, _, _ = _node_ijk(node, grid)
    coords = _node_coordinates(node, grid)
    values = coords[np.arange(total.size), comp].copy()
    right = ix == int(grid.nx)
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
    )


def _owned_nullspace(
    owned_total_dofs: np.ndarray,
    grid: HyperElasticityGrid,
) -> np.ndarray:
    node = np.asarray(owned_total_dofs, dtype=np.int64) // 3
    comp = np.asarray(owned_total_dofs, dtype=np.int64) % 3
    coords = _node_coordinates(node, grid)
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


def load_rank_local_hyperelasticity(
    mesh_level: int,
    *,
    comm: MPI.Comm,
    reorder_mode: str = "block_xyz",
) -> tuple[dict[str, object], None, np.ndarray]:
    """Load only this rank's HyperElasticity overlap domain from HDF5."""
    mode = str(reorder_mode)
    if mode not in {"none", "block_xyz"}:
        raise ValueError(
            f"rank-local HyperElasticity supports element_reorder_mode='none' "
            f"or 'block_xyz', got {mode!r}"
        )

    filename = mesh_data_path("HyperElasticity", f"HyperElasticity_level{int(mesh_level)}.h5")
    grid = _read_grid_metadata(filename, int(mesh_level))
    n_free = int(grid.n_free_dofs)
    n_total = int(grid.n_total_dofs)
    dtype = _index_dtype(n_free)
    lo, hi = petsc_ownership_range(n_free, int(comm.rank), int(comm.size), block_size=3)
    owned_reord = np.arange(lo, hi, dtype=np.int64)
    owned_total_dofs = reordered_free_to_total_dofs(owned_reord, grid, mode)
    owned_node_ids = np.unique(owned_total_dofs // 3)
    local_elem_idx = _local_candidate_element_indices(owned_node_ids, grid)

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
    elems_reordered = total_dofs_to_reordered_free(elems_total, grid, mode)
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

    local_total_to_free = total_dofs_to_reordered_free(local_total_dofs, grid, mode)
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
    owned_block_nodes = _free_block_to_node_ids(owned_block_ids, grid, mode)
    owned_block_coordinates = _node_coordinates(owned_block_nodes, grid)
    params: dict[str, object] = {
        "freedofs": np.zeros(0, dtype=dtype),
        "C1": c1,
        "D1": d1,
        "_he_grid": grid,
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
            local_total_dofs, grid, 0.0
        ),
        "_distributed_u_init_owned": _values_for_total_dofs(owned_total_dofs, grid, 0.0),
        "_distributed_owned_block_coordinates": owned_block_coordinates,
        "_distributed_owned_nullspace": _owned_nullspace(owned_total_dofs, grid),
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
