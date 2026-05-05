"""Same-mesh PETSc PCMG helpers for 3D heterogeneous slope-stability."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.reordered_element_base import (
    build_near_nullspace,
    inverse_permutation,
)
from src.problems.slope_stability_3d.support.mesh import (
    base_mesh_name_for_name,
    build_same_mesh_lagrange_case_data,
    load_same_mesh_case_hdf5_light,
    load_same_mesh_case_hdf5_rank_local_light,
    mesh_name_with_uniform_refinements,
    ownership_block_size_3d,
    select_reordered_perm_3d,
    uniform_refinement_steps_for_name,
)
from src.problems.slope_stability_3d.support.simplex_lagrange import (
    evaluate_tetra_lagrange_basis,
    tetra_reference_nodes,
)


VECTOR_BLOCK_SIZE = 3


@dataclass(frozen=True)
class MGHierarchySpec:
    mesh_name: str
    degree: int


@dataclass(frozen=True)
class MGLevelSpace:
    mesh_name: str
    degree: int
    params: dict[str, object]
    perm: np.ndarray
    iperm: np.ndarray
    lo: int
    hi: int
    n_free: int
    ownership_block_size: int
    total_to_free_orig: np.ndarray


@dataclass
class SlopeStability3DMGHierarchy:
    levels: list[MGLevelSpace]
    prolongations: list[PETSc.Mat]
    restrictions: list[PETSc.Mat]
    build_metadata: dict[str, object] | None = None
    level_nullspaces: list[PETSc.NullSpace | None] | None = None
    level_coordinates: list[np.ndarray | None] | None = None

    def cleanup(self) -> None:
        for mat in self.restrictions:
            mat.destroy()
        for mat in self.prolongations:
            mat.destroy()
        for ns in self.level_nullspaces or []:
            if ns is not None:
                ns.destroy()


@dataclass(frozen=True)
class LegacyPMGLevelSmootherConfig:
    ksp_type: str
    pc_type: str
    steps: int


def _petsc_event(name: str):
    try:
        return PETSc.Log.Event(str(name))
    except PETSc.Error:
        return nullcontext()


def _degree_from_params(params: dict[str, object]) -> int:
    degree = params.get("element_degree")
    if degree is not None:
        return int(degree)
    n_scalar = int(np.asarray(params["elems_scalar"]).shape[1])
    mapping = {4: 1, 10: 2, 35: 4}
    try:
        return int(mapping[n_scalar])
    except KeyError as exc:
        raise ValueError(f"Unsupported scalar tetra size {n_scalar!r}") from exc


def _level_layout_for_nullspace(space: MGLevelSpace):
    return SimpleNamespace(
        perm=np.asarray(space.perm, dtype=np.int64),
        lo=int(space.lo),
        hi=int(space.hi),
        n_free=int(space.n_free),
    )


def _build_level_nullspace(space: MGLevelSpace, comm: MPI.Comm) -> PETSc.NullSpace | None:
    if "elastic_kernel" not in space.params:
        return None
    return build_near_nullspace(
        _level_layout_for_nullspace(space),
        space.params,
        comm,
        kernel_key="elastic_kernel",
    )


def _build_level_coordinates(space: MGLevelSpace) -> np.ndarray | None:
    if int(space.ownership_block_size) != VECTOR_BLOCK_SIZE:
        return None
    freedofs = np.asarray(space.params["freedofs"], dtype=np.int64)
    nodes = np.asarray(space.params["nodes"], dtype=np.float64)
    owned_orig_free = np.asarray(space.perm[space.lo : space.hi], dtype=np.int64)
    if owned_orig_free.size == 0:
        return None
    owned_total_dofs = np.asarray(freedofs[owned_orig_free], dtype=np.int64)
    if owned_total_dofs.size % VECTOR_BLOCK_SIZE != 0:
        return None
    owned_total_dofs = owned_total_dofs.reshape((-1, VECTOR_BLOCK_SIZE))
    node_ids = owned_total_dofs[:, 0] // VECTOR_BLOCK_SIZE
    return np.asarray(nodes[node_ids], dtype=np.float64)


def _slim_level_runtime_payload(space: MGLevelSpace) -> None:
    object.__setattr__(space, "params", {})
    object.__setattr__(space, "perm", np.empty(0, dtype=np.int64))
    object.__setattr__(space, "iperm", np.empty(0, dtype=np.int64))
    object.__setattr__(space, "total_to_free_orig", np.empty(0, dtype=np.int64))


def _ensure_ksp_options_prefix(ksp: PETSc.KSP, *, prefix_tag: str) -> str:
    prefix = str(ksp.getOptionsPrefix() or "")
    if prefix:
        return prefix
    safe_tag = "".join(ch if ch.isalnum() else "_" for ch in str(prefix_tag))
    prefix = f"slope3d_{safe_tag}_{id(ksp)}_"
    ksp.setOptionsPrefix(prefix)
    return prefix


def _apply_hypre_system_amg_settings(
    ksp: PETSc.KSP,
    *,
    nodal_coarsen: int = 6,
    vec_interp_variant: int = 3,
    strong_threshold: float | None = None,
    coarsen_type: str | None = None,
    max_iter: int = 2,
    tol: float = 0.0,
    relax_type_all: str | None = "symmetric-SOR/Jacobi",
    coordinates: np.ndarray | None = None,
    prefix_tag: str = "mg_coarse",
) -> None:
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    if coordinates is not None:
        pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))
    prefix = _ensure_ksp_options_prefix(ksp, prefix_tag=prefix_tag)
    opts = PETSc.Options()
    if int(nodal_coarsen) >= 0:
        opts[f"{prefix}pc_hypre_boomeramg_nodal_coarsen"] = int(nodal_coarsen)
    if int(vec_interp_variant) >= 0:
        opts[f"{prefix}pc_hypre_boomeramg_vec_interp_variant"] = int(vec_interp_variant)
    if strong_threshold is not None:
        opts[f"{prefix}pc_hypre_boomeramg_strong_threshold"] = float(strong_threshold)
    if str(coarsen_type or ""):
        opts[f"{prefix}pc_hypre_boomeramg_coarsen_type"] = str(coarsen_type)
    if int(max_iter) >= 0:
        opts[f"{prefix}pc_hypre_boomeramg_max_iter"] = int(max_iter)
    if tol is not None:
        opts[f"{prefix}pc_hypre_boomeramg_tol"] = float(tol)
    if str(relax_type_all or ""):
        opts[f"{prefix}pc_hypre_boomeramg_relax_type_all"] = str(relax_type_all)
    ksp.setFromOptions()


def _configure_coarse_solver(
    coarse: PETSc.KSP,
    *,
    backend: str,
    ksp_type: str,
    pc_type: str,
    redundant_number: int = 0,
    factor_solver_type: str | None = None,
    hypre_nodal_coarsen: int,
    hypre_vec_interp_variant: int,
    hypre_strong_threshold: float | None,
    hypre_coarsen_type: str | None,
    hypre_max_iter: int,
    hypre_tol: float,
    hypre_relax_type_all: str | None,
    coordinates: np.ndarray | None = None,
) -> None:
    backend_name = str(backend or "hypre")
    coarse.setType(str(ksp_type))
    coarse.setTolerances(rtol=1.0e-10, max_it=200)
    coarse_pc = coarse.getPC()
    if backend_name in {"hypre", "lu", "jacobi"}:
        coarse_pc.setType(str(pc_type))
        if str(pc_type) == "lu" and str(factor_solver_type or ""):
            coarse_pc.setFactorSolverType(str(factor_solver_type))
        if str(pc_type) == "hypre":
            _apply_hypre_system_amg_settings(
                coarse,
                nodal_coarsen=int(hypre_nodal_coarsen),
                vec_interp_variant=int(hypre_vec_interp_variant),
                strong_threshold=hypre_strong_threshold,
                coarsen_type=hypre_coarsen_type,
                max_iter=int(hypre_max_iter),
                tol=float(hypre_tol),
                relax_type_all=hypre_relax_type_all,
                coordinates=coordinates,
                prefix_tag="mg_coarse",
            )
        return
    if backend_name in {"redundant_lu", "redundant"}:
        coarse_pc.setType("redundant")
        prefix = _ensure_ksp_options_prefix(coarse, prefix_tag="mg_coarse")
        opts = PETSc.Options()
        if int(redundant_number) > 0:
            opts[f"{prefix}pc_redundant_number"] = int(redundant_number)
        opts[f"{prefix}redundant_ksp_type"] = "preonly"
        opts[f"{prefix}redundant_pc_type"] = "lu"
        if str(factor_solver_type or ""):
            opts[f"{prefix}redundant_pc_factor_mat_solver_type"] = str(
                factor_solver_type
            )
        coarse.setFromOptions()
        return
    raise ValueError(f"Unsupported MG coarse backend {backend_name!r}")


def _mat_is_ready_for_metadata(mat: PETSc.Mat) -> bool:
    try:
        sizes = mat.getSizes()
    except PETSc.Error:
        return False
    if len(sizes) == 2 and isinstance(sizes[0], tuple):
        (m_local, n_local), (m_global, n_global) = sizes
        return (
            int(m_local) >= 0
            and int(n_local) >= 0
            and int(m_global) >= 0
            and int(n_global) >= 0
        )
    m, n = sizes
    return int(m) >= 0 and int(n) >= 0


def _build_level_space(
    *,
    mesh_name: str,
    params: dict[str, object],
    adjacency,
    reorder_mode: str,
    comm: MPI.Comm,
    perm_override: np.ndarray | None = None,
) -> MGLevelSpace:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    if perm_override is None:
        if "_distributed_perm" in params:
            perm = np.asarray(params["_distributed_perm"], dtype=np.int64)
        else:
            if adjacency is None and str(reorder_mode) not in {"none", "block_xyz"}:
                raise ValueError(
                    "rank-local MG level build currently supports only reorder modes "
                    "'none' and 'block_xyz' without a global adjacency"
                )
            perm = select_reordered_perm_3d(
                str(reorder_mode),
                adjacency=adjacency,
                coords_all=np.asarray(params["nodes"], dtype=np.float64),
                freedofs=freedofs,
                n_parts=int(comm.size),
            )
    else:
        perm = np.asarray(perm_override, dtype=np.int64)
    if "_distributed_iperm" in params:
        iperm = np.asarray(params["_distributed_iperm"], dtype=np.int64)
    else:
        iperm = inverse_permutation(np.asarray(perm, dtype=np.int64))
    ownership_block_size = int(
        params.get("_distributed_ownership_block_size", ownership_block_size_3d(freedofs))
    )
    if "_distributed_lo" in params and "_distributed_hi" in params:
        lo = int(params["_distributed_lo"])
        hi = int(params["_distributed_hi"])
    else:
        lo, hi = petsc_ownership_range(
            int(freedofs.size),
            int(comm.rank),
            int(comm.size),
            block_size=int(ownership_block_size),
        )
    if "u_0_len" in params:
        total_size = int(params["u_0_len"])
    else:
        total_size = int(len(np.asarray(params["u_0"], dtype=np.float64)))
    total_to_free_orig = np.full(
        int(total_size),
        -1,
        dtype=np.int64,
    )
    total_to_free_orig[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    return MGLevelSpace(
        mesh_name=str(mesh_name),
        degree=int(_degree_from_params(params)),
        params=dict(params),
        perm=np.asarray(perm, dtype=np.int64),
        iperm=np.asarray(iperm, dtype=np.int64),
        lo=int(lo),
        hi=int(hi),
        n_free=int(freedofs.size),
        ownership_block_size=int(ownership_block_size),
        total_to_free_orig=total_to_free_orig,
    )


def _load_level_from_spec(
    spec: MGHierarchySpec,
    reorder_mode: str,
    comm: MPI.Comm,
    *,
    build_mode: str = "replicated",
) -> MGLevelSpace:
    if str(build_mode) == "rank_local":
        params = load_same_mesh_case_hdf5_rank_local_light(
            str(spec.mesh_name),
            int(spec.degree),
            reorder_mode=str(reorder_mode),
            comm=comm,
        )
        adjacency = None
    else:
        case_data = build_same_mesh_lagrange_case_data(
            str(spec.mesh_name),
            degree=int(spec.degree),
            build_mode=str(build_mode),
            comm=comm,
        )
        params = dict(case_data.__dict__)
        adjacency = case_data.adjacency
    params["elem_type"] = f"P{int(spec.degree)}"
    params["element_degree"] = int(spec.degree)
    return _build_level_space(
        mesh_name=str(spec.mesh_name),
        params=params,
        adjacency=adjacency,
        reorder_mode=reorder_mode,
        comm=comm,
    )


def _coalesce_transfer_entries(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    if rows.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    order = np.lexsort((cols, rows))
    rows = np.asarray(rows[order], dtype=np.int64)
    cols = np.asarray(cols[order], dtype=np.int64)
    data = np.asarray(data[order], dtype=np.float64)
    group_start = np.empty(rows.size, dtype=bool)
    group_start[0] = True
    same_prev = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])
    group_start[1:] = ~same_prev
    start_idx = np.flatnonzero(group_start)
    if start_idx.size != rows.size:
        mins = np.minimum.reduceat(data, start_idx)
        maxs = np.maximum.reduceat(data, start_idx)
        if np.any(np.abs(maxs - mins) > float(tolerance)):
            bad = int(np.flatnonzero(np.abs(maxs - mins) > float(tolerance))[0])
            raise ValueError(
                "Inconsistent interpolation entry for "
                f"row {int(rows[start_idx[bad]])}, col {int(cols[start_idx[bad]])}: "
                f"min={float(mins[bad]):.16e}, max={float(maxs[bad]):.16e}"
            )
    return (
        np.asarray(rows[start_idx], dtype=np.int64),
        np.asarray(cols[start_idx], dtype=np.int64),
        np.asarray(data[start_idx], dtype=np.float64),
    )


def _expand_vector_component_entries(
    *,
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
    fine_nodes: np.ndarray,
    coarse_nodes: np.ndarray,
    weights: np.ndarray,
    build_mode: str,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fine_nodes = np.asarray(fine_nodes, dtype=np.int64).ravel()
    coarse_nodes = np.asarray(coarse_nodes, dtype=np.int64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    fine_map = np.asarray(fine.total_to_free_orig, dtype=np.int64)
    coarse_map = np.asarray(coarse.total_to_free_orig, dtype=np.int64)
    fine_iperm = np.asarray(fine.iperm, dtype=np.int64)
    coarse_iperm = np.asarray(coarse.iperm, dtype=np.int64)
    for comp in range(VECTOR_BLOCK_SIZE):
        fine_total = VECTOR_BLOCK_SIZE * fine_nodes + int(comp)
        coarse_total = VECTOR_BLOCK_SIZE * coarse_nodes + int(comp)
        fine_free_orig = np.asarray(fine_map[fine_total], dtype=np.int64)
        coarse_free_orig = np.asarray(coarse_map[coarse_total], dtype=np.int64)
        valid = (fine_free_orig >= 0) & (coarse_free_orig >= 0)
        if not np.any(valid):
            continue
        fine_row = np.asarray(fine_iperm[fine_free_orig[valid]], dtype=np.int64)
        coarse_col = np.asarray(coarse_iperm[coarse_free_orig[valid]], dtype=np.int64)
        vals = np.asarray(weights[valid], dtype=np.float64)
        if str(build_mode) == "owned_rows":
            owned = (fine_row >= int(fine.lo)) & (fine_row < int(fine.hi))
            if not np.any(owned):
                continue
            fine_row = fine_row[owned]
            coarse_col = coarse_col[owned]
            vals = vals[owned]
        rows_parts.append(fine_row)
        cols_parts.append(coarse_col)
        data_parts.append(vals)
    if not rows_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return _coalesce_transfer_entries(
        np.concatenate(rows_parts),
        np.concatenate(cols_parts),
        np.concatenate(data_parts),
        tolerance=float(tolerance),
    )


def _adjacent_same_mesh_prolongation_entries(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
    *,
    build_mode: str,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(coarse.mesh_name) != str(fine.mesh_name):
        raise ValueError("same-mesh p-transfer requires matching mesh_name values")

    coarse_elem = np.asarray(coarse.params["elems_scalar"], dtype=np.int64)
    fine_elem = np.asarray(fine.params["elems_scalar"], dtype=np.int64)
    if coarse_elem.shape[0] != fine_elem.shape[0]:
        raise ValueError("same-mesh p-transfer requires matching macro element counts")
    if "material_id" in coarse.params and "material_id" in fine.params:
        if not np.array_equal(
            np.asarray(coarse.params["material_id"], dtype=np.int64),
            np.asarray(fine.params["material_id"], dtype=np.int64),
        ):
            raise ValueError("same-mesh p-transfer requires matching material ordering")

    fine_ref = tetra_reference_nodes(int(fine.degree))
    coarse_hatp = np.asarray(
        evaluate_tetra_lagrange_basis(int(coarse.degree), fine_ref)[0],
        dtype=np.float64,
    )
    flat_fine_nodes = np.asarray(fine_elem.reshape(-1), dtype=np.int64)
    unique_fine_nodes, first_idx = np.unique(flat_fine_nodes, return_index=True)
    fine_elem_width = int(fine_elem.shape[1])
    source_elem = np.asarray(first_idx // fine_elem_width, dtype=np.int64)
    source_local = np.asarray(first_idx % fine_elem_width, dtype=np.int64)
    coarse_elem_nodes = np.asarray(coarse_elem[source_elem], dtype=np.int64)
    weights = np.asarray(coarse_hatp[:, source_local], dtype=np.float64)
    coarse_local_idx, node_pos = np.nonzero(np.abs(weights) > float(tolerance))
    if coarse_local_idx.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return _expand_vector_component_entries(
        coarse=coarse,
        fine=fine,
        fine_nodes=np.asarray(unique_fine_nodes[node_pos], dtype=np.int64),
        coarse_nodes=np.asarray(
            coarse_elem_nodes[node_pos, coarse_local_idx],
            dtype=np.int64,
        ),
        weights=np.asarray(weights[coarse_local_idx, node_pos], dtype=np.float64),
        build_mode=str(build_mode),
        tolerance=float(tolerance),
    )


def _reference_coordinates_in_parent_tet(
    point: np.ndarray,
    parent_vertices: np.ndarray,
) -> np.ndarray:
    parent_vertices = np.asarray(parent_vertices, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    base = np.asarray(parent_vertices[0], dtype=np.float64)
    jac = np.column_stack(
        (
            np.asarray(parent_vertices[1] - base, dtype=np.float64),
            np.asarray(parent_vertices[2] - base, dtype=np.float64),
            np.asarray(parent_vertices[3] - base, dtype=np.float64),
        )
    )
    rhs = np.asarray(point - base, dtype=np.float64)
    return np.linalg.solve(jac, rhs)


def _parent_mesh_prolongation_entries(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
    *,
    build_mode: str,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_mesh_name = str(fine.params.get("macro_parent_mesh_name", "") or "")
    if parent_mesh_name != str(coarse.mesh_name):
        raise ValueError(
            "cross-mesh parent transfer requires fine macro_parent_mesh_name "
            f"to match coarse mesh_name, got {parent_mesh_name!r} vs {coarse.mesh_name!r}"
        )

    coarse_elem = np.asarray(coarse.params["elems_scalar"], dtype=np.int64)
    coarse_nodes = np.asarray(coarse.params["nodes"], dtype=np.float64)
    fine_elem = np.asarray(fine.params["elems_scalar"], dtype=np.int64)
    fine_nodes = np.asarray(fine.params["nodes"], dtype=np.float64)
    fine_parent = np.asarray(fine.params["macro_parent"], dtype=np.int64).ravel()
    if fine_parent.shape[0] != fine_elem.shape[0]:
        raise ValueError("fine macro_parent length must match fine macro element count")

    flat_fine_nodes = np.asarray(fine_elem.reshape(-1), dtype=np.int64)
    flat_parent = np.repeat(fine_parent, fine_elem.shape[1]).astype(np.int64)
    unique_fine_nodes, first_idx = np.unique(flat_fine_nodes, return_index=True)
    parent_id = np.asarray(flat_parent[first_idx], dtype=np.int64)
    coarse_elem_nodes = np.asarray(coarse_elem[parent_id], dtype=np.int64)
    coarse_vertex_nodes = np.asarray(coarse_elem_nodes[:, :4], dtype=np.int64)
    parent_vertices = np.asarray(coarse_nodes[coarse_vertex_nodes], dtype=np.float64)
    base = np.asarray(parent_vertices[:, 0, :], dtype=np.float64)
    jac = np.stack(
        (
            np.asarray(parent_vertices[:, 1, :] - base, dtype=np.float64),
            np.asarray(parent_vertices[:, 2, :] - base, dtype=np.float64),
            np.asarray(parent_vertices[:, 3, :] - base, dtype=np.float64),
        ),
        axis=-1,
    )
    rhs = np.asarray(fine_nodes[unique_fine_nodes], dtype=np.float64) - base
    xi = np.linalg.solve(jac, rhs[..., None]).squeeze(-1)
    weights = np.asarray(
        evaluate_tetra_lagrange_basis(int(coarse.degree), xi.T)[0],
        dtype=np.float64,
    )
    coarse_local_idx, node_pos = np.nonzero(np.abs(weights) > float(tolerance))
    if coarse_local_idx.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return _expand_vector_component_entries(
        coarse=coarse,
        fine=fine,
        fine_nodes=np.asarray(unique_fine_nodes[node_pos], dtype=np.int64),
        coarse_nodes=np.asarray(
            coarse_elem_nodes[node_pos, coarse_local_idx],
            dtype=np.int64,
        ),
        weights=np.asarray(weights[coarse_local_idx, node_pos], dtype=np.float64),
        build_mode=str(build_mode),
        tolerance=float(tolerance),
    )


def _build_matrix_from_coo(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    *,
    row_lo: int,
    row_hi: int,
    n_rows: int,
    col_lo: int,
    col_hi: int,
    n_cols: int,
    comm: MPI.Comm,
) -> PETSc.Mat:
    owned_mask = (rows >= int(row_lo)) & (rows < int(row_hi))
    owned_rows = np.asarray(rows[owned_mask], dtype=np.int64)
    owned_cols = np.asarray(cols[owned_mask], dtype=np.int64)
    owned_vals = np.asarray(data[owned_mask], dtype=np.float64)
    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.MPIAIJ)
    mat.setSizes(((int(row_hi) - int(row_lo), int(n_rows)), (int(col_hi) - int(col_lo), int(n_cols))))
    mat.setPreallocationCOO(
        owned_rows.astype(PETSc.IntType),
        owned_cols.astype(PETSc.IntType),
    )
    mat.setBlockSize(1)
    mat.setValuesCOO(
        owned_vals.astype(PETSc.ScalarType),
        addv=PETSc.InsertMode.INSERT_VALUES,
    )
    return mat


def _build_free_reordered_prolongation(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
    comm: MPI.Comm,
    *,
    build_mode: str = "owned_rows",
) -> tuple[PETSc.Mat, PETSc.Mat, dict[str, float | int]]:
    build_mode = str(build_mode)
    if build_mode not in {"replicated", "root_bcast", "owned_rows"}:
        raise ValueError(f"Unsupported transfer build mode {build_mode!r}")

    t0 = time.perf_counter()
    def _build_entries(local_build_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if str(coarse.mesh_name) == str(fine.mesh_name):
            return _adjacent_same_mesh_prolongation_entries(
                coarse,
                fine,
                build_mode=local_build_mode,
            )
        return _parent_mesh_prolongation_entries(
            coarse,
            fine,
            build_mode=local_build_mode,
        )

    with _petsc_event("slope3d_mg:transfer_mapping"):
        if build_mode == "root_bcast" and int(comm.size) > 1:
            payload = _build_entries("replicated") if int(comm.rank) == 0 else None
            rows, cols, data = comm.bcast(payload, root=0)
        else:
            rows, cols, data = _build_entries(build_mode)
    mapping_time = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    with _petsc_event("slope3d_mg:transfer_matrix_build"):
        prolong = _build_matrix_from_coo(
            rows,
            cols,
            data,
            row_lo=fine.lo,
            row_hi=fine.hi,
            n_rows=fine.n_free,
            col_lo=coarse.lo,
            col_hi=coarse.hi,
            n_cols=coarse.n_free,
            comm=comm,
        )
        if build_mode == "owned_rows":
            restrict = prolong.copy()
            restrict.transpose()
        else:
            restrict = _build_matrix_from_coo(
                cols,
                rows,
                data,
                row_lo=coarse.lo,
                row_hi=coarse.hi,
                n_rows=coarse.n_free,
                col_lo=fine.lo,
                col_hi=fine.hi,
                n_cols=fine.n_free,
                comm=comm,
            )
    matrix_build_time = float(time.perf_counter() - t1)
    return prolong, restrict, {
        "mapping_time": float(mapping_time),
        "matrix_build_time": float(matrix_build_time),
    }


def mixed_hierarchy_specs(
    *,
    mesh_name: str,
    finest_degree: int,
    strategy: str,
) -> list[MGHierarchySpec]:
    finest_degree = int(finest_degree)
    if str(strategy) == "same_mesh_p2_p1":
        if finest_degree != 2:
            raise ValueError("same_mesh_p2_p1 requires finest degree 2")
        return [MGHierarchySpec(str(mesh_name), 1), MGHierarchySpec(str(mesh_name), 2)]
    if str(strategy) == "same_mesh_p4_p2_p1":
        if finest_degree != 4:
            raise ValueError("same_mesh_p4_p2_p1 requires finest degree 4")
        return [
            MGHierarchySpec(str(mesh_name), 1),
            MGHierarchySpec(str(mesh_name), 2),
            MGHierarchySpec(str(mesh_name), 4),
        ]
    if str(strategy) == "uniform_refined_p4_p2_p1_p1":
        if finest_degree != 4:
            raise ValueError("uniform_refined_p4_p2_p1_p1 requires finest degree 4")
        coarse_mesh_name = base_mesh_name_for_name(str(mesh_name))
        if coarse_mesh_name == str(mesh_name):
            raise ValueError(
                "uniform_refined_p4_p2_p1_p1 requires a refined mesh name such as hetero_ssr_L1_2"
            )
        return [
            MGHierarchySpec(str(coarse_mesh_name), 1),
            MGHierarchySpec(str(mesh_name), 1),
            MGHierarchySpec(str(mesh_name), 2),
            MGHierarchySpec(str(mesh_name), 4),
        ]
    if str(strategy) == "uniform_refined_p1_chain":
        if finest_degree != 1:
            raise ValueError("uniform_refined_p1_chain requires finest degree 1")
        coarse_mesh_name = base_mesh_name_for_name(str(mesh_name))
        refinement_steps = int(uniform_refinement_steps_for_name(str(mesh_name)))
        if coarse_mesh_name != str(mesh_name) and refinement_steps < 1:
            raise ValueError(
                "uniform_refined_p1_chain requires sequential refinement suffixes"
            )
        return [
            MGHierarchySpec(mesh_name_with_uniform_refinements(str(coarse_mesh_name), steps), 1)
            for steps in range(refinement_steps + 1)
        ]
    raise ValueError(f"Unsupported 3D MG strategy {strategy!r}")


def build_mixed_pmg_hierarchy(
    *,
    specs: list[MGHierarchySpec],
    finest_params: dict[str, object],
    finest_adjacency,
    finest_perm: np.ndarray,
    reorder_mode: str,
    comm: MPI.Comm,
    level_build_mode: str = "replicated",
    transfer_build_mode: str = "owned_rows",
    precompute_level_nullspaces: bool = False,
    precompute_level_coordinates: bool = False,
    slim_completed_levels: bool = False,
) -> SlopeStability3DMGHierarchy:
    if len(specs) < 2:
        raise ValueError("3D PMG hierarchy requires at least two spaces")

    levels: list[MGLevelSpace] = []
    level_records: list[dict[str, object]] = []
    level_nullspaces: list[PETSc.NullSpace | None] = []
    level_coordinates: list[np.ndarray | None] = []
    t_levels0 = time.perf_counter()
    prolongations: list[PETSc.Mat] = []
    restrictions: list[PETSc.Mat] = []
    transfer_records: list[dict[str, object]] = []
    t_transfers0 = 0.0
    for idx, spec in enumerate(specs):
        t_level0 = time.perf_counter()
        if idx == len(specs) - 1:
            level_space = _build_level_space(
                mesh_name=str(spec.mesh_name),
                params=finest_params,
                adjacency=finest_adjacency,
                reorder_mode=str(reorder_mode),
                comm=comm,
                perm_override=np.asarray(finest_perm, dtype=np.int64),
            )
        else:
            level_space = _load_level_from_spec(
                spec,
                reorder_mode,
                comm,
                build_mode=str(level_build_mode),
            )
        levels.append(level_space)
        if bool(precompute_level_nullspaces):
            level_nullspaces.append(_build_level_nullspace(level_space, comm))
        else:
            level_nullspaces.append(None)
        if bool(precompute_level_coordinates):
            level_coordinates.append(_build_level_coordinates(level_space))
        else:
            level_coordinates.append(None)
        level_records.append(
            {
                "mesh_name": str(spec.mesh_name),
                "degree": int(spec.degree),
                "build_time": float(time.perf_counter() - t_level0),
                "n_free": int(level_space.n_free),
            }
        )
        if len(levels) >= 2:
            if t_transfers0 == 0.0:
                t_transfers0 = time.perf_counter()
            coarse = levels[-2]
            fine = levels[-1]
            prolong, restrict, transfer_meta = _build_free_reordered_prolongation(
                coarse,
                fine,
                comm,
                build_mode=str(transfer_build_mode),
            )
            prolongations.append(prolong)
            restrictions.append(restrict)
            transfer_records.append(
                {
                    "coarse_degree": int(coarse.degree),
                    "fine_degree": int(fine.degree),
                    "mapping_time": float(transfer_meta["mapping_time"]),
                    "matrix_build_time": float(transfer_meta["matrix_build_time"]),
                }
            )
            if bool(slim_completed_levels):
                _slim_level_runtime_payload(coarse)
    level_build_time = float(time.perf_counter() - t_levels0)
    transfer_build_time = (
        float(time.perf_counter() - t_transfers0) if t_transfers0 > 0.0 else 0.0
    )
    return SlopeStability3DMGHierarchy(
        levels=levels,
        prolongations=prolongations,
        restrictions=restrictions,
        build_metadata={
            "level_build_time": float(level_build_time),
            "transfer_build_time": float(transfer_build_time),
            "transfer_backend": "coo_vectorized",
            "level_records": level_records,
            "transfer_records": transfer_records,
        },
        level_nullspaces=level_nullspaces,
        level_coordinates=level_coordinates,
    )


def configure_pmg(
    ksp: PETSc.KSP,
    hierarchy: SlopeStability3DMGHierarchy,
    *,
    smoother_steps: int = 3,
    smoother_pc_type: str = "jacobi",
    level_smoothers: dict[str, LegacyPMGLevelSmootherConfig] | None = None,
    coarse_backend: str = "hypre",
    coarse_ksp_type: str | None = None,
    coarse_pc_type: str | None = None,
    coarse_redundant_number: int = 0,
    coarse_factor_solver_type: str | None = None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
) -> None:
    coarse_coordinates = None
    if hierarchy.level_coordinates and hierarchy.level_coordinates[0] is not None:
        coarse_coordinates = hierarchy.level_coordinates[0]
    else:
        coarse_coordinates = _build_level_coordinates(hierarchy.levels[0])

    def _configure_level_ksp(level_ksp: PETSc.KSP, cfg: LegacyPMGLevelSmootherConfig) -> None:
        level_ksp.setType(str(cfg.ksp_type))
        level_ksp.setTolerances(max_it=int(cfg.steps))
        level_pc = level_ksp.getPC()
        level_pc.setType(str(cfg.pc_type))
        if str(cfg.pc_type) == "hypre":
            level_pc.setHYPREType("boomeramg")

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.MG)
    pc.setMGLevels(len(hierarchy.levels))
    pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
    pc.setMGCycleType(PETSc.PC.MGCycleType.V)
    for level_idx, (prolong, restrict) in enumerate(
        zip(hierarchy.prolongations, hierarchy.restrictions),
        start=1,
    ):
        pc.setMGInterpolation(level_idx, prolong)
        pc.setMGRestriction(level_idx, restrict)

    smoother_defaults = dict(level_smoothers or {})
    default_cfg = LegacyPMGLevelSmootherConfig(
        ksp_type="richardson",
        pc_type=str(smoother_pc_type),
        steps=int(smoother_steps),
    )
    p2_cfg = smoother_defaults.get("degree2", default_cfg)
    p1_cfg = smoother_defaults.get("degree1", default_cfg)
    fine_cfg = smoother_defaults.get("fine", default_cfg)
    finest_level_idx = len(hierarchy.levels) - 1
    for level_idx in range(1, len(hierarchy.levels)):
        level_space = hierarchy.levels[level_idx]
        cfg = fine_cfg if level_idx == finest_level_idx else (
            p2_cfg if int(level_space.degree) == 2 else p1_cfg
        )
        for level_ksp in (
            pc.getMGSmoother(level_idx),
            pc.getMGSmootherDown(level_idx),
            pc.getMGSmootherUp(level_idx),
        ):
            _configure_level_ksp(level_ksp, cfg)

    if coarse_ksp_type is None:
        coarse_ksp_type = "cg"
    if coarse_pc_type is None:
        coarse_pc_type = "hypre"
    _configure_coarse_solver(
        pc.getMGCoarseSolve(),
        backend=str(coarse_backend),
        ksp_type=str(coarse_ksp_type),
        pc_type=str(coarse_pc_type),
        redundant_number=int(coarse_redundant_number),
        factor_solver_type=coarse_factor_solver_type,
        hypre_nodal_coarsen=int(coarse_hypre_nodal_coarsen),
        hypre_vec_interp_variant=int(coarse_hypre_vec_interp_variant),
        hypre_strong_threshold=coarse_hypre_strong_threshold,
        hypre_coarsen_type=coarse_hypre_coarsen_type,
        hypre_max_iter=int(coarse_hypre_max_iter),
        hypre_tol=float(coarse_hypre_tol),
        hypre_relax_type_all=coarse_hypre_relax_type_all,
        coordinates=coarse_coordinates,
    )


def attach_pmg_level_metadata(
    ksp: PETSc.KSP,
    hierarchy: SlopeStability3DMGHierarchy,
    *,
    use_near_nullspace: bool = True,
    coarse_pc_type: str | None = None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
) -> dict[str, object]:
    pc = ksp.getPC()
    nullspaces: list[PETSc.NullSpace] = []
    level_records: list[dict[str, object]] = []

    def _iter_level_ksps(level_idx: int) -> list[tuple[str, PETSc.KSP]]:
        if level_idx == 0:
            return [("coarse", pc.getMGCoarseSolve())]
        return [
            ("smoother", pc.getMGSmoother(level_idx)),
            ("down", pc.getMGSmootherDown(level_idx)),
            ("up", pc.getMGSmootherUp(level_idx)),
        ]

    for level_idx, level_space in enumerate(hierarchy.levels):
        if use_near_nullspace:
            if hierarchy.level_nullspaces is None:
                hierarchy.level_nullspaces = [None] * len(hierarchy.levels)
            level_nullspace = hierarchy.level_nullspaces[level_idx]
            if level_nullspace is None:
                level_nullspace = _build_level_nullspace(level_space, ksp.comm)
                hierarchy.level_nullspaces[level_idx] = level_nullspace
        else:
            level_nullspace = None
        if hierarchy.level_coordinates is None:
            hierarchy.level_coordinates = [None] * len(hierarchy.levels)
        level_coordinates = hierarchy.level_coordinates[level_idx]
        if level_coordinates is None:
            level_coordinates = _build_level_coordinates(level_space)
            hierarchy.level_coordinates[level_idx] = level_coordinates
        record = {
            "level_index": int(level_idx),
            "mesh_name": str(level_space.mesh_name),
            "degree": int(level_space.degree),
            "ownership_block_size": int(level_space.ownership_block_size),
            "near_nullspace_requested": bool(use_near_nullspace),
            "near_nullspace_attached": bool(level_nullspace is not None),
            "coordinates_attached": bool(level_coordinates is not None),
            "matrix_block_sizes": [],
            "ksp_records": [],
        }
        for role, level_ksp in _iter_level_ksps(level_idx):
            try:
                amat, pmat = level_ksp.getOperators()
            except PETSc.Error:
                amat, pmat = None, None
            target_mats: list[PETSc.Mat] = []
            if amat is not None:
                target_mats.append(amat)
            if pmat is not None and (amat is None or pmat.handle != amat.handle):
                target_mats.append(pmat)
            for mat in target_mats:
                if not _mat_is_ready_for_metadata(mat):
                    continue
                if int(level_space.ownership_block_size) > 1:
                    mat.setBlockSize(int(level_space.ownership_block_size))
                record["matrix_block_sizes"].append(int(mat.getBlockSize()))
            if level_nullspace is not None:
                for mat in target_mats:
                    if not _mat_is_ready_for_metadata(mat):
                        continue
                    mat.setNearNullSpace(level_nullspace)
            level_pc = level_ksp.getPC()
            if (
                str(level_pc.getType()) == "hypre"
                or (level_idx == 0 and str(coarse_pc_type or level_pc.getType()) == "hypre")
            ):
                _apply_hypre_system_amg_settings(
                    level_ksp,
                    nodal_coarsen=int(coarse_hypre_nodal_coarsen),
                    vec_interp_variant=int(coarse_hypre_vec_interp_variant),
                    strong_threshold=coarse_hypre_strong_threshold,
                    coarsen_type=coarse_hypre_coarsen_type,
                    max_iter=int(coarse_hypre_max_iter),
                    tol=float(coarse_hypre_tol),
                    relax_type_all=coarse_hypre_relax_type_all,
                    coordinates=level_coordinates,
                    prefix_tag=f"mg_level_{level_idx}",
                )
            record["ksp_records"].append(
                {
                    "role": str(role),
                    "ksp_type": str(level_ksp.getType()),
                    "pc_type": str(level_pc.getType()),
                }
            )
        if level_nullspace is not None:
            nullspaces.append(level_nullspace)
        level_records.append(record)

    return {
        "nullspaces": nullspaces,
        "levels": level_records,
    }
