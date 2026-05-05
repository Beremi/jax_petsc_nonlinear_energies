"""Structured h-multigrid helpers for HyperElasticity P1 beam meshes."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.dof_partition import petsc_ownership_range
from src.problems.hyperelasticity.support.mesh import (
    HyperElasticityGrid,
    _free_block_to_node_ids,
    _node_coordinates,
    _node_ijk,
    dimensions_for_level,
    grid_for_level,
    total_dofs_to_reordered_free,
)


VECTOR_BLOCK_SIZE = 3

# Cube node order matches the checked-in mesh generator:
# n000, n100, n010, n110, n001, n101, n011, n111.
_CUBE_OFFSETS_UNIT = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

@dataclass(frozen=True)
class HELevelSpace:
    level: int
    grid: HyperElasticityGrid
    lo: int
    hi: int
    n_free: int
    reorder_mode: str


@dataclass
class HEPMGHierarchy:
    levels: list[HELevelSpace]
    prolongations: list[PETSc.Mat]
    restrictions: list[PETSc.Mat]
    build_metadata: dict[str, object]

    def cleanup(self) -> None:
        for mat in self.restrictions:
            mat.destroy()
        for mat in self.prolongations:
            mat.destroy()


@dataclass(frozen=True)
class HEPmgSmootherConfig:
    ksp_type: str
    pc_type: str
    steps: int


def _free_dofs_for_level(level: int) -> int:
    nx, ny, nz = dimensions_for_level(int(level))
    return int(3 * (nx - 1) * (ny + 1) * (nz + 1))


def choose_he_pmg_coarsest_level(
    *,
    finest_level: int,
    n_ranks: int,
    requested: str | int | None,
    min_dofs_per_rank: int = 128,
) -> int:
    """Resolve explicit or automatic HE PMG coarsest level.

    The automatic rule keeps the coarse grid from becoming almost empty per MPI
    rank: choose the lowest level whose global free DOFs divided by rank count
    exceeds ``min_dofs_per_rank``.  This gives L1 on small local jobs but avoids
    a 512-rank run solving a ~2k-DOF L1 problem with only a handful of rows per
    rank.
    """

    finest = int(finest_level)
    if finest <= 1:
        raise ValueError("HE PMG requires finest level >= 2")
    raw = "1" if requested is None else str(requested).strip().lower()
    if raw != "auto":
        coarsest = int(raw)
        if coarsest < 1 or coarsest >= finest:
            raise ValueError(
                f"HE PMG coarsest level must satisfy 1 <= coarsest < {finest}, "
                f"got {coarsest}"
            )
        return coarsest

    ranks = max(1, int(n_ranks))
    threshold = max(1, int(min_dofs_per_rank))
    for level in range(1, finest):
        if _free_dofs_for_level(level) / float(ranks) >= float(threshold):
            return int(level)
    return int(finest - 1)


def _level_space(level: int, comm: MPI.Comm, reorder_mode: str) -> HELevelSpace:
    mode = str(reorder_mode)
    if mode not in {"none", "block_xyz"}:
        raise ValueError(
            "HE PMG supports element_reorder_mode='none' or 'block_xyz'; "
            f"got {mode!r}"
        )
    grid = grid_for_level(int(level))
    n_free = int(grid.n_free_dofs)
    lo, hi = petsc_ownership_range(
        n_free,
        int(comm.rank),
        int(comm.size),
        block_size=VECTOR_BLOCK_SIZE,
    )
    return HELevelSpace(
        level=int(level),
        grid=grid,
        lo=int(lo),
        hi=int(hi),
        n_free=n_free,
        reorder_mode=mode,
    )


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


def _precompute_half_grid_trilinear_weights() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Interpolation weights for half-grid points in a structured brick.

    The checked-in HE levels are generated independently from the same brick
    grid recipe.  The fine tetrahedra are therefore not a nested subdivision of
    the coarse tetrahedra, even though the logical brick grids are nested.
    Trilinear nodal interpolation on the structured brick grid gives the
    geometrically consistent inter-level transfer for this hierarchy.
    """

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    corners = _CUBE_OFFSETS_UNIT.astype(np.int64)
    for ix2 in range(3):
        for iy2 in range(3):
            for iz2 in range(3):
                point = np.array([ix2, iy2, iz2], dtype=np.float64) / 2.0
                weights = (
                    np.where(corners[:, 0] == 1, point[0], 1.0 - point[0])
                    * np.where(corners[:, 1] == 1, point[1], 1.0 - point[1])
                    * np.where(corners[:, 2] == 1, point[2], 1.0 - point[2])
                )
                key = int(ix2 * 9 + iy2 * 3 + iz2)
                keep = np.abs(weights) > 1.0e-14
                out[key] = (
                    np.flatnonzero(keep).astype(np.int64, copy=False),
                    np.asarray(weights[keep], dtype=np.float64),
                )
    return out


_HALF_GRID_TRANSFER_WEIGHTS = _precompute_half_grid_trilinear_weights()


def _coalesce_entries(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    if rows.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    order = np.lexsort((cols, rows))
    rows = np.asarray(rows[order], dtype=np.int64)
    cols = np.asarray(cols[order], dtype=np.int64)
    data = np.asarray(data[order], dtype=np.float64)
    group_start = np.empty(rows.size, dtype=bool)
    group_start[0] = True
    group_start[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    starts = np.flatnonzero(group_start)
    return rows[starts], cols[starts], np.add.reduceat(data, starts)


def _adjacent_prolongation_entries(
    coarse: HELevelSpace,
    fine: HELevelSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if int(fine.level) != int(coarse.level) + 1:
        raise ValueError(
            "HE PMG currently builds adjacent uniform-refinement transfers only"
        )
    if str(coarse.reorder_mode) != str(fine.reorder_mode):
        raise ValueError("coarse/fine HE PMG reorder modes must match")

    owned_blocks = np.arange(fine.lo // VECTOR_BLOCK_SIZE, fine.hi // VECTOR_BLOCK_SIZE, dtype=np.int64)
    fine_nodes = _free_block_to_node_ids(owned_blocks, fine.grid, fine.reorder_mode)
    if fine_nodes.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i.copy(), empty_f, {"owned_fine_nodes": 0}

    ix, iy, iz = _node_ijk(fine_nodes, fine.grid)
    cx = np.minimum(ix // 2, int(coarse.grid.nx) - 1)
    cy = np.minimum(iy // 2, int(coarse.grid.ny) - 1)
    cz = np.minimum(iz // 2, int(coarse.grid.nz) - 1)
    lx2 = np.asarray(ix - 2 * cx, dtype=np.int64)
    ly2 = np.asarray(iy - 2 * cy, dtype=np.int64)
    lz2 = np.asarray(iz - 2 * cz, dtype=np.int64)
    keys = lx2 * 9 + ly2 * 3 + lz2
    coarse_base = (cz * int(coarse.grid.ny1) + cy) * int(coarse.grid.nx1) + cx
    coarse_offsets = _cube_offsets(coarse.grid)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for key in np.unique(keys):
        mask = keys == int(key)
        if not np.any(mask):
            continue
        brick_vertices, weights = _HALF_GRID_TRANSFER_WEIGHTS[int(key)]
        coarse_nodes = coarse_base[mask, None] + coarse_offsets[brick_vertices][None, :]
        fine_blocks = owned_blocks[mask]
        for comp in range(VECTOR_BLOCK_SIZE):
            fine_rows = 3 * fine_blocks + int(comp)
            coarse_total = 3 * coarse_nodes + int(comp)
            coarse_cols = total_dofs_to_reordered_free(
                coarse_total.reshape(-1),
                coarse.grid,
                coarse.reorder_mode,
            ).reshape(coarse_nodes.shape)
            valid = coarse_cols >= 0
            if not np.any(valid):
                continue
            repeated_rows = np.broadcast_to(fine_rows[:, None], coarse_nodes.shape)
            repeated_vals = np.broadcast_to(weights[None, :], coarse_nodes.shape)
            row_parts.append(np.asarray(repeated_rows[valid], dtype=np.int64))
            col_parts.append(np.asarray(coarse_cols[valid], dtype=np.int64))
            val_parts.append(np.asarray(repeated_vals[valid], dtype=np.float64))

    if not row_parts:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0, dtype=np.float64)
    else:
        rows, cols, vals = _coalesce_entries(
            np.concatenate(row_parts),
            np.concatenate(col_parts),
            np.concatenate(val_parts),
        )
    return rows, cols, vals, {
        "owned_fine_nodes": int(fine_nodes.size),
        "entries": int(rows.size),
    }


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
    owned = (rows >= int(row_lo)) & (rows < int(row_hi))
    owned_rows = np.asarray(rows[owned], dtype=np.int64)
    owned_cols = np.asarray(cols[owned], dtype=np.int64)
    owned_vals = np.asarray(data[owned], dtype=np.float64)
    mat = PETSc.Mat().create(comm=comm)
    mat.setType(PETSc.Mat.Type.MPIAIJ)
    mat.setSizes(
        ((int(row_hi) - int(row_lo), int(n_rows)), (int(col_hi) - int(col_lo), int(n_cols)))
    )
    mat.setPreallocationCOO(
        owned_rows.astype(PETSc.IntType, copy=False),
        owned_cols.astype(PETSc.IntType, copy=False),
    )
    if owned_vals.size:
        mat.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType, copy=False),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    return mat


def build_he_pmg_hierarchy(
    *,
    finest_level: int,
    coarsest_level: int,
    reorder_mode: str,
    comm: MPI.Comm,
) -> HEPMGHierarchy:
    t0 = time.perf_counter()
    levels = [
        _level_space(level, comm, reorder_mode)
        for level in range(int(coarsest_level), int(finest_level) + 1)
    ]
    transfer_records: list[dict[str, object]] = []
    prolongations: list[PETSc.Mat] = []
    restrictions: list[PETSc.Mat] = []
    for coarse, fine in zip(levels[:-1], levels[1:]):
        t_map = time.perf_counter()
        rows, cols, vals, meta = _adjacent_prolongation_entries(coarse, fine)
        mapping_time = time.perf_counter() - t_map
        t_mat = time.perf_counter()
        prolong = _build_matrix_from_coo(
            rows,
            cols,
            vals,
            row_lo=fine.lo,
            row_hi=fine.hi,
            n_rows=fine.n_free,
            col_lo=coarse.lo,
            col_hi=coarse.hi,
            n_cols=coarse.n_free,
            comm=comm,
        )
        restrict = prolong.copy()
        restrict.transpose()
        matrix_time = time.perf_counter() - t_mat
        prolongations.append(prolong)
        restrictions.append(restrict)
        transfer_records.append(
            {
                "coarse_level": int(coarse.level),
                "fine_level": int(fine.level),
                "mapping_time": float(mapping_time),
                "matrix_build_time": float(matrix_time),
                **meta,
            }
        )

    return HEPMGHierarchy(
        levels=levels,
        prolongations=prolongations,
        restrictions=restrictions,
        build_metadata={
            "kind": "structured_he_trilinear_pmg",
            "coarsest_level": int(coarsest_level),
            "finest_level": int(finest_level),
            "level_records": [
                {
                    "level": int(space.level),
                    "n_free": int(space.n_free),
                    "lo": int(space.lo),
                    "hi": int(space.hi),
                }
                for space in levels
            ],
            "transfer_records": transfer_records,
            "build_time": float(time.perf_counter() - t0),
        },
    )


def _ensure_ksp_prefix(ksp: PETSc.KSP, prefix: str) -> str:
    current = str(ksp.getOptionsPrefix() or "")
    if current:
        return current
    ksp.setOptionsPrefix(str(prefix))
    return str(prefix)


def _apply_hypre_settings(
    ksp: PETSc.KSP,
    *,
    coordinates: np.ndarray | None,
    nodal_coarsen: int,
    vec_interp_variant: int,
    strong_threshold: float | None,
    coarsen_type: str | None,
    max_iter: int,
    tol: float,
    relax_type_all: str | None,
    prefix: str,
) -> None:
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    if coordinates is not None:
        pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))
    opt_prefix = _ensure_ksp_prefix(ksp, prefix)
    opts = PETSc.Options()
    if int(nodal_coarsen) >= 0:
        opts[f"{opt_prefix}pc_hypre_boomeramg_nodal_coarsen"] = int(nodal_coarsen)
    if int(vec_interp_variant) >= 0:
        opts[f"{opt_prefix}pc_hypre_boomeramg_vec_interp_variant"] = int(vec_interp_variant)
    if strong_threshold is not None:
        opts[f"{opt_prefix}pc_hypre_boomeramg_strong_threshold"] = float(strong_threshold)
    if str(coarsen_type or ""):
        opts[f"{opt_prefix}pc_hypre_boomeramg_coarsen_type"] = str(coarsen_type)
    if int(max_iter) >= 0:
        opts[f"{opt_prefix}pc_hypre_boomeramg_max_iter"] = int(max_iter)
    if tol is not None:
        opts[f"{opt_prefix}pc_hypre_boomeramg_tol"] = float(tol)
    if str(relax_type_all or ""):
        opts[f"{opt_prefix}pc_hypre_boomeramg_relax_type_all"] = str(relax_type_all)


def _owned_level_coordinates(space: HELevelSpace) -> np.ndarray:
    owned_blocks = np.arange(
        int(space.lo) // VECTOR_BLOCK_SIZE,
        int(space.hi) // VECTOR_BLOCK_SIZE,
        dtype=np.int64,
    )
    node_ids = _free_block_to_node_ids(owned_blocks, space.grid, space.reorder_mode)
    return _node_coordinates(node_ids, space.grid)


def _configure_coarse_solver(
    coarse: PETSc.KSP,
    *,
    space: HELevelSpace,
    ksp_type: str | None,
    pc_type: str,
    redundant_number: int,
    telescope_reduction_factor: int,
    factor_solver_type: str | None,
    hypre_nodal_coarsen: int,
    hypre_vec_interp_variant: int,
    hypre_strong_threshold: float | None,
    hypre_coarsen_type: str | None,
    hypre_max_iter: int,
    hypre_tol: float,
    hypre_relax_type_all: str | None,
) -> None:
    pc_name = str(pc_type)
    if ksp_type is None:
        ksp_type = "preonly" if pc_name in {"lu", "redundant"} else "cg"
    coarse.setType(str(ksp_type))
    coarse.setTolerances(rtol=1.0e-10, max_it=200)
    coarse_pc = coarse.getPC()
    if pc_name == "hypre":
        _apply_hypre_settings(
            coarse,
            coordinates=_owned_level_coordinates(space),
            nodal_coarsen=int(hypre_nodal_coarsen),
            vec_interp_variant=int(hypre_vec_interp_variant),
            strong_threshold=hypre_strong_threshold,
            coarsen_type=hypre_coarsen_type,
            max_iter=int(hypre_max_iter),
            tol=float(hypre_tol),
            relax_type_all=hypre_relax_type_all,
            prefix="he_pmg_coarse_",
        )
    elif pc_name == "gamg":
        raise ValueError(
            "HE PMG coarse_pc_type='gamg' is disabled for now because PETSc "
            "needs coarse-level coordinates only after PCMG Galerkin operators "
            "exist; use hypre, lu, redundant, or telescope."
        )
    elif pc_name == "lu":
        coarse_pc.setType("lu")
        if str(factor_solver_type or ""):
            coarse_pc.setFactorSolverType(str(factor_solver_type))
    elif pc_name == "redundant":
        coarse_pc.setType("redundant")
        prefix = _ensure_ksp_prefix(coarse, "he_pmg_coarse_")
        opts = PETSc.Options()
        if int(redundant_number) > 0:
            opts[f"{prefix}pc_redundant_number"] = int(redundant_number)
        opts[f"{prefix}redundant_ksp_type"] = "preonly"
        opts[f"{prefix}redundant_pc_type"] = "lu"
        if str(factor_solver_type or ""):
            opts[f"{prefix}redundant_pc_factor_mat_solver_type"] = str(factor_solver_type)
    elif pc_name == "telescope":
        coarse_pc.setType("telescope")
        prefix = _ensure_ksp_prefix(coarse, "he_pmg_coarse_")
        opts = PETSc.Options()
        if int(telescope_reduction_factor) > 0:
            opts[f"{prefix}pc_telescope_reduction_factor"] = int(telescope_reduction_factor)
        opts[f"{prefix}telescope_ksp_type"] = "preonly"
        opts[f"{prefix}telescope_pc_type"] = "lu"
        if str(factor_solver_type or ""):
            opts[f"{prefix}telescope_pc_factor_mat_solver_type"] = str(factor_solver_type)
    else:
        coarse_pc.setType(pc_name)
    coarse.setFromOptions()


def configure_he_pmg(
    ksp: PETSc.KSP,
    hierarchy: HEPMGHierarchy,
    *,
    smoother: HEPmgSmootherConfig,
    coarse_ksp_type: str | None,
    coarse_pc_type: str,
    coarse_redundant_number: int,
    coarse_telescope_reduction_factor: int,
    coarse_factor_solver_type: str | None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
    galerkin: str = "both",
) -> None:
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

    opts = PETSc.Options()
    if str(galerkin or ""):
        opts["pc_mg_galerkin"] = str(galerkin)

    finest = len(hierarchy.levels) - 1
    for level_idx in range(1, len(hierarchy.levels)):
        for level_ksp in (
            pc.getMGSmoother(level_idx),
            pc.getMGSmootherDown(level_idx),
            pc.getMGSmootherUp(level_idx),
        ):
            level_ksp.setType(str(smoother.ksp_type))
            level_ksp.setTolerances(max_it=int(smoother.steps))
            level_pc = level_ksp.getPC()
            level_pc.setType(str(smoother.pc_type))
            if str(smoother.pc_type) == "hypre":
                level_pc.setHYPREType("boomeramg")
        if level_idx == finest:
            pc.getMGSmoother(level_idx).setTolerances(max_it=int(smoother.steps))

    _configure_coarse_solver(
        pc.getMGCoarseSolve(),
        space=hierarchy.levels[0],
        ksp_type=coarse_ksp_type,
        pc_type=str(coarse_pc_type),
        redundant_number=int(coarse_redundant_number),
        telescope_reduction_factor=int(coarse_telescope_reduction_factor),
        factor_solver_type=coarse_factor_solver_type,
        hypre_nodal_coarsen=int(coarse_hypre_nodal_coarsen),
        hypre_vec_interp_variant=int(coarse_hypre_vec_interp_variant),
        hypre_strong_threshold=coarse_hypre_strong_threshold,
        hypre_coarsen_type=coarse_hypre_coarsen_type,
        hypre_max_iter=int(coarse_hypre_max_iter),
        hypre_tol=float(coarse_hypre_tol),
        hypre_relax_type_all=coarse_hypre_relax_type_all,
    )
    ksp.setFromOptions()
