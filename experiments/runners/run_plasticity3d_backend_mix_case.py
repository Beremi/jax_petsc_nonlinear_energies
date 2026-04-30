#!/usr/bin/env python3
"""Run one Plasticity3D backend-mix comparison case."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.state_export import export_plasticity3d_state_npz
from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.minimizers import newton as local_newton
from src.core.petsc.reordered_element_base import inverse_permutation
from src.core.petsc.reasons import ksp_reason_name
from src.problems.slope_stability_3d.jax_petsc.multigrid import (
    LegacyPMGLevelSmootherConfig,
    SlopeStability3DMGHierarchy,
    _build_level_coordinates,
    _build_level_nullspace,
    attach_pmg_level_metadata,
    build_mixed_pmg_hierarchy,
    configure_pmg,
    mixed_hierarchy_specs,
)
from src.problems.slope_stability_3d.jax_petsc.reordered_element_assembler import (
    SlopeStability3DReorderedElementAssembler,
)
from src.problems.slope_stability_3d.jax_petsc.solver import (
    _apply_strength_reduction,
    _load_problem_data,
)
from src.problems.slope_stability_3d.support.mesh import (
    DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
    base_mesh_name_for_name,
    ensure_same_mesh_case_hdf5,
    load_case_hdf5_fields,
    normalize_constraint_variant,
    ownership_block_size_3d,
    raw_mesh_filename_for_name,
    same_mesh_case_hdf5_path,
    select_reordered_perm_3d,
    uniform_refinement_steps_for_name,
)

SOURCE_IMPORT_ERROR: Exception | None = None
try:
    from slope_stability.constitutive import ConstitutiveOperator
    from slope_stability.constitutive.problem import potential_2D as source_potential_2D
    from slope_stability.constitutive.problem import potential_3D as source_potential_3D
    from slope_stability.constitutive.reduction import reduction as source_reduction
    from slope_stability.export import write_history_json
    from slope_stability.fem.assembly import assemble_strain_geometry
    from slope_stability.fem import (
        assemble_strain_operator,
        prepare_owned_tangent_pattern,
        quadrature_volume_3d,
        vector_volume,
    )
    from slope_stability.fem.distributed_tangent import assemble_overlap_strain_from_values
    from slope_stability.linear import SolverFactory
    from slope_stability.mesh import (
        MaterialSpec,
        heterogenous_materials,
        load_mesh_from_file,
        reorder_mesh_nodes,
    )
    from slope_stability.nonlinear.newton import (
        _collector_delta,
        _collector_snapshot,
        _setup_linear_system,
        _solve_linear_system,
        newton as source_newton,
    )
    from slope_stability.problem_assets import load_material_rows_for_path
    from slope_stability.utils import (
        extract_submatrix_free,
        full_field_from_free_values,
        flatten_field,
        owned_block_range,
        q_to_free_indices,
        release_petsc_aij_matrix,
    )
except Exception as exc:  # pragma: no cover - exercised in real runs
    SOURCE_IMPORT_ERROR = exc

DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
LOCAL_SOLVER_FAST = "local"
LOCAL_SOLVER_PMG = "local_pmg"
LOCAL_SOLVER_PMG_SOURCEFIXED = "local_pmg_sourcefixed"
LOCAL_SOLVER_SOURCE_DFGMRES = "source_dfgmres"
_SOURCE_REFINED_MESH_FIELDS = (
    "nodes",
    "elems_scalar",
    "surf",
    "q_mask",
    "material_id",
)
_PMG_RANK_LOCAL_FIELDS = (
    "nodes",
    "elems_scalar",
    "freedofs",
    "material_id",
    "constraint_variant",
    "macro_parent",
    "macro_parent_mesh_name",
)


@dataclass(frozen=True)
class LocalPMGSettings:
    level_smoothers: dict[str, LegacyPMGLevelSmootherConfig]
    strategy: str
    level_build_mode: str
    transfer_build_mode: str
    use_near_nullspace: bool


@dataclass(frozen=True)
class LocalPMGLinearProfile:
    level_smoothers: dict[str, LegacyPMGLevelSmootherConfig]
    coarse_backend: str
    coarse_ksp_type: str
    coarse_pc_type: str
    coarse_hypre_nodal_coarsen: int
    coarse_hypre_vec_interp_variant: int
    coarse_hypre_strong_threshold: float | None
    coarse_hypre_coarsen_type: str | None
    coarse_hypre_max_iter: int
    coarse_hypre_tol: float
    coarse_hypre_relax_type_all: str | None


def _local_pmg_settings() -> LocalPMGSettings:
    cfg = LegacyPMGLevelSmootherConfig(
        ksp_type="chebyshev",
        pc_type="jacobi",
        steps=5,
    )
    return LocalPMGSettings(
        level_smoothers={
            "fine": cfg,
            "degree2": cfg,
            "degree1": cfg,
        },
        strategy="same_mesh_p4_p2_p1",
        level_build_mode="rank_local",
        transfer_build_mode="owned_rows",
        use_near_nullspace=True,
    )


def _local_pmg_settings_for_strategy(strategy: str) -> LocalPMGSettings:
    base = _local_pmg_settings()
    return LocalPMGSettings(
        level_smoothers=dict(base.level_smoothers),
        strategy=str(strategy),
        level_build_mode=str(base.level_build_mode),
        transfer_build_mode=str(base.transfer_build_mode),
        use_near_nullspace=bool(base.use_near_nullspace),
    )


def _is_local_pmg_solver_backend(solver_backend: str) -> bool:
    return str(solver_backend) in {LOCAL_SOLVER_PMG, LOCAL_SOLVER_PMG_SOURCEFIXED}


def _local_pmg_linear_profile(solver_backend: str) -> LocalPMGLinearProfile:
    name = str(solver_backend)
    if name == LOCAL_SOLVER_PMG_SOURCEFIXED:
        cfg = LegacyPMGLevelSmootherConfig(
            ksp_type="chebyshev",
            pc_type="jacobi",
            steps=3,
        )
        return LocalPMGLinearProfile(
            level_smoothers={
                "fine": cfg,
                "degree2": cfg,
                "degree1": cfg,
            },
            coarse_backend="hypre",
            coarse_ksp_type="preonly",
            coarse_pc_type="hypre",
            coarse_hypre_nodal_coarsen=-1,
            coarse_hypre_vec_interp_variant=-1,
            coarse_hypre_strong_threshold=0.5,
            coarse_hypre_coarsen_type="HMIS",
            coarse_hypre_max_iter=1,
            coarse_hypre_tol=0.0,
            coarse_hypre_relax_type_all=None,
        )
    cfg = LegacyPMGLevelSmootherConfig(
        ksp_type="chebyshev",
        pc_type="jacobi",
        steps=5,
    )
    return LocalPMGLinearProfile(
        level_smoothers={
            "fine": cfg,
            "degree2": cfg,
            "degree1": cfg,
        },
        coarse_backend="hypre",
        coarse_ksp_type="cg",
        coarse_pc_type="hypre",
        coarse_hypre_nodal_coarsen=6,
        coarse_hypre_vec_interp_variant=3,
        coarse_hypre_strong_threshold=0.5,
        coarse_hypre_coarsen_type="HMIS",
        coarse_hypre_max_iter=2,
        coarse_hypre_tol=0.0,
        coarse_hypre_relax_type_all="symmetric-SOR/Jacobi",
    )


def _require_source_imports() -> None:
    if SOURCE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The slope_stability source package is not importable. "
            "Launch this script with PYTHONPATH including <source-root>/src."
        ) from SOURCE_IMPORT_ERROR


def _local_hypre_options(prefix: str) -> None:
    opts = PETSc.Options()
    opts[f"{prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{prefix}pc_hypre_boomeramg_nodal_coarsen"] = 6
    opts[f"{prefix}pc_hypre_boomeramg_vec_interp_variant"] = 3
    opts[f"{prefix}pc_hypre_boomeramg_strong_threshold"] = 0.5
    opts[f"{prefix}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    opts[f"{prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{prefix}pc_hypre_boomeramg_tol"] = 0.0


def _append_stage_event(
    path: Path | None,
    *,
    stage: str,
    started: float,
    **fields: object,
) -> None:
    if path is None or PETSc.COMM_WORLD.getRank() != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "elapsed_s": float(time.perf_counter() - started),
        "wall_time_unix": float(time.time()),
    }
    payload.update(fields)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_jsonl_record(path: Path | None, payload: dict[str, object]) -> None:
    if path is None or PETSc.COMM_WORLD.getRank() != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _set_vec_from_global(vec: PETSc.Vec, global_arr: np.ndarray) -> None:
    ownership = vec.getOwnershipRange()
    flat = np.asarray(global_arr, dtype=np.float64).reshape(-1)
    vec.array[:] = flat[ownership[0] : ownership[1]]
    vec.assemble()


def _global_from_vec(vec: PETSc.Vec) -> np.ndarray:
    comm = vec.getComm().tompi4py()
    local = np.asarray(vec.array[:], dtype=np.float64).copy()
    parts = comm.allgather(local)
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def _global_dofs_for_nodes(nodes: np.ndarray, dim: int) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    return (
        int(dim) * np.repeat(nodes, int(dim))
        + np.tile(np.arange(int(dim), dtype=np.int64), nodes.size)
    ).astype(np.int64, copy=False)


def _downcast_int_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.int32:
        return arr
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        min_i32 = int(np.iinfo(np.int32).min)
        max_i32 = int(np.iinfo(np.int32).max)
        if int(np.min(arr)) >= min_i32 and int(np.max(arr)) <= max_i32:
            return np.asarray(arr, dtype=np.int32)
    return arr


def _copy_mat(A: PETSc.Mat) -> PETSc.Mat:
    B = A.copy()
    if int(A.getBlockSize() or 1) > 1:
        B.setBlockSize(int(A.getBlockSize() or 1))
    return B


def _ensure_mpiaij_if_serial(A: PETSc.Mat) -> PETSc.Mat:
    if int(A.getComm().getSize()) == 1 and str(A.getType()).lower() == "seqaij":
        A = A.convert("mpiaij")
    return A


def _destroy_mat(A: PETSc.Mat | None) -> None:
    if A is None:
        return
    release = getattr(sys.modules.get("slope_stability.utils", None), "release_petsc_aij_matrix", None)
    if callable(release):
        try:
            release(A)
        except Exception:
            pass
    A.destroy()


@dataclass
class LocalAssemblyBackend:
    assembler: SlopeStability3DReorderedElementAssembler
    params: dict[str, object]
    adjacency: object
    mesh_name: str

    def __post_init__(self) -> None:
        self.comm = self.assembler.comm
        self.layout = self.assembler.layout
        self.rhs_global = self._gather_owned(np.asarray(self.assembler._f_owned, dtype=np.float64))
        self.freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        self.perm = np.asarray(self.layout.perm, dtype=np.int64)
        self.force = np.asarray(self.params["force"], dtype=np.float64)
        self.coords_ref = np.asarray(self.params["nodes"], dtype=np.float64)
        self._elastic_mat: PETSc.Mat | None = None
        self._owned_tangent_mat: PETSc.Mat | None = None
        self._owned_regularized_mat: PETSc.Mat | None = None

    @property
    def n_free(self) -> int:
        return int(self.layout.n_free)

    @property
    def source_q(self) -> np.ndarray:
        return np.ones((1, self.n_free), dtype=bool)

    @property
    def source_f(self) -> np.ndarray:
        return np.asarray(self.rhs_global, dtype=np.float64).reshape((1, self.n_free), order="F")

    def _gather_owned(self, owned: np.ndarray) -> np.ndarray:
        parts = self.comm.allgather(np.asarray(owned, dtype=np.float64))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def create_vec(self, global_arr: np.ndarray | None = None) -> PETSc.Vec:
        return self.assembler.create_vec(global_arr)

    def global_from_vec(self, vec: PETSc.Vec) -> np.ndarray:
        return self._gather_owned(np.asarray(vec.array[:], dtype=np.float64))

    def vec_energy(self, vec: PETSc.Vec) -> float:
        return float(self.assembler.energy_fn(vec))

    def vec_gradient(self, vec: PETSc.Vec, g: PETSc.Vec) -> None:
        self.assembler.gradient_fn(vec, g)

    def vec_tangent(self, vec: PETSc.Vec) -> PETSc.Mat:
        local_owned = np.asarray(vec.array[:], dtype=np.float64)
        self.assembler.assemble_hessian_with_mode(local_owned, constitutive_mode="plastic")
        self._owned_tangent_mat = self.assembler.A
        return self.assembler.A

    def elastic_matrix(self) -> PETSc.Mat:
        if self._elastic_mat is not None:
            return self._elastic_mat
        zero = np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)
        self.assembler.assemble_hessian_with_mode(zero, constitutive_mode="elastic")
        self._elastic_mat = _copy_mat(self.assembler.A)
        return self._elastic_mat

    def source_u0(self) -> np.ndarray:
        return np.zeros((1, self.n_free), dtype=np.float64)

    def _u_global(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(U, dtype=np.float64).reshape(-1, order="F")

    def build_F_reduced(self, U: np.ndarray) -> np.ndarray:
        u_global = self._u_global(U)
        vec = self.create_vec(u_global)
        grad = vec.duplicate()
        try:
            self.assembler.gradient_fn(vec, grad)
            total_grad = self.global_from_vec(grad)
        finally:
            grad.destroy()
            vec.destroy()
        internal = total_grad + self.rhs_global
        return internal.reshape((1, self.n_free), order="F")

    def build_F_reduced_free(self, U: np.ndarray) -> np.ndarray:
        return np.asarray(self.build_F_reduced(U), dtype=np.float64).reshape(-1, order="F")

    def build_F_K_tangent_reduced(self, U: np.ndarray):
        F = self.build_F_reduced(U)
        _F_free, K_tangent = self.build_F_K_tangent_reduced_free(U)
        return F, K_tangent

    def build_F_K_tangent_reduced_free(self, U: np.ndarray):
        u_global = self._u_global(U)
        F_free = self.build_F_reduced_free(U)
        vec = self.create_vec(u_global)
        try:
            self.assembler.assemble_hessian_with_mode(
                np.asarray(vec.array[:], dtype=np.float64),
                constitutive_mode="plastic",
            )
            self._owned_tangent_mat = self.assembler.A
        finally:
            vec.destroy()
        return F_free, self._owned_tangent_mat

    def build_K_regularized(self, r: float):
        if self._owned_tangent_mat is None:
            raise RuntimeError("Tangent matrix is not available before build_K_regularized().")
        elastic = self.elastic_matrix()
        if self._owned_regularized_mat is None:
            self._owned_regularized_mat = _copy_mat(elastic)
        self._owned_regularized_mat.zeroEntries()
        self._owned_regularized_mat.axpy(
            float(1.0 - r),
            self._owned_tangent_mat,
            structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
        )
        self._owned_regularized_mat.axpy(
            float(r),
            elastic,
            structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
        )
        self._owned_regularized_mat.assemble()
        return self._owned_regularized_mat

    def build_F_K_regularized_reduced_free(self, U: np.ndarray, r: float):
        F_free, _ = self.build_F_K_tangent_reduced_free(U)
        return F_free, self.build_K_regularized(r)

    def build_F_K_regularized_reduced(self, U: np.ndarray, r: float):
        F = self.build_F_reduced(U)
        _F_free, K_r = self.build_F_K_regularized_reduced_free(U, r)
        return F, K_r

    def energy_global(self, u_global: np.ndarray) -> float:
        vec = self.create_vec(u_global)
        try:
            return float(self.assembler.energy_fn(vec))
        finally:
            vec.destroy()

    def gradient_global(self, u_global: np.ndarray) -> np.ndarray:
        vec = self.create_vec(u_global)
        grad = vec.duplicate()
        try:
            self.assembler.gradient_fn(vec, grad)
            return self.global_from_vec(grad)
        finally:
            grad.destroy()
            vec.destroy()

    def tangent_global(self, u_global: np.ndarray) -> PETSc.Mat:
        vec = self.create_vec(u_global)
        try:
            self.assembler.assemble_hessian_with_mode(
                np.asarray(vec.array[:], dtype=np.float64),
                constitutive_mode="plastic",
            )
            self._owned_tangent_mat = self.assembler.A
            return self.assembler.A
        finally:
            vec.destroy()

    def final_observables(self, u_global: np.ndarray) -> dict[str, float]:
        full_original = np.empty_like(np.asarray(u_global, dtype=np.float64))
        full_original[self.perm] = np.asarray(u_global, dtype=np.float64)
        u_full = np.asarray(self.params["u_0"], dtype=np.float64).copy()
        u_full[self.freedofs] = full_original
        coords_final = self.coords_ref + u_full.reshape((-1, 3))
        displacement = coords_final - self.coords_ref
        return {
            "energy": float(self.energy_global(np.asarray(u_global, dtype=np.float64))),
            "omega": float(np.dot(self.force, u_full)),
            "u_max": float(np.max(np.linalg.norm(displacement, axis=1))),
        }

    def close(self) -> None:
        _destroy_mat(self._elastic_mat)
        _destroy_mat(self._owned_regularized_mat)
        self.assembler.cleanup()


@dataclass
class LocalPMGSupport:
    hierarchy: SlopeStability3DMGHierarchy | None
    settings: LocalPMGSettings
    nullspaces_live: list[PETSc.NullSpace]
    realized_levels: int
    pc_backend: str

    def close(self) -> None:
        if self.hierarchy is not None:
            self.hierarchy.cleanup()


def _petsc_mat_to_owned_coo(mat: PETSc.Mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr, indices, data = mat.getValuesCSR()
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    row_lo, _row_hi = (int(v) for v in mat.getOwnershipRange())
    local_rows = np.diff(indptr)
    rows = np.repeat(np.arange(local_rows.size, dtype=np.int64) + int(row_lo), local_rows)
    return rows, indices, data


def _adapt_local_pmg_hierarchy_to_source(hierarchy: SlopeStability3DMGHierarchy):
    levels = []
    for level in hierarchy.levels:
        levels.append(
            SimpleNamespace(
                order=int(level.degree),
                dim=3,
                free_size=int(level.n_free),
                owned_row_range=(int(level.lo), int(level.hi)),
                lo=int(level.lo),
                hi=int(level.hi),
                coord=np.asarray(getattr(level, "source_coord", np.empty((3, 0))), dtype=np.float64),
                q_mask=np.asarray(getattr(level, "source_q_mask", np.empty((3, 0), dtype=bool)), dtype=bool),
            )
        )
    prolongations = []
    for mat in hierarchy.prolongations:
        rows, cols, data = _petsc_mat_to_owned_coo(mat)
        prolongations.append(
            SimpleNamespace(
                coo_rows=np.asarray(rows, dtype=np.int64),
                coo_cols=np.asarray(cols, dtype=np.int64),
                coo_data=np.asarray(data, dtype=np.float64),
                global_shape=tuple(int(v) for v in mat.getSize()),
                owned_row_range=tuple(int(v) for v in mat.getOwnershipRange()),
            )
        )
    return SimpleNamespace(
        levels=tuple(levels),
        prolongations=tuple(prolongations),
        coarse_level=levels[0] if levels else None,
    )


def _pmg_support_pc_type(pmg_support: LocalPMGSupport | None) -> str:
    if pmg_support is None:
        return "hypre"
    return "mg" if pmg_support.hierarchy is not None else "hypre"


@dataclass
class SourceAssemblyBackend:
    const_builder: object
    tangent_pattern: object
    coord: np.ndarray
    mesh_name: str
    q_mask_actual: np.ndarray
    free_idx_actual: np.ndarray
    rhs_free: np.ndarray
    elastic_full_mat: PETSc.Mat | None
    elastic_free_mat: PETSc.Mat | None
    data_dir: Path
    has_energy_operator: bool = False
    lambda_target: float = 1.5

    def __post_init__(self) -> None:
        self.comm = PETSc.COMM_WORLD.tompi4py()
        self.free_idx_actual = _downcast_int_array(self.free_idx_actual)
        self.n_free = int(self.free_idx_actual.size)
        self.source_q = np.ones((1, self.n_free), dtype=bool)
        self.source_f = np.asarray(self.rhs_free, dtype=np.float64).reshape((1, self.n_free), order="F")
        self.owned_tangent_pattern = self.tangent_pattern
        owned_free_rows = getattr(self.tangent_pattern, "owned_free_local_rows", None)
        if owned_free_rows is not None:
            local_free = int(np.asarray(owned_free_rows, dtype=np.int64).size)
        else:
            row0, row1 = owned_block_range(
                int(self.q_mask_actual.shape[1]),
                int(self.q_mask_actual.shape[0]),
                PETSc.COMM_WORLD,
            )
            owned_free = np.asarray(self.q_mask_actual, dtype=bool).reshape(-1, order="F")[row0:row1]
            local_free = int(np.count_nonzero(owned_free))
        rank = int(self.comm.rank) if hasattr(self.comm, "rank") else int(self.comm.Get_rank())
        lo = self.comm.exscan(local_free)
        if lo is None or rank == 0:
            lo = 0
        self._free_lo = int(lo)
        self._free_hi = int(lo) + int(local_free)
        self.rhs_free_local = np.asarray(
            np.asarray(self.rhs_free, dtype=np.float64)[self._free_lo : self._free_hi],
            dtype=np.float64,
        )
        if self.elastic_free_mat is not None:
            self.elastic_free_mat = _ensure_mpiaij_if_serial(self.elastic_free_mat)
        overlap_global_dofs = np.asarray(self.tangent_pattern.overlap_global_dofs, dtype=np.int64).reshape(-1)
        free_idx = np.asarray(self.free_idx_actual, dtype=np.int64).reshape(-1)
        overlap_pos = np.searchsorted(free_idx, overlap_global_dofs)
        valid = overlap_pos < free_idx.size
        if np.any(valid):
            valid_idx = np.flatnonzero(valid)
            valid[valid_idx] = free_idx[overlap_pos[valid_idx]] == overlap_global_dofs[valid_idx]
        self._overlap_free_positions = np.full(overlap_global_dofs.shape, -1, dtype=np.int32)
        if np.any(valid):
            self._overlap_free_positions[valid] = np.asarray(overlap_pos[valid], dtype=np.int32)
        self._owned_tangent_mat: PETSc.Mat | None = None
        self._owned_regularized_mat: PETSc.Mat | None = None

    def _to_full(self, values: np.ndarray) -> np.ndarray:
        return full_field_from_free_values(
            np.asarray(values, dtype=np.float64),
            self.free_idx_actual,
            tuple(int(v) for v in self.q_mask_actual.shape),
        )

    def _to_overlap(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        overlap = np.zeros(self._overlap_free_positions.shape[0], dtype=np.float64)
        valid = np.asarray(self._overlap_free_positions, dtype=np.int32) >= 0
        if np.any(valid):
            overlap[valid] = flat[np.asarray(self._overlap_free_positions[valid], dtype=np.int64)]
        return overlap

    def create_vec(self, global_arr: np.ndarray | None = None) -> PETSc.Vec:
        if global_arr is None:
            global_arr = np.zeros(self.n_free, dtype=np.float64)
        if self.elastic_free_mat is not None:
            vec = self.elastic_free_mat.createVecLeft()
        else:
            vec = PETSc.Vec().createMPI(
                size=(int(self._free_hi - self._free_lo), int(self.n_free)),
                comm=PETSc.COMM_WORLD,
            )
        _set_vec_from_global(vec, np.asarray(global_arr, dtype=np.float64))
        return vec

    def _unique_energy_global(self, u_global: np.ndarray) -> float:
        pattern = self.tangent_pattern
        owner_mask = np.asarray(
            getattr(pattern, "local_overlap_owner_mask", np.empty(0, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if owner_mask.size == 0 or not np.any(owner_mask):
            return 0.0
        u_overlap = self._to_overlap(np.asarray(u_global, dtype=np.float64).reshape(-1))
        strain = assemble_overlap_strain_from_values(
            pattern,
            u_overlap,
            use_compiled=bool(getattr(self.const_builder, "use_compiled_owned_constitutive", True)),
        )
        local_idx = np.asarray(pattern.local_int_indices, dtype=np.int64).reshape(-1)
        c_bar = getattr(self.const_builder, "_owned_overlap_c_bar", None)
        sin_phi = getattr(self.const_builder, "_owned_overlap_sin_phi", None)
        if c_bar is None or sin_phi is None:
            c_bar, sin_phi = source_reduction(
                np.asarray(self.const_builder.c0[local_idx], dtype=np.float64),
                np.asarray(self.const_builder.phi[local_idx], dtype=np.float64),
                np.asarray(self.const_builder.psi[local_idx], dtype=np.float64),
                float(self.lambda_target),
                str(self.const_builder.Davis_type),
            )
        shear = getattr(self.const_builder, "_owned_overlap_shear", None)
        bulk = getattr(self.const_builder, "_owned_overlap_bulk", None)
        lame = getattr(self.const_builder, "_owned_overlap_lame", None)
        if shear is None or bulk is None or lame is None:
            shear = np.asarray(self.const_builder.shear[local_idx], dtype=np.float64)
            bulk = np.asarray(self.const_builder.bulk[local_idx], dtype=np.float64)
            lame = np.asarray(self.const_builder.lame[local_idx], dtype=np.float64)
        if int(self.const_builder.dim) == 2:
            strain_energy = np.asarray(strain, dtype=np.float64).copy()
            # Source potential_2D expects tensor shear e12, while the overlap strain
            # assembly and constitutive routines use engineering gamma12.
            strain_energy[2, :] *= 0.5
            psi = source_potential_2D(strain_energy, c_bar, sin_phi, shear, bulk, lame)
        else:
            # Source constitutive_problem_3D consumes engineering shear ordered as
            # [g12, g23, g13], but potential_3D expects tensor shear ordered as
            # [e12, e13, e23]. Convert explicitly so the line-search energy matches
            # the stress/tangent convention.
            strain_energy = np.asarray(strain[[0, 1, 2, 3, 5, 4], :], dtype=np.float64).copy()
            strain_energy[3:, :] *= 0.5
            psi = source_potential_3D(strain_energy, c_bar, sin_phi, shear, bulk, lame)
        weight = np.asarray(pattern.overlap_assembly_weight, dtype=np.float64)
        local_energy = float(np.dot(weight[owner_mask], np.asarray(psi, dtype=np.float64)[owner_mask]))
        return float(self.comm.allreduce(local_energy, op=MPI.SUM))

    def _reduce_full_matrix(self, mat_full: PETSc.Mat) -> PETSc.Mat:
        free_arr = np.asarray(self.free_idx_actual, dtype=PETSc.IntType).reshape(-1)
        iset = PETSc.IS().createGeneral(free_arr, comm=mat_full.getComm())
        sub = mat_full.createSubMatrix(iset, iset)
        try:
            block_size = int(sub.getBlockSize() or 1)
            n_global, _m_global = sub.getSize()
            if block_size > 1 and int(n_global) % block_size != 0:
                sub.setBlockSize(1)
        except Exception:
            pass
        return _ensure_mpiaij_if_serial(sub)

    def _as_free_matrix(self, mat: PETSc.Mat) -> PETSc.Mat:
        n_global, m_global = mat.getSize()
        if int(n_global) == int(self.n_free) and int(m_global) == int(self.n_free):
            return _ensure_mpiaij_if_serial(mat)
        return self._reduce_full_matrix(mat)

    def global_from_vec(self, vec: PETSc.Vec) -> np.ndarray:
        return _global_from_vec(vec)

    def copy_vec_data(self, dst: PETSc.Vec, src: PETSc.Vec) -> bool:
        if tuple(int(v) for v in dst.getOwnershipRange()) != tuple(int(v) for v in src.getOwnershipRange()):
            return False
        dst.array[:] = np.asarray(src.array_r, dtype=np.float64)
        dst.assemble()
        return True

    def _clear_runtime_cache(self) -> None:
        clear_local = getattr(self.const_builder, "_clear_owned_local_cache", None)
        if callable(clear_local):
            clear_local()
        if hasattr(self.const_builder, "S"):
            self.const_builder.S = None
        if hasattr(self.const_builder, "DS"):
            self.const_builder.DS = None

    def vec_energy(self, vec: PETSc.Vec) -> float:
        return float(self.energy_global(self.global_from_vec(vec)))

    def vec_gradient(self, vec: PETSc.Vec, g: PETSc.Vec) -> None:
        u_free = self.global_from_vec(vec)
        build_local = getattr(self.const_builder, "build_F_reduced_free_local_from_overlap_u", None)
        if callable(build_local):
            grad_local = np.asarray(build_local(self._to_overlap(u_free)), dtype=np.float64).reshape(-1)
        else:
            full = self._to_full(u_free)
            grad_local = np.asarray(
                self.const_builder.build_F_reduced_free_local(full),
                dtype=np.float64,
            ).reshape(-1)
        g.array[:] = grad_local - np.asarray(self.rhs_free_local, dtype=np.float64)
        g.assemble()
        self._clear_runtime_cache()

    def vec_tangent(self, vec: PETSc.Vec) -> PETSc.Mat:
        return self.tangent_global(self.global_from_vec(vec))

    def elastic_matrix(self) -> PETSc.Mat:
        if self.elastic_free_mat is None:
            zero = np.zeros(self.n_free, dtype=np.float64)
            _unused_F_free, tangent = self.build_F_K_tangent_reduced_free(zero)
            self.elastic_free_mat = _copy_mat(tangent)
            self.elastic_free_mat = _ensure_mpiaij_if_serial(self.elastic_free_mat)
        return self.elastic_free_mat

    def source_u0(self) -> np.ndarray:
        return np.zeros((1, self.n_free), dtype=np.float64)

    def build_F_reduced(self, U: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free = np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)
        return F_free.reshape((1, self.n_free), order="F")

    def build_F_reduced_free(self, U: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        return np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)

    def build_F_K_tangent_reduced_free(self, U: np.ndarray):
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free, K_tangent_full = self.const_builder.build_F_K_tangent_reduced_free(full)
        _destroy_mat(self._owned_tangent_mat)
        self._owned_tangent_mat = self._as_free_matrix(K_tangent_full)
        return np.asarray(F_free, dtype=np.float64).reshape(-1), self._owned_tangent_mat

    def build_F_K_tangent_reduced(self, U: np.ndarray):
        F_free, K_tangent = self.build_F_K_tangent_reduced_free(U)
        return np.asarray(F_free, dtype=np.float64).reshape((1, self.n_free), order="F"), K_tangent

    def build_K_regularized(self, r: float):
        K_r_full = self.const_builder.build_K_regularized(float(r))
        _destroy_mat(self._owned_regularized_mat)
        self._owned_regularized_mat = self._as_free_matrix(K_r_full)
        return self._owned_regularized_mat

    def build_F_K_regularized_reduced_free(self, U: np.ndarray, r: float):
        full = self._to_full(np.asarray(U, dtype=np.float64).reshape(-1, order="F"))
        F_free, K_r_full = self.const_builder.build_F_K_regularized_reduced_free(full, float(r))
        _destroy_mat(self._owned_regularized_mat)
        self._owned_regularized_mat = self._as_free_matrix(K_r_full)
        return np.asarray(F_free, dtype=np.float64).reshape(-1), self._owned_regularized_mat

    def energy_global(self, u_global: np.ndarray) -> float:
        if not bool(self.has_energy_operator):
            return float("nan")
        potential = float(self._unique_energy_global(np.asarray(u_global, dtype=np.float64)))
        return potential - float(np.dot(self.rhs_free, np.asarray(u_global, dtype=np.float64)))

    def gradient_global(self, u_global: np.ndarray) -> np.ndarray:
        full = self._to_full(np.asarray(u_global, dtype=np.float64))
        F_free = np.asarray(self.const_builder.build_F_reduced_free(full), dtype=np.float64).reshape(-1)
        return F_free - self.rhs_free

    def tangent_global(self, u_global: np.ndarray) -> PETSc.Mat:
        build_local = getattr(self.const_builder, "build_K_tangent_reduced_free_from_overlap_u", None)
        if callable(build_local):
            mat = build_local(self._to_overlap(np.asarray(u_global, dtype=np.float64)))
        else:
            full = self._to_full(np.asarray(u_global, dtype=np.float64))
            mat = self.const_builder.build_K_tangent_reduced_free(full)
        mat = self._as_free_matrix(mat)
        if self._owned_tangent_mat is not None and self._owned_tangent_mat != mat:
            _destroy_mat(self._owned_tangent_mat)
        self._owned_tangent_mat = mat
        self._clear_runtime_cache()
        return mat

    def preallocate_tangent_matrix(self) -> None:
        preallocate = getattr(self.const_builder, "preallocate_owned_tangent_matrix", None)
        if callable(preallocate):
            preallocate()

    def final_observables(self, u_global: np.ndarray) -> dict[str, float]:
        full = self._to_full(np.asarray(u_global, dtype=np.float64))
        coords_ref = np.asarray(self.coord.T, dtype=np.float64)
        coords_final = coords_ref + np.asarray(full.T, dtype=np.float64)
        displacement = coords_final - coords_ref
        return {
            "energy": float(self.energy_global(np.asarray(u_global, dtype=np.float64))),
            "omega": float(np.dot(self.rhs_free, np.asarray(u_global, dtype=np.float64))),
            "u_max": float(np.max(np.linalg.norm(displacement, axis=1))),
        }

    def close(self) -> None:
        self.const_builder.release_petsc_caches()
        _destroy_mat(self.elastic_full_mat)
        _destroy_mat(self.elastic_free_mat)


def _local_problem_args(
    *,
    mesh_name: str = "hetero_ssr_L1",
    elem_degree: int = 4,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    local_hessian_mode: str = "element",
    autodiff_tangent_mode: str = "element",
    ksp_rtol: float = 1.0e-2,
    ksp_max_it: int = 100,
) -> SimpleNamespace:
    reuse_hessian_value_buffers = (
        str(os.environ.get("MIX_LOCAL_REUSE_HESSIAN_VALUE_BUFFERS", "1")).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    p4_hessian_chunk_size = int(
        str(os.environ.get("MIX_LOCAL_P4_HESSIAN_CHUNK_SIZE", "32")).strip()
    )
    p4_chunk_scatter_cache = str(
        os.environ.get("MIX_LOCAL_P4_CHUNK_SCATTER_CACHE", "auto")
    ).strip()
    p4_chunk_scatter_cache_max_gib = float(
        str(os.environ.get("MIX_LOCAL_P4_CHUNK_SCATTER_CACHE_MAX_GIB", "0.5")).strip()
    )
    assembly_backend = str(
        os.environ.get("MIX_LOCAL_ASSEMBLY_BACKEND", "coo")
    ).strip()
    return SimpleNamespace(
        mesh_name=str(mesh_name),
        elem_degree=int(elem_degree),
        constraint_variant=str(normalize_constraint_variant(constraint_variant)),
        problem_build_mode="rank_local",
        element_reorder_mode="block_xyz",
        distribution_strategy="overlap_p2p",
        local_hessian_mode=str(local_hessian_mode),
        autodiff_tangent_mode=str(autodiff_tangent_mode),
        reuse_hessian_value_buffers=bool(reuse_hessian_value_buffers),
        p4_hessian_chunk_size=int(p4_hessian_chunk_size),
        p4_chunk_scatter_cache=str(p4_chunk_scatter_cache),
        p4_chunk_scatter_cache_max_gib=float(p4_chunk_scatter_cache_max_gib),
        assembly_backend=str(assembly_backend),
        profile="performance",
        ksp_type="fgmres",
        pc_type="hypre",
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
        use_near_nullspace=False,
        jax_trace_dir="",
        enable_petsc_log_events=False,
    )


def _local_perm_override(*, params: dict[str, object], adjacency, comm: MPI.Comm) -> np.ndarray:
    if "_distributed_perm" in params:
        return np.asarray(params["_distributed_perm"], dtype=np.int64)
    return select_reordered_perm_3d(
        "block_xyz",
        adjacency=adjacency,
        coords_all=np.asarray(params["nodes"], dtype=np.float64),
        freedofs=np.asarray(params["freedofs"], dtype=np.int64),
        n_parts=int(comm.size),
    )


def _build_rank_local_layout_metadata(
    *,
    params: dict[str, object],
    reorder_mode: str,
    comm: MPI.Comm,
) -> dict[str, object]:
    nodes = np.asarray(params["nodes"], dtype=np.float64)
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    n_free = int(freedofs.size)
    index_dtype = np.int32 if int(n_free) <= int(np.iinfo(np.int32).max) else np.int64
    perm = select_reordered_perm_3d(
        str(reorder_mode),
        adjacency=None,
        coords_all=nodes,
        freedofs=freedofs,
        n_parts=int(comm.size),
    )
    iperm = inverse_permutation(np.asarray(perm, dtype=np.int64))
    ownership_block_size = int(ownership_block_size_3d(freedofs))
    lo, hi = petsc_ownership_range(
        n_free,
        int(comm.rank),
        int(comm.size),
        block_size=ownership_block_size,
    )
    return {
        "_distributed_perm": np.asarray(perm, dtype=index_dtype),
        "_distributed_iperm": np.asarray(iperm, dtype=index_dtype),
        "_distributed_lo": int(lo),
        "_distributed_hi": int(hi),
        "_distributed_ownership_block_size": int(ownership_block_size),
    }


def _load_local_problem_for_pmg(
    *,
    comm: MPI.Comm,
    mesh_name: str,
    elem_degree: int,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float,
    ksp_rtol: float,
    ksp_max_it: int,
    use_near_nullspace: bool = True,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> tuple[str, dict[str, object], object, np.ndarray]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    variant = normalize_constraint_variant(constraint_variant)
    path = ensure_same_mesh_case_hdf5(
        str(mesh_name),
        int(elem_degree),
        constraint_variant=variant,
    )
    params, _ = load_case_hdf5_fields(path, fields=_PMG_RANK_LOCAL_FIELDS, load_adjacency=False)
    params["u_0_len"] = int(3 * np.asarray(params["nodes"], dtype=np.float64).shape[0])
    params.update(_build_rank_local_layout_metadata(params=params, reorder_mode="block_xyz", comm=comm))
    mesh_name = str(mesh_name)
    adjacency = None
    _append_stage_event(stage_path, stage="local_pmg_problem_loaded", started=started)
    if not bool(use_near_nullspace):
        params.pop("elastic_kernel", None)
    perm_override = _local_perm_override(params=params, adjacency=adjacency, comm=comm)
    return str(mesh_name), params, adjacency, perm_override


def _build_local_assembly_backend(
    *,
    mesh_name: str = "hetero_ssr_L1",
    elem_degree: int = 4,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float = 1.5,
    local_hessian_mode: str = "element",
    autodiff_tangent_mode: str = "element",
    ksp_rtol: float = 1.0e-2,
    ksp_max_it: int = 100,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> LocalAssemblyBackend:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    comm = MPI.COMM_WORLD
    args = _local_problem_args(
        mesh_name=str(mesh_name),
        elem_degree=int(elem_degree),
        constraint_variant=str(constraint_variant),
        local_hessian_mode=str(local_hessian_mode),
        autodiff_tangent_mode=str(autodiff_tangent_mode),
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
    )
    mesh_name, params, adjacency = _load_problem_data(args, comm)
    _append_stage_event(stage_path, stage="local_problem_loaded", started=started)
    _apply_strength_reduction(params, float(lambda_target))
    perm_override = _local_perm_override(params=params, adjacency=adjacency, comm=comm)
    assembler = SlopeStability3DReorderedElementAssembler(
        params,
        comm,
        adjacency,
        ksp_rtol=float(ksp_rtol),
        ksp_type="fgmres",
        pc_type="hypre",
        ksp_max_it=int(ksp_max_it),
        use_near_nullspace=False,
        pc_options={},
        reorder_mode="block_xyz",
        local_hessian_mode=str(local_hessian_mode),
        autodiff_tangent_mode=str(autodiff_tangent_mode),
        perm_override=perm_override,
        block_size_override=ownership_block_size_3d(
            np.asarray(params["freedofs"], dtype=np.int64)
        ),
        distribution_strategy="overlap_p2p",
        reuse_hessian_value_buffers=True,
        assembly_backend="coo",
        petsc_log_events=False,
        jax_trace_dir="",
    )
    _append_stage_event(stage_path, stage="local_assembler_ready", started=started)
    return LocalAssemblyBackend(
        assembler=assembler,
        params=params,
        adjacency=adjacency,
        mesh_name=str(mesh_name),
    )


def _find_overlap_partition(
    elem: np.ndarray,
    owned_node_range: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    elem = np.asarray(elem, dtype=np.int64)
    node0, node1 = map(int, owned_node_range)
    touches_owned = np.any((elem >= node0) & (elem < node1), axis=0)
    overlap_elements = np.flatnonzero(touches_owned).astype(np.int64)
    if overlap_elements.size == 0:
        return np.empty(0, dtype=np.int64), overlap_elements
    overlap_nodes = np.unique(elem[:, overlap_elements].reshape(-1, order="F")).astype(np.int64)
    return overlap_nodes, overlap_elements


def _assemble_source_owned_rhs(
    *,
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray,
    material_identifier: np.ndarray,
    materials: list[object],
    owned_node_range: tuple[int, int],
    elem_type: str,
) -> np.ndarray:
    dim, n_nodes = np.asarray(coord, dtype=np.float64).shape
    node0, node1 = map(int, owned_node_range)
    row0 = int(dim) * node0
    row1 = int(dim) * node1
    overlap_nodes, overlap_elements = _find_overlap_partition(elem, (node0, node1))
    if overlap_elements.size == 0:
        return np.zeros(row1 - row0, dtype=np.float64)

    node_lids = np.full(n_nodes, -1, dtype=np.int64)
    node_lids[overlap_nodes] = np.arange(overlap_nodes.size, dtype=np.int64)
    coord_overlap = np.asarray(coord, dtype=np.float64)[:, overlap_nodes]
    elem_overlap = node_lids[np.asarray(elem, dtype=np.int64)[:, overlap_elements]]
    asm = assemble_strain_geometry(coord_overlap, elem_overlap, str(elem_type), dim=int(dim))
    _unused_c0, _unused_phi, _unused_psi, _unused_shear, _unused_bulk, _unused_lame, gamma = heterogenous_materials(
        np.asarray(material_identifier, dtype=np.int64)[overlap_elements],
        np.ones(int(asm.n_int), dtype=bool),
        int(asm.n_q),
        materials,
    )
    f_v_int = np.zeros((int(dim), int(asm.n_int)), dtype=np.float64)
    if int(dim) >= 2:
        f_v_int[1, :] = -np.asarray(gamma, dtype=np.float64)
    f_overlap = flatten_field(vector_volume(asm, f_v_int, np.asarray(asm.weight, dtype=np.float64)))
    owned_nodes = np.arange(node0, node1, dtype=np.int64)
    owned_local_nodes = node_lids[owned_nodes]
    if np.any(owned_local_nodes < 0):
        raise RuntimeError("Owned nodes must be present in the source overlap submesh")
    owned_local_dofs = _global_dofs_for_nodes(owned_local_nodes, int(dim))
    local_rhs = np.asarray(f_overlap[owned_local_dofs], dtype=np.float64).copy()
    owned_global_rows = np.arange(row0, row1, dtype=np.int64)
    owned_free = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")[owned_global_rows]
    local_rhs[~owned_free] = 0.0
    return local_rhs


def _owner_ranks_for_nodes_local(node_ids: np.ndarray, owned_ranges: list[tuple[int, int]]) -> np.ndarray:
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    starts = np.asarray([int(a) for a, _b in owned_ranges], dtype=np.int64)
    stops = np.asarray([int(b) for _a, b in owned_ranges], dtype=np.int64)
    pos = np.searchsorted(starts, node_ids, side="right") - 1
    pos = np.clip(pos, 0, len(owned_ranges) - 1)
    valid = (node_ids >= starts[pos]) & (node_ids < stops[pos])
    if not np.all(valid):
        bad = int(node_ids[np.flatnonzero(~valid)[0]])
        raise RuntimeError(f"Could not assign owner rank for node {bad}")
    return np.asarray(pos, dtype=np.int32)


def _slim_source_pattern_for_local_harness(pattern: object) -> None:
    shrink_to_empty = {
        "recv_global_ip",
        "send_global_ip",
        "overlap_nodes",
        "overlap_elements",
        "owned_local_overlap_dofs",
        "unique_nodes",
        "unique_elements",
        "unique_global_dofs",
        "unique_local_int_indices",
    }
    downcast_only = {
        "overlap_global_dofs",
        "local_int_indices",
        "owned_free_local_rows",
    }
    for field in shrink_to_empty | downcast_only:
        if not hasattr(pattern, field):
            continue
        value = getattr(pattern, field)
        if field in shrink_to_empty:
            object.__setattr__(pattern, field, np.empty(0, dtype=np.int32))
        else:
            object.__setattr__(pattern, field, _downcast_int_array(value))
    if hasattr(pattern, "stats"):
        object.__setattr__(pattern, "stats", {})
    if hasattr(pattern, "timings"):
        object.__setattr__(pattern, "timings", {})
    if hasattr(pattern, "overlap_B"):
        object.__setattr__(pattern, "overlap_B", None)


def _source_free_layout_metadata(
    *,
    source_root: Path,
    mesh_name: str,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
) -> tuple[int, int, int]:
    mesh_name = str(mesh_name)
    constraint_variant = normalize_constraint_variant(constraint_variant)
    base_mesh_name = str(base_mesh_name_for_name(mesh_name))
    mesh_path = source_root / "meshes" / "3d_hetero_ssr" / raw_mesh_filename_for_name(base_mesh_name)
    if int(uniform_refinement_steps_for_name(mesh_name)) > 0:
        refined_path = ensure_same_mesh_case_hdf5(
            mesh_name,
            4,
            constraint_variant=constraint_variant,
        )
        raw, _ = load_case_hdf5_fields(
            refined_path,
            fields=("nodes", "elems_scalar", "surf", "q_mask"),
            load_adjacency=False,
        )
        reordered = reorder_mesh_nodes(
            np.asarray(raw["nodes"], dtype=np.float64).T,
            np.asarray(raw["elems_scalar"], dtype=np.int64).T,
            np.asarray(raw["surf"], dtype=np.int64).T,
            np.asarray(raw["q_mask"], dtype=bool).T,
            strategy="block_xyz",
            n_parts=None,
        )
        q_mask = reordered.q_mask.astype(bool)
        n_nodes = int(reordered.coord.shape[1])
        del raw, reordered
    else:
        mesh = load_mesh_from_file(mesh_path, boundary_type=0, elem_type="P4")
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy="block_xyz",
            n_parts=None,
        )
        q_mask = reordered.q_mask.astype(bool)
        n_nodes = int(reordered.coord.shape[1])
        del mesh, reordered
    gc.collect()
    row0, row1 = owned_block_range(n_nodes, 3, PETSc.COMM_WORLD)
    local_free = int(np.count_nonzero(np.asarray(q_mask, dtype=bool).reshape(-1, order="F")[row0:row1]))
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = int(comm.rank) if hasattr(comm, "rank") else int(comm.Get_rank())
    lo = comm.exscan(local_free)
    if lo is None or rank == 0:
        lo = 0
    hi = int(lo) + int(local_free)
    return int(lo), int(hi), 1


def _build_source_assembly_backend(
    *,
    source_root: Path,
    mesh_name: str,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float,
    data_dir: Path,
    need_energy_operator: bool,
    build_elastic_operator: bool,
    compute_pattern_elastic_values: bool,
    preallocate_tangent_matrix: bool = True,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> SourceAssemblyBackend:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    _require_source_imports()
    mesh_name = str(mesh_name)
    constraint_variant = normalize_constraint_variant(constraint_variant)
    base_mesh_name = str(base_mesh_name_for_name(mesh_name))
    mesh_path = source_root / "meshes" / "3d_hetero_ssr" / raw_mesh_filename_for_name(base_mesh_name)
    material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        raise FileNotFoundError(f"Could not load material rows for {mesh_path}")
    materials = [
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in np.asarray(material_rows, dtype=np.float64)
    ]
    if int(uniform_refinement_steps_for_name(mesh_name)) > 0:
        refined_path = ensure_same_mesh_case_hdf5(
            mesh_name,
            4,
            constraint_variant=constraint_variant,
        )
        raw, _ = load_case_hdf5_fields(
            refined_path,
            fields=_SOURCE_REFINED_MESH_FIELDS,
            load_adjacency=False,
        )
        reordered = reorder_mesh_nodes(
            np.asarray(raw["nodes"], dtype=np.float64).T,
            np.asarray(raw["elems_scalar"], dtype=np.int64).T,
            np.asarray(raw["surf"], dtype=np.int64).T,
            np.asarray(raw["q_mask"], dtype=bool).T,
            strategy="block_xyz",
            n_parts=None,
        )
        coord = reordered.coord.astype(np.float64)
        elem = reordered.elem.astype(np.int64)
        q_mask = reordered.q_mask.astype(bool)
        material_identifier = np.asarray(raw["material_id"], dtype=np.int64).ravel()
        del raw
    else:
        mesh = load_mesh_from_file(mesh_path, boundary_type=0, elem_type="P4")
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy="block_xyz",
            n_parts=None,
        )
        coord = reordered.coord.astype(np.float64)
        elem = reordered.elem.astype(np.int64)
        q_mask = reordered.q_mask.astype(bool)
        material_identifier = mesh.material.astype(np.int64).ravel()
        del mesh
    del reordered
    gc.collect()
    _append_stage_event(stage_path, stage="source_mesh_ready", started=started)

    n_q = int(quadrature_volume_3d("P4")[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )
    _append_stage_event(stage_path, stage="source_constitutive_ready", started=started)
    row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)

    overlap_rhs = _assemble_source_owned_rhs(
        coord=coord,
        elem=elem,
        q_mask=q_mask,
        material_identifier=material_identifier,
        materials=materials,
        owned_node_range=(row0 // coord.shape[0], row1 // coord.shape[0]),
        elem_type="P4",
    )
    _append_stage_event(
        stage_path,
        stage="source_rhs_local_ready",
        started=started,
        rhs_size=int(np.asarray(overlap_rhs, dtype=np.float64).size),
    )
    rhs_full_parts = PETSc.COMM_WORLD.tompi4py().allgather(np.asarray(overlap_rhs, dtype=np.float64))
    rhs_full = np.concatenate(rhs_full_parts) if rhs_full_parts else np.empty(0, dtype=np.float64)
    _append_stage_event(
        stage_path,
        stage="source_rhs_global_ready",
        started=started,
        rhs_size=int(np.asarray(rhs_full, dtype=np.float64).size),
    )
    del overlap_rhs

    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        (row0 // coord.shape[0], row1 // coord.shape[0]),
        elem_type="P4",
        include_unique=False,
        include_legacy_scatter=False,
        include_overlap_B=False,
        elastic_rows=None,
        compute_elastic_values=bool(compute_pattern_elastic_values),
    )
    if bool(need_energy_operator):
        owned_ranges = PETSc.COMM_WORLD.tompi4py().allgather(
            (int(row0 // coord.shape[0]), int(row1 // coord.shape[0]))
        )
        elem_owner_nodes = np.min(np.asarray(elem, dtype=np.int64), axis=0)
        elem_owner_ranks = _owner_ranks_for_nodes_local(elem_owner_nodes, owned_ranges)
        overlap_elements = np.asarray(pattern.overlap_elements, dtype=np.int64).reshape(-1)
        owner_mask = np.repeat(
            np.asarray(
                elem_owner_ranks[overlap_elements] == int(PETSc.COMM_WORLD.getRank()),
                dtype=bool,
            ),
            int(pattern.n_q),
        )
        object.__setattr__(pattern, "local_overlap_owner_mask", np.ascontiguousarray(owner_mask, dtype=bool))
    _append_stage_event(stage_path, stage="source_tangent_pattern_ready", started=started)
    const_builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=True,
        tangent_kernel="rows",
        constitutive_mode="overlap",
        use_compiled_constitutive=True,
    )
    _append_stage_event(stage_path, stage="source_pattern_registered", started=started)
    const_builder.reduction(float(lambda_target))
    _append_stage_event(stage_path, stage="source_reduction_done", started=started)
    if not bool(need_energy_operator):
        slim_owned_materials = getattr(const_builder, "slim_owned_constitutive_materials", None)
        if callable(slim_owned_materials):
            slim_owned_materials(drop_global=True)
            _append_stage_event(stage_path, stage="source_owned_materials_slimmed", started=started)
    _slim_source_pattern_for_local_harness(pattern)
    free_idx = q_to_free_indices(q_mask)
    rhs_free = np.asarray(rhs_full[free_idx], dtype=np.float64)
    _append_stage_event(
        stage_path,
        stage="source_rhs_free_ready",
        started=started,
        rhs_free_size=int(rhs_free.size),
    )
    del rhs_full_parts, rhs_full
    elastic_full: PETSc.Mat | None = None
    elastic_free: PETSc.Mat | None = None
    if bool(build_elastic_operator):
        zero_full = np.zeros(tuple(int(v) for v in q_mask.shape), dtype=np.float64, order="F")
        _unused_F_free, elastic_owned_live = const_builder.build_F_K_tangent_reduced_free(zero_full)
        elastic_full = elastic_owned_live.duplicate(copy=True)
        free_is = PETSc.IS().createGeneral(
            np.asarray(free_idx, dtype=PETSc.IntType).reshape(-1),
            comm=elastic_full.getComm(),
        )
        elastic_free = elastic_full.createSubMatrix(free_is, free_is)
        try:
            block_size = int(elastic_free.getBlockSize() or 1)
            n_global, _m_global = elastic_free.getSize()
            if block_size > 1 and int(n_global) % block_size != 0:
                elastic_free.setBlockSize(1)
        except Exception:
            pass
        _append_stage_event(stage_path, stage="source_elastic_matrix_ready", started=started)
    del material_identifier, materials
    gc.collect()
    backend = SourceAssemblyBackend(
        const_builder=const_builder,
        tangent_pattern=pattern,
        coord=coord,
        mesh_name=str(mesh_name),
        q_mask_actual=q_mask,
        free_idx_actual=free_idx,
        rhs_free=rhs_free,
        elastic_full_mat=elastic_full,
        elastic_free_mat=elastic_free,
        data_dir=data_dir,
        has_energy_operator=bool(need_energy_operator),
        lambda_target=float(lambda_target),
    )
    if not bool(build_elastic_operator) and bool(preallocate_tangent_matrix):
        backend.preallocate_tangent_matrix()
        _append_stage_event(stage_path, stage="source_tangent_matrix_preallocated", started=started)
    return backend


def _build_local_pmg_support(
    *,
    backend: LocalAssemblyBackend | SourceAssemblyBackend | None,
    mesh_name: str,
    elem_degree: int | None,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float,
    pmg_strategy: str,
    ksp_rtol: float,
    ksp_max_it: int,
    use_near_nullspace: bool | None = None,
    distributed_layout: tuple[int, int, int] | None = None,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> LocalPMGSupport:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    constraint_variant = normalize_constraint_variant(constraint_variant)
    settings = _local_pmg_settings_for_strategy(str(pmg_strategy))
    if use_near_nullspace is not None:
        settings = LocalPMGSettings(
            level_smoothers=dict(settings.level_smoothers),
            strategy=str(settings.strategy),
            level_build_mode=str(settings.level_build_mode),
            transfer_build_mode=str(settings.transfer_build_mode),
            use_near_nullspace=bool(use_near_nullspace),
        )
    comm = PETSc.COMM_WORLD.tompi4py()
    resolved_elem_degree = 4 if elem_degree is None else int(elem_degree)
    if backend is not None:
        if isinstance(backend, LocalAssemblyBackend):
            mesh_name = str(backend.mesh_name)
            params = backend.params
            adjacency = backend.adjacency
            finest_perm = np.asarray(backend.layout.perm, dtype=np.int64)
            resolved_elem_degree = int(params.get("element_degree", resolved_elem_degree))
        else:
            mesh_name, params, adjacency, finest_perm = _load_local_problem_for_pmg(
                comm=comm,
                mesh_name=str(mesh_name),
                elem_degree=int(resolved_elem_degree),
                constraint_variant=str(constraint_variant),
                lambda_target=float(lambda_target),
                ksp_rtol=float(ksp_rtol),
                ksp_max_it=int(ksp_max_it),
                use_near_nullspace=bool(settings.use_near_nullspace),
                stage_path=stage_path,
                stage_started=started,
            )
            lo, hi = backend.elastic_matrix().getOwnershipRange()
            params = dict(params)
            params["_distributed_lo"] = int(lo)
            params["_distributed_hi"] = int(hi)
            block_size = int(backend.elastic_matrix().getBlockSize() or 1)
            params["_distributed_ownership_block_size"] = int(block_size)
            _append_stage_event(
                stage_path,
                stage="local_pmg_source_layout_aligned",
                started=started,
                ownership_lo=int(lo),
                ownership_hi=int(hi),
                ownership_block_size=int(block_size),
            )
    else:
        mesh_name, params, adjacency, finest_perm = _load_local_problem_for_pmg(
            comm=comm,
            mesh_name=str(mesh_name),
            elem_degree=int(resolved_elem_degree),
            constraint_variant=str(constraint_variant),
            lambda_target=float(lambda_target),
            ksp_rtol=float(ksp_rtol),
            ksp_max_it=int(ksp_max_it),
            use_near_nullspace=bool(settings.use_near_nullspace),
            stage_path=stage_path,
            stage_started=started,
        )
        if distributed_layout is not None:
            lo, hi, block_size = tuple(int(v) for v in distributed_layout)
            params = dict(params)
            params["_distributed_lo"] = int(lo)
            params["_distributed_hi"] = int(hi)
            params["_distributed_ownership_block_size"] = int(block_size)
            _append_stage_event(
                stage_path,
                stage="local_pmg_source_layout_aligned",
                started=started,
                ownership_lo=int(lo),
                ownership_hi=int(hi),
                ownership_block_size=int(block_size),
            )
    specs = mixed_hierarchy_specs(
        mesh_name=str(mesh_name),
        finest_degree=int(resolved_elem_degree),
        strategy=str(settings.strategy),
    )
    realized_levels = int(len(specs))
    if realized_levels == 1:
        _append_stage_event(
            stage_path,
            stage="local_pmg_single_level_ready",
            started=started,
            realized_levels=int(realized_levels),
            pc_backend="hypre",
        )
        return LocalPMGSupport(
            hierarchy=None,
            settings=settings,
            nullspaces_live=[],
            realized_levels=int(realized_levels),
            pc_backend="hypre",
        )
    _append_stage_event(stage_path, stage="local_pmg_hierarchy_build_start", started=started)
    hierarchy = build_mixed_pmg_hierarchy(
        specs=specs,
        finest_params=params,
        finest_adjacency=adjacency,
        finest_perm=np.asarray(finest_perm, dtype=np.int64),
        reorder_mode="block_xyz",
        comm=comm,
        level_build_mode=str(settings.level_build_mode),
        transfer_build_mode=str(settings.transfer_build_mode),
        precompute_level_nullspaces=bool(settings.use_near_nullspace),
        precompute_level_coordinates=True,
        slim_completed_levels=not bool(settings.use_near_nullspace),
    )
    if hierarchy.level_nullspaces is None:
        hierarchy.level_nullspaces = [None] * len(hierarchy.levels)
    if hierarchy.level_coordinates is None:
        hierarchy.level_coordinates = [None] * len(hierarchy.levels)
    for level_idx, level in enumerate(hierarchy.levels):
        if hierarchy.level_nullspaces[level_idx] is None and bool(settings.use_near_nullspace):
            hierarchy.level_nullspaces[level_idx] = _build_level_nullspace(level, comm)
        if hierarchy.level_coordinates[level_idx] is None:
            hierarchy.level_coordinates[level_idx] = _build_level_coordinates(level)
        level_nodes = np.asarray(level.params.get("nodes", np.empty((0, 3))), dtype=np.float64)
        level_freedofs = np.asarray(level.params.get("freedofs", np.empty(0, dtype=np.int64)), dtype=np.int64)
        if level_nodes.ndim == 2 and level_nodes.shape[1] == 3:
            level_coord_full = np.asarray(level_nodes.T, dtype=np.float64)
            level_q_mask = np.zeros((3, int(level_nodes.shape[0])), dtype=bool)
            if level_freedofs.size:
                level_q_mask.reshape(-1, order="F")[level_freedofs] = True
            object.__setattr__(level, "source_coord", level_coord_full)
            object.__setattr__(level, "source_q_mask", level_q_mask)
        # The PMG runtime only needs the cached coordinates/nullspaces after setup.
        # Dropping the full per-level parameter payload keeps the mixed source/local
        # path under control at high MPI counts.
        object.__setattr__(level, "params", {})
        object.__setattr__(level, "perm", np.empty(0, dtype=np.int64))
        object.__setattr__(level, "iperm", np.empty(0, dtype=np.int64))
        object.__setattr__(level, "total_to_free_orig", np.empty(0, dtype=np.int64))
    gc.collect()
    _append_stage_event(stage_path, stage="local_pmg_hierarchy_ready", started=started)
    return LocalPMGSupport(
        hierarchy=hierarchy,
        settings=settings,
        nullspaces_live=[],
        realized_levels=int(realized_levels),
        pc_backend="mg",
    )


def _make_local_ksp(
    *,
    prefix: str,
    comm: PETSc.Comm,
    solver_backend: str,
    ksp_rtol: float,
    ksp_max_it: int,
    pmg_support: LocalPMGSupport | None = None,
) -> PETSc.KSP:
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setType("fgmres")
    if _is_local_pmg_solver_backend(str(solver_backend)):
        if pmg_support is None:
            raise ValueError("PMG local solver requested without PMG support.")
        if pmg_support.hierarchy is None:
            _local_hypre_options(prefix)
            ksp.getPC().setType("hypre")
        else:
            profile = _local_pmg_linear_profile(str(solver_backend))
            PETSc.Options()[f"{prefix}pc_mg_galerkin"] = "both"
            ksp.getPC().setType("mg")
            configure_pmg(
                ksp,
                pmg_support.hierarchy,
                level_smoothers=profile.level_smoothers,
                coarse_backend=str(profile.coarse_backend),
                coarse_ksp_type=str(profile.coarse_ksp_type),
                coarse_pc_type=str(profile.coarse_pc_type),
                coarse_hypre_nodal_coarsen=int(profile.coarse_hypre_nodal_coarsen),
                coarse_hypre_vec_interp_variant=int(profile.coarse_hypre_vec_interp_variant),
                coarse_hypre_strong_threshold=profile.coarse_hypre_strong_threshold,
                coarse_hypre_coarsen_type=profile.coarse_hypre_coarsen_type,
                coarse_hypre_max_iter=int(profile.coarse_hypre_max_iter),
                coarse_hypre_tol=float(profile.coarse_hypre_tol),
                coarse_hypre_relax_type_all=profile.coarse_hypre_relax_type_all,
            )
    else:
        _local_hypre_options(prefix)
        ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
    ksp.setFromOptions()
    return ksp


def _attach_local_pmg_metadata(
    ksp: PETSc.KSP,
    pmg_support: LocalPMGSupport,
    *,
    solver_backend: str,
) -> None:
    if pmg_support.hierarchy is None:
        return
    profile = _local_pmg_linear_profile(str(solver_backend))
    meta = attach_pmg_level_metadata(
        ksp,
        pmg_support.hierarchy,
        use_near_nullspace=bool(pmg_support.settings.use_near_nullspace),
        coarse_pc_type=str(profile.coarse_pc_type),
        coarse_hypre_nodal_coarsen=int(profile.coarse_hypre_nodal_coarsen),
        coarse_hypre_vec_interp_variant=int(profile.coarse_hypre_vec_interp_variant),
        coarse_hypre_strong_threshold=profile.coarse_hypre_strong_threshold,
        coarse_hypre_coarsen_type=profile.coarse_hypre_coarsen_type,
        coarse_hypre_max_iter=int(profile.coarse_hypre_max_iter),
        coarse_hypre_tol=float(profile.coarse_hypre_tol),
        coarse_hypre_relax_type_all=profile.coarse_hypre_relax_type_all,
    )
    for ns in list(meta.get("nullspaces", [])):
        if ns is None:
            continue
        if not any(existing is ns for existing in pmg_support.nullspaces_live):
            pmg_support.nullspaces_live.append(ns)


def _local_initial_guess(
    backend,
    *,
    out_dir: Path,
    solver_backend: str,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    ksp_rtol: float,
    ksp_max_it: int,
    pmg_support: LocalPMGSupport | None = None,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> tuple[PETSc.Vec, dict[str, object]]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    _append_stage_event(stage_path, stage="local_initial_guess_start", started=started)
    local_zero_initial_guess = str(os.environ.get("MIX_LOCAL_INITIAL_GUESS", "")).strip().lower()
    source_initial_guess_mode = str(os.environ.get("MIX_SOURCE_INITIAL_GUESS", "")).strip().lower()
    if local_zero_initial_guess == "zero":
        sol = backend.create_vec()
        rhs_vec = (
            np.asarray(getattr(backend, "rhs_global", getattr(backend, "rhs_free", np.empty(0))), dtype=np.float64)
        )
        meta = {
            "enabled": False,
            "success": True,
            "strategy": "zero",
            "ksp_type": "none",
            "pc_type": "none",
            "ksp_iterations": 0,
            "ksp_reason": "SKIPPED_ZERO_INITIAL_GUESS",
            "ksp_reason_code": 0,
            "rhs_norm": float(np.linalg.norm(rhs_vec)),
            "residual_norm": float(np.linalg.norm(rhs_vec)),
            "solve_time": 0.0,
            "vector_norm": 0.0,
        }
        _append_stage_event(
            stage_path,
            stage="local_initial_guess_done",
            started=started,
            ksp_iterations=0,
            success=True,
            strategy="zero",
        )
        return sol, meta
    if (
        isinstance(backend, SourceAssemblyBackend)
        and _is_local_pmg_solver_backend(str(solver_backend))
        and source_initial_guess_mode == "zero"
    ):
        sol = backend.create_vec()
        meta = {
            "enabled": False,
            "success": True,
            "strategy": "zero",
            "ksp_type": "none",
            "pc_type": "none",
            "ksp_iterations": 0,
            "ksp_reason": "SKIPPED_ZERO_INITIAL_GUESS",
            "ksp_reason_code": 0,
            "rhs_norm": float(np.linalg.norm(np.asarray(backend.rhs_free, dtype=np.float64))),
            "residual_norm": float(np.linalg.norm(np.asarray(backend.rhs_free, dtype=np.float64))),
            "solve_time": 0.0,
            "vector_norm": 0.0,
        }
        _append_stage_event(
            stage_path,
            stage="local_initial_guess_done",
            started=started,
            ksp_iterations=0,
            success=True,
            strategy="zero",
        )
        return sol, meta
    if (
        isinstance(backend, SourceAssemblyBackend)
        and _is_local_pmg_solver_backend(str(solver_backend))
        and source_initial_guess_mode in {"local_elastic", "local_constitutivead_elastic"}
    ):
        mesh_name = str(base_mesh_name_for_name(str(backend.mesh_name)))
        _append_stage_event(
            stage_path,
            stage="local_initial_guess_local_backend_build_start",
            started=started,
            mode=str(source_initial_guess_mode),
            mesh_name=mesh_name,
        )
        temp_backend = _build_local_assembly_backend(
            mesh_name=mesh_name,
            elem_degree=4,
            constraint_variant=str(constraint_variant),
            lambda_target=float(backend.lambda_target),
            autodiff_tangent_mode="constitutive",
            ksp_rtol=float(ksp_rtol),
            ksp_max_it=int(ksp_max_it),
            stage_path=stage_path,
            stage_started=started,
        )
        temp_pmg_support = None
        temp_rhs = None
        temp_sol = None
        try:
            temp_pmg_support = _build_local_pmg_support(
                backend=temp_backend,
                mesh_name=mesh_name,
                elem_degree=4,
                constraint_variant=str(constraint_variant),
                lambda_target=float(backend.lambda_target),
                pmg_strategy="same_mesh_p4_p2_p1",
                ksp_rtol=float(ksp_rtol),
                ksp_max_it=int(ksp_max_it),
                stage_path=stage_path,
                stage_started=started,
            )
            ksp = _make_local_ksp(
                prefix="mix_init_",
                comm=temp_backend.elastic_matrix().getComm(),
                solver_backend=str(solver_backend),
                ksp_rtol=float(ksp_rtol),
                ksp_max_it=int(ksp_max_it),
                pmg_support=temp_pmg_support,
            )
            temp_rhs = temp_backend.create_vec(
                temp_backend.rhs_global if hasattr(temp_backend, "rhs_global") else temp_backend.rhs_free
            )
            temp_sol = temp_backend.create_vec()
            ksp.setOperators(temp_backend.elastic_matrix())
            ksp.setUp()
            _attach_local_pmg_metadata(
                ksp,
                temp_pmg_support,
                solver_backend=str(solver_backend),
            )
            t0 = time.perf_counter()
            ksp.solve(temp_rhs, temp_sol)
            elapsed = time.perf_counter() - t0
            sol = backend.create_vec()
            _set_vec_from_global(sol, temp_backend.global_from_vec(temp_sol))
            meta = {
                "enabled": True,
                "success": bool(int(ksp.getConvergedReason()) > 0),
                "strategy": "local_constitutivead_elastic",
                "ksp_type": "fgmres",
                "pc_type": _pmg_support_pc_type(temp_pmg_support),
                "ksp_iterations": int(ksp.getIterationNumber()),
                "ksp_reason": str(ksp_reason_name(int(ksp.getConvergedReason()))),
                "ksp_reason_code": int(ksp.getConvergedReason()),
                "rhs_norm": float(temp_rhs.norm(PETSc.NormType.NORM_2)),
                "residual_norm": float(ksp.getResidualNorm()),
                "solve_time": float(elapsed),
                "vector_norm": float(np.linalg.norm(temp_backend.global_from_vec(temp_sol))),
            }
            ksp.destroy()
        finally:
            if temp_rhs is not None:
                temp_rhs.destroy()
            if temp_sol is not None:
                temp_sol.destroy()
            if temp_pmg_support is not None:
                temp_pmg_support.close()
            temp_backend.close()
        _append_stage_event(
            stage_path,
            stage="local_initial_guess_done",
            started=started,
            ksp_iterations=int(meta["ksp_iterations"]),
            success=bool(meta["success"]),
            strategy=str(meta["strategy"]),
        )
        return sol, meta
    ksp = _make_local_ksp(
        prefix="mix_init_",
        comm=backend.elastic_matrix().getComm(),
        solver_backend=str(solver_backend),
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
        pmg_support=pmg_support,
    )
    rhs = backend.create_vec(backend.rhs_global if hasattr(backend, "rhs_global") else backend.rhs_free)
    sol = backend.create_vec()
    sol_right = None
    ksp.setOperators(backend.elastic_matrix())
    if _is_local_pmg_solver_backend(str(solver_backend)):
        ksp.setUp()
        _attach_local_pmg_metadata(
            ksp,
            pmg_support,
            solver_backend=str(solver_backend),
        )
    t0 = time.perf_counter()
    if isinstance(backend, SourceAssemblyBackend):
        sol_right = backend.elastic_matrix().createVecRight()
        ksp.solve(rhs, sol_right)
        if not backend.copy_vec_data(sol, sol_right):
            _set_vec_from_global(sol, backend.global_from_vec(sol_right))
    else:
        ksp.solve(rhs, sol)
    elapsed = time.perf_counter() - t0
    meta = {
        "enabled": True,
        "success": bool(int(ksp.getConvergedReason()) > 0),
        "ksp_type": "fgmres",
        "pc_type": (
            _pmg_support_pc_type(pmg_support)
            if _is_local_pmg_solver_backend(str(solver_backend))
            else "hypre"
        ),
        "ksp_iterations": int(ksp.getIterationNumber()),
        "ksp_reason": str(ksp_reason_name(int(ksp.getConvergedReason()))),
        "ksp_reason_code": int(ksp.getConvergedReason()),
        "rhs_norm": float(rhs.norm(PETSc.NormType.NORM_2)),
        "residual_norm": float(ksp.getResidualNorm()),
        "solve_time": float(elapsed),
        "vector_norm": float(np.linalg.norm(backend.global_from_vec(sol))),
    }
    rhs.destroy()
    if sol_right is not None:
        sol_right.destroy()
    ksp.destroy()
    _append_stage_event(
        stage_path,
        stage="local_initial_guess_done",
        started=started,
        ksp_iterations=int(meta["ksp_iterations"]),
        success=bool(meta["success"]),
    )
    return sol, meta


def _run_local_solver_backend(
    backend,
    *,
    out_dir: Path,
    state_out: Path | None,
    mesh_name: str,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float,
    solver_backend: str,
    convergence_mode: str,
    grad_stop_tol: float | None,
    stop_tol: float,
    maxit: int,
    ksp_rtol: float,
    ksp_max_it: int,
    line_search: str,
    linesearch_tol: float,
    armijo_alpha0: float,
    armijo_c1: float,
    armijo_shrink: float,
    armijo_max_ls: int,
    pmg_support: LocalPMGSupport | None = None,
    lazy_pmg_config: dict[str, object] | None = None,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> dict[str, object]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    constraint_variant = normalize_constraint_variant(constraint_variant)
    convergence_mode = str(convergence_mode)
    grad_stop_tol_value = (
        float(grad_stop_tol)
        if grad_stop_tol is not None and np.isfinite(float(grad_stop_tol))
        else None
    )
    progress_path = out_dir / "data" / "progress.jsonl"

    def _ensure_pmg_support() -> LocalPMGSupport:
        nonlocal pmg_support
        if pmg_support is not None:
            return pmg_support
        if not _is_local_pmg_solver_backend(str(solver_backend)) or lazy_pmg_config is None:
            raise ValueError("PMG support is not available for the requested local solver path")
        pmg_support = _build_local_pmg_support(
            backend=None,
            mesh_name=str(lazy_pmg_config["mesh_name"]),
            elem_degree=int(lazy_pmg_config["elem_degree"]),
            constraint_variant=str(lazy_pmg_config.get("constraint_variant", constraint_variant)),
            lambda_target=float(lazy_pmg_config["lambda_target"]),
            pmg_strategy=str(lazy_pmg_config["pmg_strategy"]),
            ksp_rtol=float(ksp_rtol),
            ksp_max_it=int(ksp_max_it),
            use_near_nullspace=bool(lazy_pmg_config.get("use_near_nullspace", True)),
            distributed_layout=tuple(int(v) for v in lazy_pmg_config["distributed_layout"]),
            stage_path=stage_path,
            stage_started=started,
        )
        return pmg_support

    if (
        _is_local_pmg_solver_backend(str(solver_backend))
        and pmg_support is None
        and not (
            isinstance(backend, SourceAssemblyBackend)
            and str(os.environ.get("MIX_SOURCE_INITIAL_GUESS", "")).strip().lower() == "zero"
        )
    ):
        _ensure_pmg_support()

    x, init_meta = _local_initial_guess(
        backend,
        out_dir=out_dir,
        solver_backend=str(solver_backend),
        constraint_variant=str(constraint_variant),
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
        pmg_support=pmg_support,
        stage_path=stage_path,
        stage_started=started,
    )
    linear_records: list[dict[str, object]] = []
    reuse_ksp = not _is_local_pmg_solver_backend(str(solver_backend))
    persistent_ksp = (
        _make_local_ksp(
            prefix="mix_newton_",
            comm=backend.elastic_matrix().getComm(),
            solver_backend=str(solver_backend),
            ksp_rtol=float(ksp_rtol),
            ksp_max_it=int(ksp_max_it),
            pmg_support=pmg_support,
        )
        if reuse_ksp
        else None
    )

    def hessian_solve_fn(vec, rhs, sol):
        iteration = int(len(linear_records) + 1)
        _append_stage_event(
            stage_path,
            stage="local_linear_iteration_start",
            started=started,
            newton_iteration=iteration,
        )
        if _is_local_pmg_solver_backend(str(solver_backend)) and pmg_support is None:
            _ensure_pmg_support()
        t0 = time.perf_counter()
        _append_stage_event(
            stage_path,
            stage="local_linear_tangent_start",
            started=started,
            newton_iteration=iteration,
        )
        A = backend.vec_tangent(vec)
        t_assemble = time.perf_counter() - t0
        _append_stage_event(
            stage_path,
            stage="local_linear_tangent_ready",
            started=started,
            newton_iteration=iteration,
            t_assemble=float(t_assemble),
        )
        ksp = persistent_ksp
        if ksp is None:
            ksp = _make_local_ksp(
                prefix="mix_newton_",
                comm=A.getComm(),
                solver_backend=str(solver_backend),
                ksp_rtol=float(ksp_rtol),
                ksp_max_it=int(ksp_max_it),
                pmg_support=pmg_support,
            )
        t1 = time.perf_counter()
        _append_stage_event(
            stage_path,
            stage="local_linear_setup_start",
            started=started,
            newton_iteration=iteration,
        )
        ksp.setOperators(A)
        ksp.setUp()
        if _is_local_pmg_solver_backend(str(solver_backend)):
            _attach_local_pmg_metadata(
                ksp,
                pmg_support,
                solver_backend=str(solver_backend),
            )
        t_setup = time.perf_counter() - t1
        _append_stage_event(
            stage_path,
            stage="local_linear_setup_done",
            started=started,
            newton_iteration=iteration,
            t_setup=float(t_setup),
        )
        t2 = time.perf_counter()
        _append_stage_event(
            stage_path,
            stage="local_linear_solve_start",
            started=started,
            newton_iteration=iteration,
        )
        sol_right = None
        if isinstance(backend, SourceAssemblyBackend):
            sol_right = A.createVecRight()
            ksp.solve(rhs, sol_right)
            if not backend.copy_vec_data(sol, sol_right):
                _set_vec_from_global(sol, backend.global_from_vec(sol_right))
        else:
            ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t2
        _append_stage_event(
            stage_path,
            stage="local_linear_solve_done",
            started=started,
            newton_iteration=iteration,
            t_solve=float(t_solve),
            ksp_iterations=int(ksp.getIterationNumber()),
            ksp_reason_code=int(ksp.getConvergedReason()),
        )
        if sol_right is not None:
            sol_right.destroy()
        rec = {
            "newton_iteration": iteration,
            "ksp_its": int(ksp.getIterationNumber()),
            "ksp_reason_code": int(ksp.getConvergedReason()),
            "ksp_reason_name": str(ksp_reason_name(int(ksp.getConvergedReason()))),
            "ksp_residual_norm": float(ksp.getResidualNorm()),
            "t_assemble": float(t_assemble),
            "t_setup": float(t_setup),
            "t_solve": float(t_solve),
        }
        if persistent_ksp is None:
            ksp.destroy()
        linear_records.append(rec)
        _append_stage_event(
            stage_path,
            stage="local_linear_iteration_done",
            started=started,
            newton_iteration=int(rec["newton_iteration"]),
            ksp_iterations=int(rec["ksp_its"]),
            ksp_reason=str(rec["ksp_reason_name"]),
        )
        return int(rec["ksp_its"])

    def _iteration_progress_callback(entry: dict[str, object], _history: list[dict[str, object]]) -> None:
        payload = dict(entry)
        payload["event"] = "newton_iteration"
        payload["elapsed_s"] = float(time.perf_counter() - started)
        _append_jsonl_record(progress_path, payload)

    solve_t0 = time.perf_counter()
    _append_stage_event(stage_path, stage="local_newton_start", started=started)
    def _energy_placeholder(_vec: PETSc.Vec) -> float:
        return 0.0

    energy_fn = _energy_placeholder
    try:
        probe_energy = float(backend.vec_energy(x))
    except Exception:
        probe_energy = float("nan")
    if np.isfinite(probe_energy):
        energy_fn = backend.vec_energy
    elif str(line_search) == "armijo":
        raise RuntimeError(
            "Armijo line search requires a finite energy callback, but the selected "
            "backend does not provide one."
        )

    if convergence_mode == "gradient_only":
        grad_target = (
            float(grad_stop_tol_value)
            if grad_stop_tol_value is not None and grad_stop_tol_value > 0.0
            else 1.0e-2
        )
        require_all_convergence = False
        energy_tol = 0.0
        step_tol_rel = 0.0
    else:
        grad_target = 1.0e100
        require_all_convergence = True
        energy_tol = 1.0e100
        step_tol_rel = float(stop_tol)

    result = local_newton(
        energy_fn=energy_fn,
        gradient_fn=backend.vec_gradient,
        hessian_solve_fn=hessian_solve_fn,
        x=x,
        tolf=float(energy_tol),
        tolg=float(grad_target),
        tolg_rel=0.0,
        line_search=str(line_search),
        linesearch_tol=float(linesearch_tol),
        armijo_alpha0=float(armijo_alpha0),
        armijo_c1=float(armijo_c1),
        armijo_shrink=float(armijo_shrink),
        armijo_max_ls=int(armijo_max_ls),
        armijo_gradient_fallback=bool(
            str(line_search) == "armijo" and isinstance(backend, SourceAssemblyBackend)
        ),
        maxit=int(maxit),
        tolx_rel=float(step_tol_rel),
        tolx_abs=0.0,
        require_all_convergence=bool(require_all_convergence),
        fail_on_nonfinite=True,
        verbose=False,
        comm=PETSc.COMM_WORLD.tompi4py(),
        save_history=True,
        iteration_callback=_iteration_progress_callback,
    )
    solve_time = time.perf_counter() - solve_t0
    g_final = x.duplicate()
    try:
        backend.vec_gradient(x, g_final)
        final_grad_norm = float(g_final.norm(PETSc.NormType.NORM_2))
    finally:
        g_final.destroy()
    final_global = backend.global_from_vec(x)
    final_metric = (
        float(final_grad_norm)
        if convergence_mode == "gradient_only"
        else (
            float(result["history"][-1]["step_rel"])
            if result.get("history")
            else float("nan")
        )
    )
    observables = backend.final_observables(final_global)
    _export_backend_state(
        backend,
        state_out=state_out,
        mesh_name=str(mesh_name),
        constraint_variant=str(constraint_variant),
        lambda_target=float(lambda_target),
        u_global=final_global,
        energy=float(observables.get("energy", float("nan"))),
    )
    message_text = str(result.get("message", "")).lower()
    status = (
        "completed"
        if ("converged" in message_text)
        else "failed"
    )
    x.destroy()
    if persistent_ksp is not None:
        persistent_ksp.destroy()
    _append_stage_event(
        stage_path,
        stage="local_newton_done",
        started=started,
        nit=int(result.get("nit", 0)),
        status=str(status),
        final_metric=float(final_metric),
    )
    assembly_callbacks = (
        dict(backend.assembler.callback_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    assembler_setup = (
        dict(backend.assembler.setup_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    assembler_memory = (
        dict(backend.assembler.memory_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    return {
        "status": str(status),
        "solver_success": bool(status == "completed"),
        "message": str(result.get("message", "")),
        "nit": int(result.get("nit", 0)),
        "solve_time": float(solve_time),
        "total_time": float(solve_time),
        "linear_iterations_total": int(sum(int(row["ksp_its"]) for row in linear_records)),
        "linear_history": list(linear_records),
        "history": list(result.get("history", [])),
        "final_metric": float(final_metric),
        "final_metric_name": (
            "grad_norm" if convergence_mode == "gradient_only" else "relative_correction"
        ),
        "final_grad_norm": float(final_grad_norm),
        "initial_guess": init_meta,
        "assembly_callbacks": assembly_callbacks,
        "assembler_setup": assembler_setup,
        "assembler_memory": assembler_memory,
        "state_out": "" if state_out is None else str(state_out),
        **observables,
    }


def _make_source_dfgmres_solver(
    *,
    pmg_support: LocalPMGSupport,
    ksp_rtol: float,
    ksp_max_it: int,
) -> object:
    _require_source_imports()
    if pmg_support.hierarchy is None:
        raise ValueError("source DFGMRES requires a multilevel PMG hierarchy")
    hierarchy = _adapt_local_pmg_hierarchy_to_source(pmg_support.hierarchy)
    solver = SolverFactory.create(
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        tolerance=float(ksp_rtol),
        max_iterations=int(ksp_max_it),
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=np.array([], dtype=bool),
        coord=None,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "pmg_hierarchy": hierarchy,
            "mpi_distribute_by_nodes": True,
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "recycle_preconditioner": True,
            "compiled_outer": False,
            "full_system_preconditioner": False,
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_pc_type": "jacobi",
            "mg_levels_ksp_max_it": 3,
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "hypre",
            "mg_coarse_pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_P_max": 4,
            "pc_hypre_boomeramg_grid_sweeps_all": 1,
            "pc_hypre_boomeramg_agg_nl": 0,
        },
    )
    enable_diag = getattr(solver, "enable_diagnostics", None)
    if callable(enable_diag):
        enable_diag(True)
    return solver


def _run_local_solver_backend_with_source_linear(
    backend,
    *,
    out_dir: Path,
    state_out: Path | None,
    mesh_name: str,
    constraint_variant: str = DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    lambda_target: float,
    convergence_mode: str,
    grad_stop_tol: float | None,
    stop_tol: float,
    maxit: int,
    ksp_rtol: float,
    ksp_max_it: int,
    line_search: str,
    linesearch_tol: float,
    armijo_alpha0: float,
    armijo_c1: float,
    armijo_shrink: float,
    armijo_max_ls: int,
    pmg_support: LocalPMGSupport,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> dict[str, object]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    constraint_variant = normalize_constraint_variant(constraint_variant)
    convergence_mode = str(convergence_mode)
    grad_stop_tol_value = (
        float(grad_stop_tol)
        if grad_stop_tol is not None and np.isfinite(float(grad_stop_tol))
        else None
    )
    progress_path = out_dir / "data" / "progress.jsonl"
    linear_solver = _make_source_dfgmres_solver(
        pmg_support=pmg_support,
        ksp_rtol=float(ksp_rtol),
        ksp_max_it=int(ksp_max_it),
    )
    rhs_global = np.asarray(
        getattr(backend, "rhs_global", getattr(backend, "rhs_free", np.empty(0))),
        dtype=np.float64,
    ).reshape(-1)
    _append_stage_event(stage_path, stage="local_initial_guess_start", started=started)
    init_before = _collector_snapshot(linear_solver)
    t_init0 = time.perf_counter()
    linear_solver.setup_preconditioner(backend.elastic_matrix())
    if getattr(linear_solver, "supports_a_orthogonalization", lambda: False)():
        linear_solver.A_orthogonalize(backend.elastic_matrix())
    x_global = np.asarray(linear_solver.solve(backend.elastic_matrix(), rhs_global), dtype=np.float64).reshape(-1)
    init_elapsed = time.perf_counter() - t_init0
    init_after = _collector_snapshot(linear_solver)
    init_delta = _collector_delta(init_before, init_after)
    init_reason = getattr(linear_solver, "get_last_solve_info", lambda: {})()
    if getattr(linear_solver, "supports_dynamic_deflation_basis", lambda: False)():
        linear_solver.expand_deflation_basis(np.asarray(x_global, dtype=np.float64))
    x = backend.create_vec(x_global)
    init_meta = {
        "enabled": True,
        "success": True,
        "strategy": "source_dfgmres_elastic",
        "ksp_type": "dfgmres",
        "pc_type": "pmg_shell",
        "ksp_iterations": int(init_delta["iterations"]),
        "ksp_reason": str(init_reason.get("converged_reason", "SOURCE_DFGMRES")),
        "ksp_reason_code": int(init_reason.get("converged_reason", 0) or 0),
        "rhs_norm": float(np.linalg.norm(rhs_global)),
        "residual_norm": float(init_reason.get("true_residual_final", np.nan)),
        "solve_time": float(init_elapsed),
        "vector_norm": float(np.linalg.norm(x_global)),
        "linear_preconditioner_time": float(init_delta["preconditioner_time"]),
        "linear_orthogonalization_time": float(init_delta["orthogonalization_time"]),
    }
    _append_stage_event(
        stage_path,
        stage="local_initial_guess_done",
        started=started,
        ksp_iterations=int(init_meta["ksp_iterations"]),
        success=True,
        strategy=str(init_meta["strategy"]),
    )

    linear_records: list[dict[str, object]] = []

    def hessian_solve_fn(vec, rhs, sol):
        iteration = int(len(linear_records) + 1)
        _append_stage_event(
            stage_path,
            stage="local_linear_iteration_start",
            started=started,
            newton_iteration=iteration,
        )
        t0 = time.perf_counter()
        _append_stage_event(
            stage_path,
            stage="local_linear_tangent_start",
            started=started,
            newton_iteration=iteration,
        )
        A = backend.vec_tangent(vec)
        t_assemble = time.perf_counter() - t0
        _append_stage_event(
            stage_path,
            stage="local_linear_tangent_ready",
            started=started,
            newton_iteration=iteration,
            t_assemble=float(t_assemble),
        )
        before = _collector_snapshot(linear_solver)
        rhs_global_iter = backend.global_from_vec(rhs)
        t1 = time.perf_counter()
        linear_solver.setup_preconditioner(A)
        if getattr(linear_solver, "supports_a_orthogonalization", lambda: False)():
            linear_solver.A_orthogonalize(A)
        t_setup = time.perf_counter() - t1
        _append_stage_event(
            stage_path,
            stage="local_linear_setup_done",
            started=started,
            newton_iteration=iteration,
            t_setup=float(t_setup),
        )
        t2 = time.perf_counter()
        sol_global = np.asarray(linear_solver.solve(A, rhs_global_iter), dtype=np.float64).reshape(-1)
        t_solve = time.perf_counter() - t2
        after = _collector_snapshot(linear_solver)
        delta = _collector_delta(before, after)
        solve_info = getattr(linear_solver, "get_last_solve_info", lambda: {})()
        _set_vec_from_global(sol, sol_global)
        if getattr(linear_solver, "supports_dynamic_deflation_basis", lambda: False)():
            linear_solver.expand_deflation_basis(sol_global)
        release_iter = getattr(linear_solver, "release_iteration_resources", None)
        if callable(release_iter):
            release_iter()
        rec = {
            "newton_iteration": iteration,
            "ksp_its": int(delta["iterations"]),
            "ksp_reason_code": int(solve_info.get("converged_reason", 0) or 0),
            "ksp_reason_name": str(
                ksp_reason_name(int(solve_info.get("converged_reason", 0) or 0))
            ),
            "ksp_residual_norm": float(solve_info.get("true_residual_final", np.nan)),
            "t_assemble": float(t_assemble),
            "t_setup": float(t_setup),
            "t_solve": float(t_solve),
            "linear_preconditioner_time": float(delta["preconditioner_time"]),
            "linear_orthogonalization_time": float(delta["orthogonalization_time"]),
            "deflation_basis_cols": int(solve_info.get("basis_cols", 0) or 0),
        }
        linear_records.append(rec)
        _append_stage_event(
            stage_path,
            stage="local_linear_iteration_done",
            started=started,
            newton_iteration=int(rec["newton_iteration"]),
            ksp_iterations=int(rec["ksp_its"]),
            ksp_reason=str(rec["ksp_reason_name"]),
        )
        return int(rec["ksp_its"])

    def _iteration_progress_callback(entry: dict[str, object], _history: list[dict[str, object]]) -> None:
        payload = dict(entry)
        payload["event"] = "newton_iteration"
        payload["elapsed_s"] = float(time.perf_counter() - started)
        _append_jsonl_record(progress_path, payload)

    solve_t0 = time.perf_counter()
    _append_stage_event(stage_path, stage="local_newton_start", started=started)
    if convergence_mode == "gradient_only":
        grad_target = (
            float(grad_stop_tol_value)
            if grad_stop_tol_value is not None and grad_stop_tol_value > 0.0
            else 1.0e-2
        )
        require_all_convergence = False
        energy_tol = 0.0
        step_tol_rel = 0.0
    else:
        grad_target = 1.0e100
        require_all_convergence = True
        energy_tol = 1.0e100
        step_tol_rel = float(stop_tol)

    result = local_newton(
        energy_fn=backend.vec_energy,
        gradient_fn=backend.vec_gradient,
        hessian_solve_fn=hessian_solve_fn,
        x=x,
        tolf=float(energy_tol),
        tolg=float(grad_target),
        tolg_rel=0.0,
        line_search=str(line_search),
        linesearch_tol=float(linesearch_tol),
        armijo_alpha0=float(armijo_alpha0),
        armijo_c1=float(armijo_c1),
        armijo_shrink=float(armijo_shrink),
        armijo_max_ls=int(armijo_max_ls),
        armijo_gradient_fallback=False,
        maxit=int(maxit),
        tolx_rel=float(step_tol_rel),
        tolx_abs=0.0,
        require_all_convergence=bool(require_all_convergence),
        fail_on_nonfinite=True,
        verbose=False,
        comm=PETSc.COMM_WORLD.tompi4py(),
        save_history=True,
        iteration_callback=_iteration_progress_callback,
    )
    solve_time = time.perf_counter() - solve_t0
    g_final = x.duplicate()
    try:
        backend.vec_gradient(x, g_final)
        final_grad_norm = float(g_final.norm(PETSc.NormType.NORM_2))
    finally:
        g_final.destroy()
    final_global = backend.global_from_vec(x)
    final_metric = (
        float(final_grad_norm)
        if convergence_mode == "gradient_only"
        else (
            float(result["history"][-1]["step_rel"])
            if result.get("history")
            else float("nan")
        )
    )
    observables = backend.final_observables(final_global)
    _export_backend_state(
        backend,
        state_out=state_out,
        mesh_name=str(mesh_name),
        constraint_variant=str(constraint_variant),
        lambda_target=float(lambda_target),
        u_global=final_global,
        energy=float(observables.get("energy", float("nan"))),
    )
    message_text = str(result.get("message", "")).lower()
    status = "completed" if ("converged" in message_text) else "failed"
    x.destroy()
    close_solver = getattr(linear_solver, "close", None)
    if callable(close_solver):
        close_solver()
    _append_stage_event(
        stage_path,
        stage="local_newton_done",
        started=started,
        nit=int(result.get("nit", 0)),
        status=str(status),
        final_metric=float(final_metric),
    )
    assembly_callbacks = (
        dict(backend.assembler.callback_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    assembler_setup = (
        dict(backend.assembler.setup_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    assembler_memory = (
        dict(backend.assembler.memory_summary())
        if hasattr(backend, "assembler")
        else {}
    )
    return {
        "status": str(status),
        "solver_success": bool(status == "completed"),
        "message": str(result.get("message", "")),
        "nit": int(result.get("nit", 0)),
        "solve_time": float(solve_time),
        "total_time": float(solve_time),
        "linear_iterations_total": int(sum(int(row["ksp_its"]) for row in linear_records)),
        "linear_history": list(linear_records),
        "history": list(result.get("history", [])),
        "final_metric": float(final_metric),
        "final_metric_name": (
            "grad_norm" if convergence_mode == "gradient_only" else "relative_correction"
        ),
        "final_grad_norm": float(final_grad_norm),
        "initial_guess": init_meta,
        "assembly_callbacks": assembly_callbacks,
        "assembler_setup": assembler_setup,
        "assembler_memory": assembler_memory,
        "state_out": "" if state_out is None else str(state_out),
        **observables,
    }


def _make_source_solver():
    _require_source_imports()
    tolerance = float(os.environ.get("MIX_SOURCE_LINEAR_RTOL", "1.0e-2"))
    max_iterations = int(os.environ.get("MIX_SOURCE_LINEAR_MAX_IT", "100"))
    return SolverFactory.create(
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=np.array([], dtype=bool),
        coord=None,
        preconditioner_options={
            "pc_backend": "hypre",
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_max_iter": 1,
            "mpi_distribute_by_nodes": True,
        },
    )


def _run_source_solver_backend(
    backend,
    *,
    out_dir: Path,
    stop_tol: float,
    maxit: int,
    stage_path: Path | None = None,
    stage_started: float | None = None,
) -> dict[str, object]:
    started = float(stage_started if stage_started is not None else time.perf_counter())
    linear_solver = _make_source_solver()
    snap0 = _collector_snapshot(linear_solver)
    _append_stage_event(stage_path, stage="source_initial_guess_start", started=started)
    _setup_linear_system(linear_solver, backend.elastic_matrix())
    init = _solve_linear_system(
        linear_solver,
        backend.elastic_matrix(),
        np.asarray(backend.source_f, dtype=np.float64).reshape(-1, order="F"),
    )
    snap1 = _collector_snapshot(linear_solver)
    init_delta = _collector_delta(snap0, snap1)
    init_meta = {
        "enabled": True,
        "success": True,
        "ksp_iterations": int(init_delta["iterations"]),
        "solve_time": float(init_delta["solve_time"]),
        "vector_norm": float(np.linalg.norm(np.asarray(init, dtype=np.float64).reshape(-1))),
    }
    _append_stage_event(
        stage_path,
        stage="source_initial_guess_done",
        started=started,
        ksp_iterations=int(init_meta["ksp_iterations"]),
    )
    progress_events: list[dict[str, object]] = []

    def progress_callback(event: dict[str, object]) -> None:
        if PETSc.COMM_WORLD.getRank() == 0:
            progress_events.append(dict(event))
            if str(event.get("event", "")) == "newton_iteration":
                _append_stage_event(
                    stage_path,
                    stage="source_newton_iteration_done",
                    started=started,
                    iteration=int(event.get("iteration", 0)),
                    linear_iterations=int(event.get("linear_iterations", 0)),
                    stopping_value=float(event.get("stopping_value", np.nan)),
                    status=str(event.get("status", "")),
                )

    solve_t0 = time.perf_counter()
    _append_stage_event(stage_path, stage="source_newton_start", started=started)
    U_final, flag_N, nit = source_newton(
        np.asarray(init, dtype=np.float64).reshape((1, backend.n_free), order="F"),
        tol=1.0e-4,
        it_newt_max=int(maxit),
        it_damp_max=10,
        r_min=1.0e-4,
        K_elast=backend.elastic_matrix(),
        Q=backend.source_q,
        f=backend.source_f,
        constitutive_matrix_builder=backend,
        linear_system_solver=linear_solver,
        progress_callback=progress_callback,
        stopping_criterion="relative_correction",
        stopping_tol=float(stop_tol),
    )
    solve_time = time.perf_counter() - solve_t0
    final_global = np.asarray(U_final, dtype=np.float64).reshape(-1, order="F")
    history = [
        {
            "iteration": int(event.get("iteration", 0)),
            "metric": float(event.get("stopping_value", np.nan)),
            "metric_name": str(event.get("stop_criterion", "relative_correction")),
            "alpha": None if event.get("alpha") is None else float(event.get("alpha")),
            "linear_iterations": int(event.get("linear_iterations", 0)),
            "linear_solve_time": float(event.get("linear_solve_time", 0.0)),
            "linear_preconditioner_time": float(
                event.get("linear_preconditioner_time", 0.0)
            ),
            "linear_orthogonalization_time": float(
                event.get("linear_orthogonalization_time", 0.0)
            ),
            "iteration_wall_time": float(event.get("iteration_wall_time", 0.0)),
            "accepted_relative_correction_norm": (
                None
                if event.get("accepted_relative_correction_norm") is None
                else float(event.get("accepted_relative_correction_norm"))
            ),
            "status": str(event.get("status", "")),
        }
        for event in progress_events
        if str(event.get("event", "")) == "newton_iteration"
    ]
    final_metric = float(history[-1]["metric"]) if history else float("nan")
    observables = backend.final_observables(final_global)
    status = "completed" if int(flag_N) == 0 and np.isfinite(final_metric) else "failed"
    builder_timings = None
    builder_timings_path = out_dir / "data" / "source_builder_timings.json"
    try:
        local_builder_timings = dict(backend.const_builder.get_total_time())
        builder_timings = {
            str(key): float(backend.comm.allreduce(float(value), op=MPI.MAX))
            for key, value in local_builder_timings.items()
        }
    except Exception:
        builder_timings = None
    if PETSc.COMM_WORLD.getRank() == 0 and builder_timings is not None:
        builder_timings_path.parent.mkdir(parents=True, exist_ok=True)
        builder_timings_path.write_text(
            json.dumps(builder_timings, indent=2) + "\n",
            encoding="utf-8",
        )
    close = getattr(linear_solver, "close", None)
    if callable(close):
        close()
    _append_stage_event(
        stage_path,
        stage="source_newton_done",
        started=started,
        nit=int(nit),
        status=str(status),
        final_metric=float(final_metric),
    )
    return {
        "status": str(status),
        "solver_success": bool(status == "completed"),
        "message": "Converged" if status == "completed" else "Newton failed or hit maxit",
        "nit": int(nit),
        "solve_time": float(solve_time),
        "total_time": float(solve_time),
        "linear_iterations_total": int(
            sum(int(row["linear_iterations"]) for row in history)
        ),
        "history": history,
        "final_metric": float(final_metric),
        "final_metric_name": "relative_correction",
        "initial_guess": init_meta,
        "source_builder_timings": (
            None if builder_timings is None else dict(builder_timings)
        ),
        "source_builder_timings_path": (
            str(builder_timings_path)
            if PETSc.COMM_WORLD.getRank() == 0 and builder_timings is not None
            else ""
        ),
        **observables,
    }


def _case_payload(
    *,
    assembly_backend: str,
    solver_backend: str,
    stop_tol: float,
    maxit: int,
    line_search: str,
    linesearch_tol: float,
    armijo_max_ls: int,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "assembly_backend": str(assembly_backend),
        "solver_backend": str(solver_backend),
        "ranks": int(PETSc.COMM_WORLD.getSize()),
        "stop_metric_name": str(result.get("final_metric_name", "relative_correction")),
        "stop_tol": float(stop_tol),
        "maxit": int(maxit),
        "line_search": str(line_search),
        "linesearch_tol": float(linesearch_tol),
        "armijo_max_ls": int(armijo_max_ls),
        **result,
    }


def _parallel_setup_summary(backend) -> dict[str, object]:
    comm = PETSc.COMM_WORLD.tompi4py()
    if hasattr(backend, "assembler"):
        assembler = backend.assembler
        memory = dict(assembler.memory_summary())
        local_elements = int(memory.get("local_elements", 0))
        owned_free = int(getattr(assembler.layout, "hi", 0) - getattr(assembler.layout, "lo", 0))
        overlap_total = int(memory.get("local_overlap_dofs", 0))
    else:
        local_elements = 0
        owned_free = int(getattr(backend, "n_local_free", 0))
        overlap_total = int(getattr(backend, "n_local_free", 0))
    elems_all = comm.allgather(local_elements)
    owned_all = comm.allgather(owned_free)
    overlap_all = comm.allgather(overlap_total)
    return {
        "local_elements_min": int(min(elems_all) if elems_all else 0),
        "local_elements_max": int(max(elems_all) if elems_all else 0),
        "local_elements_sum": int(sum(elems_all)),
        "owned_free_dofs_sum": int(sum(owned_all)),
        "overlap_total_dofs_sum": int(sum(overlap_all)),
    }


def _backend_element_degree(backend) -> int:
    if isinstance(backend, LocalAssemblyBackend):
        try:
            return int(backend.params.get("element_degree", 4))
        except Exception:
            return 4
    if isinstance(backend, SourceAssemblyBackend):
        return 4
    return 4


def _export_backend_state(
    backend,
    *,
    state_out: Path | None,
    mesh_name: str,
    constraint_variant: str,
    lambda_target: float,
    u_global: np.ndarray,
    energy: float,
) -> None:
    if state_out is None or PETSc.COMM_WORLD.getRank() != 0:
        return
    u_global = np.asarray(u_global, dtype=np.float64).reshape(-1)
    if u_global.size == 0 or not np.all(np.isfinite(u_global)):
        return
    if not isinstance(backend, LocalAssemblyBackend):
        return

    full_original = np.empty_like(u_global)
    full_original[np.asarray(backend.perm, dtype=np.int64)] = u_global
    u_full = np.asarray(backend.params["u_0"], dtype=np.float64).copy()
    u_full[np.asarray(backend.freedofs, dtype=np.int64)] = full_original
    coords_ref = np.asarray(backend.coords_ref, dtype=np.float64).reshape((-1, 3))
    coords_final = coords_ref + u_full.reshape((-1, 3))
    surface_faces = np.asarray(backend.params.get("surf", np.empty((0, 6))), dtype=np.int32)
    boundary_label = np.asarray(
        backend.params.get("boundary_label", np.zeros(surface_faces.shape[0], dtype=np.int32)),
        dtype=np.int32,
    ).reshape((-1,))

    export_plasticity3d_state_npz(
        state_out,
        coords_ref=coords_ref,
        x_final=coords_final,
        tetrahedra=np.asarray(backend.params["elems_scalar"], dtype=np.int32),
        surface_faces=surface_faces,
        boundary_label=boundary_label,
        mesh_name=str(mesh_name),
        element_degree=int(_backend_element_degree(backend)),
        lambda_target=float(lambda_target),
        energy=float(energy),
        metadata={
            "solver_family": "backend_mix",
            "assembly_backend": "local",
            "mpi_ranks": int(PETSc.COMM_WORLD.getSize()),
            "constraint_variant": str(normalize_constraint_variant(constraint_variant)),
        },
    )


def _stage_timings_from_stage_file(stage_path: Path) -> dict[str, float]:
    if not stage_path.exists():
        return {}
    events: dict[str, float] = {}
    for line in stage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        events[str(obj.get("stage", ""))] = float(obj.get("elapsed_s", 0.0))

    def _dur(end: str, start: str | None = None) -> float:
        end_v = float(events.get(end, 0.0))
        start_v = float(events.get(start, 0.0)) if start is not None else 0.0
        return max(0.0, end_v - start_v)

    return {
        "problem_load": float(events.get("local_problem_loaded", events.get("local_pmg_problem_loaded", 0.0))),
        "assembler_create": _dur("local_assembler_ready", "local_problem_loaded"),
        "mg_hierarchy_build": _dur("local_pmg_hierarchy_ready", "backend_ready"),
        "initial_guess_total": _dur("local_initial_guess_done", "local_initial_guess_start"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Plasticity3D backend-mix case."
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("local", "local_constitutiveAD", "local_sfd", "source"),
        required=True,
    )
    parser.add_argument(
        "--solver-backend",
        choices=(
            LOCAL_SOLVER_FAST,
            LOCAL_SOLVER_PMG,
            LOCAL_SOLVER_PMG_SOURCEFIXED,
            LOCAL_SOLVER_SOURCE_DFGMRES,
            "source",
        ),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--mesh-name", type=str, default="hetero_ssr_L1")
    parser.add_argument("--elem-degree", type=int, default=4)
    parser.add_argument(
        "--constraint-variant",
        choices=(
            DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
            PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
        ),
        default=DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT,
    )
    parser.add_argument("--lambda-target", type=float, default=1.5)
    parser.add_argument(
        "--pmg-strategy",
        choices=(
            "same_mesh_p2_p1",
            "same_mesh_p4_p2_p1",
            "uniform_refined_p4_p2_p1_p1",
            "uniform_refined_p1_chain",
        ),
        default="same_mesh_p4_p2_p1",
    )
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-2)
    parser.add_argument("--ksp-max-it", type=int, default=100)
    parser.add_argument(
        "--convergence-mode",
        choices=("all", "gradient_only"),
        default="all",
    )
    parser.add_argument("--grad-stop-tol", type=float, default=None)
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument(
        "--line-search",
        choices=("armijo", "residual_bisection", "residual_bisection_tol"),
        default="residual_bisection",
    )
    parser.add_argument("--linesearch-tol", type=float, default=1.0e-3)
    parser.add_argument("--armijo-alpha0", type=float, default=1.0)
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument("--armijo-max-ls", type=int, default=40)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    elem_degree = int(args.elem_degree)
    constraint_variant = normalize_constraint_variant(args.constraint_variant)
    out_dir = Path(args.out_dir).resolve()
    output_json = Path(args.output_json).resolve()
    if elem_degree not in {1, 2, 4}:
        raise ValueError("Plasticity3D backend-mix runner supports elem-degree 1, 2, or 4")
    if str(args.assembly_backend) == "source" and elem_degree != 4:
        raise ValueError("Source Plasticity3D assembly supports only --elem-degree 4")
    if str(args.solver_backend) in {LOCAL_SOLVER_SOURCE_DFGMRES, "source"} and elem_degree != 4:
        raise ValueError("Source-backed Plasticity3D linear solvers support only --elem-degree 4")
    case_started = time.perf_counter()
    stage_path = out_dir / "data" / "stage.jsonl"
    if PETSc.COMM_WORLD.getRank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
    _append_stage_event(
        stage_path,
        stage="case_start",
        started=case_started,
        assembly_backend=str(args.assembly_backend),
        solver_backend=str(args.solver_backend),
        constraint_variant=str(constraint_variant),
    )

    pmg_support = None
    lazy_pmg_config: dict[str, object] | None = None
    if str(args.assembly_backend) in {"local", "local_constitutiveAD", "local_sfd"}:
        backend = _build_local_assembly_backend(
            mesh_name=str(args.mesh_name),
            elem_degree=int(elem_degree),
            constraint_variant=str(constraint_variant),
            lambda_target=float(args.lambda_target),
            local_hessian_mode=(
                "sfd_local"
                if str(args.assembly_backend) == "local_sfd"
                else "element"
            ),
            autodiff_tangent_mode=(
                "constitutive"
                if str(args.assembly_backend) == "local_constitutiveAD"
                else "element"
            ),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            stage_path=stage_path,
            stage_started=case_started,
        )
    else:
        source_need_energy_operator = bool(
            str(args.solver_backend)
            in {
                LOCAL_SOLVER_FAST,
                LOCAL_SOLVER_PMG,
                LOCAL_SOLVER_PMG_SOURCEFIXED,
                LOCAL_SOLVER_SOURCE_DFGMRES,
            }
            and str(args.line_search) == "armijo"
        )
        backend = _build_source_assembly_backend(
            source_root=Path(args.source_root).resolve(),
            mesh_name=str(args.mesh_name),
            constraint_variant=str(constraint_variant),
            lambda_target=float(args.lambda_target),
            data_dir=out_dir / "data",
            need_energy_operator=bool(source_need_energy_operator),
            build_elastic_operator=str(args.solver_backend) == "source",
            compute_pattern_elastic_values=str(args.solver_backend) == "source",
            preallocate_tangent_matrix=str(args.solver_backend) == "source",
            stage_path=stage_path,
            stage_started=case_started,
        )
    _append_stage_event(stage_path, stage="backend_ready", started=case_started)
    if _is_local_pmg_solver_backend(str(args.solver_backend)) and pmg_support is None:
        if isinstance(backend, SourceAssemblyBackend):
            lazy_pmg_config = {
                "mesh_name": str(args.mesh_name),
                "elem_degree": int(elem_degree),
                "constraint_variant": str(constraint_variant),
                "lambda_target": float(args.lambda_target),
                "pmg_strategy": str(args.pmg_strategy),
                "use_near_nullspace": False,
                "distributed_layout": (
                    int(backend._free_lo),
                    int(backend._free_hi),
                    1,
                ),
            }
        else:
            pmg_support = _build_local_pmg_support(
                backend=backend,
                mesh_name=str(args.mesh_name),
                elem_degree=int(elem_degree),
                constraint_variant=str(constraint_variant),
                lambda_target=float(args.lambda_target),
                pmg_strategy=str(args.pmg_strategy),
                ksp_rtol=float(args.ksp_rtol),
                ksp_max_it=int(args.ksp_max_it),
                stage_path=stage_path,
                stage_started=case_started,
            )

    if str(args.solver_backend) == LOCAL_SOLVER_SOURCE_DFGMRES and pmg_support is None:
        pmg_support = _build_local_pmg_support(
            backend=backend,
            mesh_name=str(args.mesh_name),
            elem_degree=int(elem_degree),
            constraint_variant=str(constraint_variant),
            lambda_target=float(args.lambda_target),
            pmg_strategy=str(args.pmg_strategy),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            stage_path=stage_path,
            stage_started=case_started,
        )

    total_t0 = time.perf_counter()
    if str(args.solver_backend) in {
        LOCAL_SOLVER_FAST,
        LOCAL_SOLVER_PMG,
        LOCAL_SOLVER_PMG_SOURCEFIXED,
    }:
        result = _run_local_solver_backend(
            backend,
            out_dir=out_dir,
            state_out=None if args.state_out is None else Path(args.state_out).resolve(),
            mesh_name=str(args.mesh_name),
            constraint_variant=str(constraint_variant),
            lambda_target=float(args.lambda_target),
            solver_backend=str(args.solver_backend),
            convergence_mode=str(args.convergence_mode),
            grad_stop_tol=args.grad_stop_tol,
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            line_search=str(args.line_search),
            linesearch_tol=float(args.linesearch_tol),
            armijo_alpha0=float(args.armijo_alpha0),
            armijo_c1=float(args.armijo_c1),
            armijo_shrink=float(args.armijo_shrink),
            armijo_max_ls=int(args.armijo_max_ls),
            pmg_support=pmg_support,
            lazy_pmg_config=lazy_pmg_config,
            stage_path=stage_path,
            stage_started=case_started,
        )
    elif str(args.solver_backend) == LOCAL_SOLVER_SOURCE_DFGMRES:
        if pmg_support is None:
            raise RuntimeError("source_dfgmres backend requires PMG support")
        result = _run_local_solver_backend_with_source_linear(
            backend,
            out_dir=out_dir,
            state_out=None if args.state_out is None else Path(args.state_out).resolve(),
            mesh_name=str(args.mesh_name),
            constraint_variant=str(constraint_variant),
            lambda_target=float(args.lambda_target),
            convergence_mode=str(args.convergence_mode),
            grad_stop_tol=args.grad_stop_tol,
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            line_search=str(args.line_search),
            linesearch_tol=float(args.linesearch_tol),
            armijo_alpha0=float(args.armijo_alpha0),
            armijo_c1=float(args.armijo_c1),
            armijo_shrink=float(args.armijo_shrink),
            armijo_max_ls=int(args.armijo_max_ls),
            pmg_support=pmg_support,
            stage_path=stage_path,
            stage_started=case_started,
        )
    else:
        result = _run_source_solver_backend(
            backend,
            out_dir=out_dir,
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            stage_path=stage_path,
            stage_started=case_started,
        )
    result["total_time"] = float(time.perf_counter() - total_t0)
    payload = _case_payload(
        assembly_backend=str(args.assembly_backend),
        solver_backend=str(args.solver_backend),
        stop_tol=float(args.stop_tol),
        maxit=int(args.maxit),
        line_search=str(args.line_search),
        linesearch_tol=float(args.linesearch_tol),
        armijo_max_ls=int(args.armijo_max_ls),
        result=result,
    )
    payload["mesh_name"] = str(args.mesh_name)
    payload["elem_degree"] = int(_backend_element_degree(backend))
    payload["constraint_variant"] = str(constraint_variant)
    payload["lambda_target"] = float(args.lambda_target)
    payload["pmg_strategy"] = str(args.pmg_strategy)
    payload["ksp_rtol"] = float(args.ksp_rtol)
    payload["ksp_max_it"] = int(args.ksp_max_it)
    payload["state_out"] = "" if args.state_out is None else str(Path(args.state_out).resolve())
    payload["same_mesh_case_path"] = (
        str(same_mesh_case_hdf5_path(str(args.mesh_name), int(elem_degree), constraint_variant))
        if str(args.assembly_backend) in {"local", "local_constitutiveAD", "local_sfd"}
        else ""
    )
    payload["stage_timings"] = _stage_timings_from_stage_file(stage_path)
    payload["parallel_setup"] = _parallel_setup_summary(backend)
    if pmg_support is not None:
        payload["pmg_realized_levels"] = int(pmg_support.realized_levels)
        payload["pmg_pc_backend"] = str(pmg_support.pc_backend)

    if PETSc.COMM_WORLD.getRank() == 0:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
    _append_stage_event(
        stage_path,
        stage="payload_written",
        started=case_started,
        status=str(payload.get("status", "")),
    )
    if pmg_support is not None:
        pmg_support.close()
    backend.close()


if __name__ == "__main__":
    main()
