"""Experimental PETSc PCMG hierarchies for slope-stability h/p ladders."""

from __future__ import annotations

import time
from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import basix
import matplotlib.tri as mtri
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.petsc.reordered_element_base import build_near_nullspace
from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.reordered_element_base import inverse_permutation, select_permutation
from src.core.petsc.reasons import ksp_reason_name
from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D
from src.problems.slope_stability.support import (
    build_same_mesh_lagrange_case_data,
    case_name_for_level,
    load_same_mesh_case_hdf5_light,
)


COORD_DECIMALS = 12
VECTOR_BLOCK_SIZE = 2
TRANSFER_CACHE_DIR = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "meshes"
    / "SlopeStability"
    / "pmg_transfer_cache"
)


@dataclass(frozen=True)
class MGLevelSpace:
    level: int
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
class SlopeStabilityMGHierarchy:
    levels: list[MGLevelSpace]
    prolongations: list[PETSc.Mat]
    restrictions: list[PETSc.Mat]
    injection_indices: list[np.ndarray]
    build_metadata: dict[str, float | int] | None = None

    def cleanup(self) -> None:
        for mat in self.restrictions:
            mat.destroy()
        for mat in self.prolongations:
            mat.destroy()


@dataclass(frozen=True)
class LegacyPMGLevelSmootherConfig:
    ksp_type: str
    pc_type: str
    steps: int


class _KSPInvocationObserver:
    def __init__(self, label: str):
        self.label = str(label)
        self.reset()

    def reset(self) -> None:
        self.solve_invocations = 0
        self.total_iterations = 0
        self.observed_time = 0.0
        self.first_residuals: list[float] = []
        self.last_residuals: list[float] = []
        self._active_start = None
        self._active_first = None
        self._active_last = None

    def preSolve(self, ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        del ksp, b, x
        self.solve_invocations += 1
        self._active_start = float(time.perf_counter())
        self._active_first = None
        self._active_last = None

    def postSolve(self, ksp: PETSc.KSP, b: PETSc.Vec, x: PETSc.Vec) -> None:
        del b, x
        self._close_active(time.perf_counter(), int(ksp.getIterationNumber()))

    def _close_active(self, now: float, iterations: int | None = None) -> None:
        if self._active_start is None:
            return
        self.observed_time += max(0.0, float(now) - float(self._active_start))
        if iterations is not None:
            self.total_iterations += int(iterations)
        if self._active_first is not None:
            self.first_residuals.append(float(self._active_first))
        if self._active_last is not None:
            self.last_residuals.append(float(self._active_last))
        self._active_start = None
        self._active_first = None
        self._active_last = None

    def monitor(self, ksp: PETSc.KSP, its: int, rnorm: float) -> None:
        if int(its) == 0:
            if self._active_start is None:
                self.preSolve(ksp, None, None)
            self._active_first = float(rnorm)
            self._active_last = float(rnorm)
            return
        if self._active_start is None:
            self.preSolve(ksp, None, None)
            self._active_first = float(rnorm)
        self._active_last = float(rnorm)

    def snapshot(self, ksp: PETSc.KSP) -> dict[str, object]:
        self._close_active(time.perf_counter(), None)
        first_arr = np.asarray(self.first_residuals, dtype=np.float64)
        last_arr = np.asarray(self.last_residuals, dtype=np.float64)
        valid = (
            (first_arr.size == last_arr.size)
            and first_arr.size > 0
            and np.all(np.isfinite(first_arr))
            and np.all(np.isfinite(last_arr))
            and np.all(first_arr > 0.0)
        )
        avg_contraction = None
        if valid:
            avg_contraction = float(np.mean(last_arr / first_arr))
        return {
            "label": str(self.label),
            "kind": "ksp",
            "solve_invocations": int(self.solve_invocations),
            "total_iterations": int(self.total_iterations),
            "observed_time_sec": float(self.observed_time),
            "average_residual_contraction": avg_contraction,
            "final_ksp_iterations": int(ksp.getIterationNumber()),
            "final_ksp_reason_code": int(ksp.getConvergedReason()),
            "final_ksp_reason_name": str(ksp_reason_name(int(ksp.getConvergedReason()))),
            "final_residual_norm": float(ksp.getResidualNorm()),
        }


class _TransferApplyObserver:
    def __init__(self, label: str):
        self.label = str(label)
        self.reset()

    def reset(self) -> None:
        self.mult_calls = 0
        self.mult_transpose_calls = 0
        self.mult_time = 0.0
        self.mult_transpose_time = 0.0

    def record_mult(self, elapsed: float) -> None:
        self.mult_calls += 1
        self.mult_time += max(0.0, float(elapsed))

    def record_mult_transpose(self, elapsed: float) -> None:
        self.mult_transpose_calls += 1
        self.mult_transpose_time += max(0.0, float(elapsed))

    def snapshot(self) -> dict[str, object]:
        return {
            "label": str(self.label),
            "kind": "transfer",
            "mult_calls": int(self.mult_calls),
            "mult_time_sec": float(self.mult_time),
            "mult_transpose_calls": int(self.mult_transpose_calls),
            "mult_transpose_time_sec": float(self.mult_transpose_time),
            "observed_time_sec": float(self.mult_time + self.mult_transpose_time),
        }


class _TimedTransferContext:
    def __init__(self, backing: PETSc.Mat, observer: _TransferApplyObserver):
        self.backing = backing
        self.observer = observer

    def mult(self, mat, x, y):
        del mat
        t0 = time.perf_counter()
        self.backing.mult(x, y)
        self.observer.record_mult(time.perf_counter() - t0)

    def multTranspose(self, mat, x, y):
        del mat
        t0 = time.perf_counter()
        self.backing.multTranspose(x, y)
        self.observer.record_mult_transpose(time.perf_counter() - t0)

    def createVecs(self, mat):
        del mat
        return self.backing.createVecs()

    def duplicate(self, mat, op):
        del mat, op
        return self.backing


def _wrap_transfer_operator(
    mat: PETSc.Mat,
    *,
    observer: _TransferApplyObserver,
) -> PETSc.Mat:
    ctx = _TimedTransferContext(mat, observer)
    wrapped = PETSc.Mat().createPython(mat.getSizes(), comm=mat.comm, context=ctx)
    if int(mat.getBlockSize()) > 1:
        wrapped.setBlockSize(int(mat.getBlockSize()))
    wrapped.setUp()
    return wrapped


class PMGObserverSuite:
    def __init__(
        self,
        entries: list[dict[str, object]],
        *,
        keepalive_mats: list[PETSc.Mat] | None = None,
        keepalive_objects: list[object] | None = None,
    ):
        self._entries = list(entries)
        self._keepalive_mats = list(keepalive_mats or [])
        self._keepalive_objects = list(keepalive_objects or [])

    def reset(self) -> None:
        for entry in self._entries:
            observer = entry.get("observer")
            if observer is not None:
                observer.reset()

    def snapshot(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for entry in self._entries:
            observer = entry.get("observer")
            if observer is None:
                continue
            if entry.get("kind") == "transfer":
                record = dict(observer.snapshot())
                record.update(
                    {
                        "level_index": int(entry["level_index"]),
                        "mesh_level": int(entry["mesh_level"]),
                        "degree": int(entry["degree"]),
                        "family": str(entry["family"]),
                        "sweep_role": str(entry["sweep_role"]),
                        "target_mesh_level": int(entry["target_mesh_level"]),
                        "target_degree": int(entry["target_degree"]),
                    }
                )
            else:
                ksp = entry.get("ksp")
                if ksp is None:
                    continue
                record = dict(observer.snapshot(ksp))
                record.update(
                    {
                        "level_index": int(entry["level_index"]),
                        "mesh_level": int(entry["mesh_level"]),
                        "degree": int(entry["degree"]),
                        "family": str(entry["family"]),
                        "sweep_role": str(entry["sweep_role"]),
                        "ksp_type": str(ksp.getType()),
                        "pc_type": str(ksp.getPC().getType()),
                        "max_it": int(ksp.getTolerances()[3]),
                    }
                )
            records.append(record)
        return records


def _level_layout_for_nullspace(space: MGLevelSpace):
    return SimpleNamespace(
        perm=np.asarray(space.perm, dtype=np.int64),
        lo=int(space.lo),
        hi=int(space.hi),
        n_free=int(space.n_free),
    )


def _build_level_nullspace(space: MGLevelSpace, comm: MPI.Comm) -> PETSc.NullSpace | None:
    if "elastic_kernel" not in space.params and (
        "nodes" not in space.params or "freedofs" not in space.params
    ):
        return None
    return build_near_nullspace(
        _level_layout_for_nullspace(space),
        space.params,
        comm,
        kernel_key="elastic_kernel",
    )


def _build_level_coordinates(
    space: MGLevelSpace,
    *,
    block_size: int = VECTOR_BLOCK_SIZE,
) -> np.ndarray | None:
    if int(block_size) <= 1 or int(space.ownership_block_size) != int(block_size):
        return None
    freedofs = np.asarray(space.params["freedofs"], dtype=np.int64)
    nodes = np.asarray(space.params["nodes"], dtype=np.float64)
    owned_orig_free = np.asarray(space.perm[space.lo : space.hi], dtype=np.int64)
    if owned_orig_free.size == 0:
        return None
    owned_total_dofs = np.asarray(freedofs[owned_orig_free], dtype=np.int64)
    if owned_total_dofs.size % int(block_size) != 0:
        return None
    owned_total_dofs = owned_total_dofs.reshape((-1, int(block_size)))
    node_ids = owned_total_dofs[:, 0] // int(block_size)
    return np.asarray(nodes[node_ids], dtype=np.float64)


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
    prefix = str(ksp.getOptionsPrefix() or "")
    if not prefix:
        safe_tag = "".join(ch if ch.isalnum() else "_" for ch in str(prefix_tag))
        prefix = f"slope_{safe_tag}_{id(ksp)}_"
        ksp.setOptionsPrefix(prefix)
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


def _ensure_ksp_options_prefix(ksp: PETSc.KSP, *, prefix_tag: str) -> str:
    prefix = str(ksp.getOptionsPrefix() or "")
    if prefix:
        return prefix
    safe_tag = "".join(ch if ch.isalnum() else "_" for ch in str(prefix_tag))
    prefix = f"slope_{safe_tag}_{id(ksp)}_"
    ksp.setOptionsPrefix(prefix)
    return prefix


def _apply_hypre_system_amg_prefix_options(
    prefix: str,
    *,
    nodal_coarsen: int = 6,
    vec_interp_variant: int = 3,
    strong_threshold: float | None = None,
    coarsen_type: str | None = None,
    max_iter: int = 2,
    tol: float = 0.0,
    relax_type_all: str | None = "symmetric-SOR/Jacobi",
) -> None:
    opts = PETSc.Options()
    opts[f"{prefix}pc_hypre_type"] = "boomeramg"
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


def _configure_coarse_solver(
    coarse: PETSc.KSP,
    *,
    backend: str,
    ksp_type: str,
    pc_type: str,
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
    coarse_prefix = _ensure_ksp_options_prefix(
        coarse, prefix_tag=f"mg_coarse_{backend_name}"
    )
    comm_size = int(coarse.comm.getSize())

    if backend_name in {"hypre", "lu", "jacobi"}:
        coarse.setType(str(ksp_type))
        coarse.setTolerances(rtol=1.0e-10, max_it=200)
        coarse.getPC().setType(str(pc_type))
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

    opts = PETSc.Options()
    if backend_name == "redundant_lu":
        coarse.setType("preonly")
        opts[f"{coarse_prefix}pc_type"] = "redundant"
        opts[f"{coarse_prefix}pc_redundant_number"] = max(1, comm_size)
        opts[f"{coarse_prefix}redundant_ksp_type"] = "preonly"
        opts[f"{coarse_prefix}redundant_pc_type"] = "lu"
        coarse.setFromOptions()
        return

    if backend_name == "redundant_hypre":
        coarse.setType("preonly")
        opts[f"{coarse_prefix}pc_type"] = "redundant"
        opts[f"{coarse_prefix}pc_redundant_number"] = max(1, comm_size)
        opts[f"{coarse_prefix}redundant_ksp_type"] = "cg"
        opts[f"{coarse_prefix}redundant_ksp_rtol"] = 1.0e-10
        opts[f"{coarse_prefix}redundant_ksp_max_it"] = 200
        opts[f"{coarse_prefix}redundant_pc_type"] = "hypre"
        _apply_hypre_system_amg_prefix_options(
            f"{coarse_prefix}redundant_",
            nodal_coarsen=int(hypre_nodal_coarsen),
            vec_interp_variant=int(hypre_vec_interp_variant),
            strong_threshold=hypre_strong_threshold,
            coarsen_type=hypre_coarsen_type,
            max_iter=int(hypre_max_iter),
            tol=float(hypre_tol),
            relax_type_all=hypre_relax_type_all,
        )
        coarse.setFromOptions()
        return

    if backend_name == "rank0_lu_broadcast":
        coarse.setType("preonly")
        opts[f"{coarse_prefix}pc_type"] = "telescope"
        opts[f"{coarse_prefix}pc_telescope_reduction_factor"] = max(1, comm_size)
        opts[f"{coarse_prefix}telescope_ksp_type"] = "preonly"
        opts[f"{coarse_prefix}telescope_pc_type"] = "lu"
        coarse.setFromOptions()
        return

    if backend_name == "rank0_hypre_broadcast":
        coarse.setType("preonly")
        opts[f"{coarse_prefix}pc_type"] = "telescope"
        opts[f"{coarse_prefix}pc_telescope_reduction_factor"] = max(1, comm_size)
        opts[f"{coarse_prefix}telescope_ksp_type"] = "cg"
        opts[f"{coarse_prefix}telescope_ksp_rtol"] = 1.0e-10
        opts[f"{coarse_prefix}telescope_ksp_max_it"] = 200
        opts[f"{coarse_prefix}telescope_pc_type"] = "hypre"
        _apply_hypre_system_amg_prefix_options(
            f"{coarse_prefix}telescope_",
            nodal_coarsen=int(hypre_nodal_coarsen),
            vec_interp_variant=int(hypre_vec_interp_variant),
            strong_threshold=hypre_strong_threshold,
            coarsen_type=hypre_coarsen_type,
            max_iter=int(hypre_max_iter),
            tol=float(hypre_tol),
            relax_type_all=hypre_relax_type_all,
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


def attach_pmg_level_metadata(
    ksp: PETSc.KSP,
    hierarchy: SlopeStabilityMGHierarchy,
    *,
    use_near_nullspace: bool = True,
    block_size: int = VECTOR_BLOCK_SIZE,
    coarse_pc_type: str | None = None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
) -> dict[str, object]:
    def _iter_level_ksps(level_idx: int) -> list[PETSc.KSP]:
        if level_idx == 0:
            return [pc.getMGCoarseSolve()]
        return [pc.getMGSmoother(level_idx)]

    pc = ksp.getPC()
    nullspaces: list[PETSc.NullSpace] = []
    level_records: list[dict[str, object]] = []
    for level_idx, level_space in enumerate(hierarchy.levels):
        level_block_size = int(level_space.ownership_block_size)
        level_nullspace = _build_level_nullspace(level_space, ksp.comm) if use_near_nullspace else None
        level_coordinates = _build_level_coordinates(level_space, block_size=level_block_size)
        level_record = {
            "level_index": int(level_idx),
            "mesh_level": int(level_space.level),
            "degree": int(level_space.degree),
            "ownership_block_size": int(level_block_size),
            "near_nullspace_requested": bool(use_near_nullspace),
            "near_nullspace_attached": bool(level_nullspace is not None),
            "coordinates_attached": bool(level_coordinates is not None),
            "matrix_block_sizes": [],
            "ksp_records": [],
        }
        for level_ksp in _iter_level_ksps(level_idx):
            amat, pmat = level_ksp.getOperators()
            target_mats = []
            if amat is not None:
                target_mats.append(amat)
            if pmat is not None and (amat is None or pmat.handle != amat.handle):
                target_mats.append(pmat)
            for mat in target_mats:
                if not _mat_is_ready_for_metadata(mat):
                    continue
                if level_block_size > 1:
                    mat.setBlockSize(level_block_size)
                level_record["matrix_block_sizes"].append(int(mat.getBlockSize()))
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
            level_record["ksp_records"].append(
                {
                    "ksp_type": str(level_ksp.getType()),
                    "pc_type": str(level_pc.getType()),
                }
            )
        if use_near_nullspace:
            if level_nullspace is not None:
                nullspaces.append(level_nullspace)
        level_records.append(level_record)
    return {
        "nullspaces": nullspaces,
        "levels": level_records,
    }


def _create_level_template_vec(space: MGLevelSpace, comm: MPI.Comm) -> PETSc.Vec:
    vec = PETSc.Vec().createMPI((space.hi - space.lo, space.n_free), comm=comm)
    if int(space.ownership_block_size) > 1:
        vec.setBlockSize(int(space.ownership_block_size))
    vec.set(0.0)
    vec.assemble()
    return vec


@dataclass(frozen=True)
class MGHierarchySpec:
    level: int
    degree: int


_CUSTOM_SPEC_PATTERNS = (
    re.compile(r"^\s*L?(?P<level>\d+)\s*[:/]\s*P?(?P<degree>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*L(?P<level>\d+)\s*P(?P<degree>\d+)\s*$", re.IGNORECASE),
)


def parse_custom_mg_hierarchy_specs(spec_string: str) -> list[MGHierarchySpec]:
    tokens = [token.strip() for token in str(spec_string).split(",") if token.strip()]
    if len(tokens) < 2:
        raise ValueError(
            "custom mixed MG hierarchy requires at least two entries like "
            "'1:1,2:1,6:2,6:4'"
        )
    specs: list[MGHierarchySpec] = []
    for token in tokens:
        match = None
        for pattern in _CUSTOM_SPEC_PATTERNS:
            match = pattern.match(token)
            if match is not None:
                break
        if match is None:
            raise ValueError(
                "Could not parse custom mixed MG entry "
                f"{token!r}; expected entries like '1:1' or 'L1P1'"
            )
        specs.append(
            MGHierarchySpec(
                level=int(match.group("level")),
                degree=int(match.group("degree")),
            )
        )
    return specs


def _coord_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(np.round(np.asarray(point, dtype=np.float64), COORD_DECIMALS).tolist())


def _build_level_space(
    *,
    level: int,
    params: dict[str, object],
    adjacency,
    reorder_mode: str,
    comm: MPI.Comm,
    perm_override: np.ndarray | None = None,
) -> MGLevelSpace:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    if perm_override is None:
        if freedofs.size % VECTOR_BLOCK_SIZE != 0:
            perm = np.arange(freedofs.size, dtype=np.int64)
        else:
            if adjacency is None and str(reorder_mode) not in {"none", "block_xyz"}:
                raise ValueError(
                    "rank_local MG level build currently supports only reorder "
                    "modes 'none' and 'block_xyz' without a global adjacency"
                )
            perm = select_permutation(
                reorder_mode,
                adjacency=adjacency,
                coords_all=np.asarray(params["nodes"], dtype=np.float64),
                freedofs=freedofs,
                n_parts=comm.size,
                block_size=VECTOR_BLOCK_SIZE,
            )
    else:
        perm = np.asarray(perm_override, dtype=np.int64)
    iperm = inverse_permutation(perm)
    ownership_block_size = VECTOR_BLOCK_SIZE if freedofs.size % VECTOR_BLOCK_SIZE == 0 else 1
    lo, hi = petsc_ownership_range(
        int(freedofs.size), comm.rank, comm.size, block_size=ownership_block_size
    )
    total_to_free_orig = np.full(len(np.asarray(params["u_0"], dtype=np.float64)), -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    return MGLevelSpace(
        level=int(level),
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


def _barycentric(point: np.ndarray, verts: np.ndarray) -> np.ndarray:
    p0 = np.asarray(verts[0], dtype=np.float64)
    p1 = np.asarray(verts[1], dtype=np.float64)
    p2 = np.asarray(verts[2], dtype=np.float64)
    mat = np.column_stack((p1 - p0, p2 - p0))
    rhs = np.asarray(point, dtype=np.float64) - p0
    xi = np.linalg.solve(mat, rhs)
    l2 = float(xi[0])
    l3 = float(xi[1])
    l1 = 1.0 - l2 - l3
    return np.array([l1, l2, l3], dtype=np.float64)


def _p2_shape_values(lam: np.ndarray) -> np.ndarray:
    l1, l2, l3 = [float(v) for v in lam]
    return np.array(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l2 * l3,
            4.0 * l1 * l3,
            4.0 * l1 * l2,
        ],
        dtype=np.float64,
    )


def _degree_from_params(params: dict[str, object]) -> int:
    degree = params.get("element_degree")
    if degree is not None:
        return int(degree)
    n_scalar = int(np.asarray(params["elems_scalar"]).shape[1])
    mapping = {3: 1, 6: 2, 15: 4}
    try:
        return int(mapping[n_scalar])
    except KeyError as exc:
        raise ValueError(f"Unsupported scalar triangle size {n_scalar!r}") from exc


def _lagrange_shape_values(degree: int, ref_point: np.ndarray) -> np.ndarray:
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        int(degree),
        basix.LagrangeVariant.equispaced,
    )
    tab = element.tabulate(0, np.asarray(ref_point, dtype=np.float64)[None, :])
    return np.asarray(tab[0, 0, :, 0], dtype=np.float64)


def _find_triangle(
    point: np.ndarray,
    *,
    coarse_nodes: np.ndarray,
    coarse_elems_scalar: np.ndarray,
    trifinder,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    tol: float = 1.0e-10,
) -> int:
    tri_idx = int(trifinder(float(point[0]), float(point[1])))
    if tri_idx >= 0:
        return tri_idx

    candidate_mask = (
        (bbox_min[:, 0] - tol <= point[0])
        & (point[0] <= bbox_max[:, 0] + tol)
        & (bbox_min[:, 1] - tol <= point[1])
        & (point[1] <= bbox_max[:, 1] + tol)
    )
    candidates = np.where(candidate_mask)[0]
    for idx in candidates.tolist():
        verts = coarse_nodes[coarse_elems_scalar[idx, :3]]
        lam = _barycentric(point, verts)
        if np.all(lam >= -tol) and np.all(lam <= 1.0 + tol):
            return int(idx)
    raise RuntimeError(f"Could not locate containing coarse triangle for point {point.tolist()}")


def _build_node_prolongation(
    coarse_nodes: np.ndarray,
    coarse_elems_scalar: np.ndarray,
    coarse_degree: int,
    fine_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse_tris = np.asarray(coarse_elems_scalar[:, :3], dtype=np.int32)
    triang = mtri.Triangulation(
        coarse_nodes[:, 0],
        coarse_nodes[:, 1],
        coarse_tris,
    )
    trifinder = triang.get_trifinder()
    coarse_vert_coords = coarse_nodes[coarse_tris]
    bbox_min = np.min(coarse_vert_coords, axis=1)
    bbox_max = np.max(coarse_vert_coords, axis=1)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for fine_node in range(fine_nodes.shape[0]):
        point = fine_nodes[fine_node]

        tri_idx = _find_triangle(
            point,
            coarse_nodes=coarse_nodes,
            coarse_elems_scalar=coarse_elems_scalar,
            trifinder=trifinder,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
        coarse_elem = np.asarray(coarse_elems_scalar[tri_idx], dtype=np.int64)
        lam = _barycentric(point, coarse_nodes[coarse_elem[:3]])
        ref_point = np.array([lam[1], lam[2]], dtype=np.float64)
        values = _lagrange_shape_values(int(coarse_degree), ref_point)
        for coarse_node, weight in zip(coarse_elem.tolist(), values.tolist()):
            if abs(weight) > 1.0e-14:
                rows.append(int(fine_node))
                cols.append(int(coarse_node))
                data.append(float(weight))

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
    )


def _node_transfer_cache_path(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
) -> Path:
    TRANSFER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSFER_CACHE_DIR / (
        f"level{int(coarse.level)}_p{int(coarse.degree)}"
        f"_to_level{int(fine.level)}_p{int(fine.degree)}.npz"
    )


def _load_or_build_node_prolongation(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    cache_path = _node_transfer_cache_path(coarse, fine)
    t0 = time.perf_counter()
    if cache_path.exists():
        with np.load(cache_path) as handle:
            rows = np.asarray(handle["rows"], dtype=np.int64)
            cols = np.asarray(handle["cols"], dtype=np.int64)
            data = np.asarray(handle["data"], dtype=np.float64)
        return rows, cols, data, {
            "cache_hit": 1,
            "cache_io_time": float(time.perf_counter() - t0),
            "cache_build_time": 0.0,
        }

    rows, cols, data = _build_node_prolongation(
        np.asarray(coarse.params["nodes"], dtype=np.float64),
        np.asarray(coarse.params["elems_scalar"], dtype=np.int64),
        coarse.degree,
        np.asarray(fine.params["nodes"], dtype=np.float64),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(
        f".{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    np.savez_compressed(tmp_path, rows=rows, cols=cols, data=data)
    tmp_path.replace(cache_path)
    return rows, cols, data, {
        "cache_hit": 0,
        "cache_io_time": 0.0,
        "cache_build_time": float(time.perf_counter() - t0),
    }


def _build_free_reordered_prolongation(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
    comm: MPI.Comm,
    *,
    build_mode: str = "replicated",
) -> tuple[PETSc.Mat, PETSc.Mat]:
    coarse_nodes = np.asarray(coarse.params["nodes"], dtype=np.float64)
    fine_nodes = np.asarray(fine.params["nodes"], dtype=np.float64)
    coarse_elems_scalar = np.asarray(coarse.params["elems_scalar"], dtype=np.int64)
    build_mode = str(build_mode)
    if build_mode not in {"replicated", "root_bcast", "owned_rows"}:
        raise ValueError(f"Unsupported transfer build mode {build_mode!r}")

    transfer_meta = {
        "cache_hits": 0,
        "cache_io_time": 0.0,
        "cache_build_time": 0.0,
        "mapping_time": 0.0,
        "matrix_build_time": 0.0,
    }
    if build_mode == "root_bcast" and int(comm.size) > 1:
        if int(comm.rank) == 0:
            node_rows, node_cols, node_data, cache_meta = _load_or_build_node_prolongation(
                coarse, fine
            )
        else:
            node_rows = node_cols = node_data = cache_meta = None
        node_rows, node_cols, node_data, cache_meta = comm.bcast(
            (node_rows, node_cols, node_data, cache_meta),
            root=0,
        )
    else:
        node_rows, node_cols, node_data, cache_meta = _load_or_build_node_prolongation(
            coarse, fine
        )
    transfer_meta["cache_hits"] = int(cache_meta["cache_hit"])
    transfer_meta["cache_io_time"] = float(cache_meta["cache_io_time"])
    transfer_meta["cache_build_time"] = float(cache_meta["cache_build_time"])

    t_map0 = time.perf_counter()
    row_list: list[int] = []
    col_list: list[int] = []
    data_list: list[float] = []
    for node_row, node_col, weight in zip(
        node_rows.tolist(),
        node_cols.tolist(),
        node_data.tolist(),
    ):
        for comp in range(VECTOR_BLOCK_SIZE):
            total_row = VECTOR_BLOCK_SIZE * int(node_row) + comp
            total_col = VECTOR_BLOCK_SIZE * int(node_col) + comp
            fine_free_orig = int(fine.total_to_free_orig[total_row])
            coarse_free_orig = int(coarse.total_to_free_orig[total_col])
            if fine_free_orig < 0 or coarse_free_orig < 0:
                continue
            fine_reordered = int(fine.iperm[fine_free_orig])
            if build_mode == "owned_rows" and not (int(fine.lo) <= fine_reordered < int(fine.hi)):
                continue
            row_list.append(fine_reordered)
            col_list.append(int(coarse.iperm[coarse_free_orig]))
            data_list.append(float(weight))
    rows = np.asarray(row_list, dtype=np.int64)
    cols = np.asarray(col_list, dtype=np.int64)
    data = np.asarray(data_list, dtype=np.float64)
    transfer_meta["mapping_time"] = float(time.perf_counter() - t_map0)

    def _build_matrix(
        mat_rows: np.ndarray,
        mat_cols: np.ndarray,
        mat_vals: np.ndarray,
        *,
        row_lo: int,
        row_hi: int,
        n_rows: int,
        col_lo: int,
        col_hi: int,
        n_cols: int,
    ) -> PETSc.Mat:
        owned_mask = (mat_rows >= row_lo) & (mat_rows < row_hi)
        owned_rows = mat_rows[owned_mask]
        owned_cols = mat_cols[owned_mask]
        owned_vals = mat_vals[owned_mask]
        mat = PETSc.Mat().create(comm=comm)
        mat.setType(PETSc.Mat.Type.MPIAIJ)
        mat.setSizes(((row_hi - row_lo, n_rows), (col_hi - col_lo, n_cols)))
        mat.setPreallocationCOO(
            owned_rows.astype(PETSc.IntType),
            owned_cols.astype(PETSc.IntType),
        )
        mat.setBlockSize(1)
        mat.setValuesCOO(
            owned_vals.astype(PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        mat.assemble()
        return mat

    t_mat0 = time.perf_counter()
    prolong = _build_matrix(
        rows,
        cols,
        data,
        row_lo=fine.lo,
        row_hi=fine.hi,
        n_rows=fine.n_free,
        col_lo=coarse.lo,
        col_hi=coarse.hi,
        n_cols=coarse.n_free,
    )
    if build_mode == "owned_rows":
        restrict = prolong.copy()
        restrict.transpose()
    else:
        restrict = _build_matrix(
            cols,
            rows,
            data,
            row_lo=coarse.lo,
            row_hi=coarse.hi,
            n_rows=coarse.n_free,
            col_lo=fine.lo,
            col_hi=fine.hi,
            n_cols=fine.n_free,
        )
    transfer_meta["matrix_build_time"] = float(time.perf_counter() - t_mat0)
    return prolong, restrict, transfer_meta


def build_reordered_injection_indices(
    coarse: MGLevelSpace,
    fine: MGLevelSpace,
) -> np.ndarray:
    """Map coarse reordered free DOFs to matching fine reordered free DOFs."""
    coarse_nodes = np.asarray(coarse.params["nodes"], dtype=np.float64)
    fine_nodes = np.asarray(fine.params["nodes"], dtype=np.float64)
    coarse_freedofs = np.asarray(coarse.params["freedofs"], dtype=np.int64)

    fine_node_lookup = {
        _coord_key(point): int(node_idx)
        for node_idx, point in enumerate(np.asarray(fine_nodes, dtype=np.float64))
    }
    fine_iperm = np.asarray(fine.iperm, dtype=np.int64)

    coarse_to_fine = np.empty(coarse.n_free, dtype=np.int64)
    for coarse_reordered_pos, coarse_free_orig in enumerate(
        np.asarray(coarse.perm, dtype=np.int64)
    ):
        coarse_total_dof = int(coarse_freedofs[coarse_free_orig])
        coarse_node = coarse_total_dof // VECTOR_BLOCK_SIZE
        comp = coarse_total_dof % VECTOR_BLOCK_SIZE
        try:
            fine_node = fine_node_lookup[_coord_key(coarse_nodes[coarse_node])]
        except KeyError as exc:
            raise RuntimeError(
                "Could not inject coarse MG state into finer space because node "
                f"{coarse_node} at {coarse_nodes[coarse_node].tolist()} is missing"
            ) from exc
        fine_total_dof = VECTOR_BLOCK_SIZE * int(fine_node) + int(comp)
        fine_free_orig = int(fine.total_to_free_orig[fine_total_dof])
        if fine_free_orig < 0:
            raise RuntimeError(
                "Injection from fine to coarse MG state hit a constrained DOF in the "
                f"fine space for total dof {fine_total_dof}"
            )
        coarse_to_fine[coarse_reordered_pos] = int(fine_iperm[fine_free_orig])
    return coarse_to_fine


def _load_level_from_mesh(level: int, reorder_mode: str, comm: MPI.Comm) -> MGLevelSpace:
    mesh = MeshSlopeStability2D(level=level)
    params, adjacency, _ = mesh.get_data()
    return _build_level_space(
        level=level,
        params=params,
        adjacency=adjacency,
        reorder_mode=reorder_mode,
        comm=comm,
    )


def _load_level_from_spec(
    spec: MGHierarchySpec,
    reorder_mode: str,
    comm: MPI.Comm,
    *,
    build_mode: str = "replicated",
) -> MGLevelSpace:
    if str(build_mode) == "rank_local":
        params = load_same_mesh_case_hdf5_light(int(spec.level), int(spec.degree))
        adjacency = None
    else:
        case_data = build_same_mesh_lagrange_case_data(
            case_name_for_level(int(spec.level)),
            degree=int(spec.degree),
            build_mode=str(build_mode),
            comm=comm,
        )
        params = dict(case_data.__dict__)
        adjacency = case_data.adjacency
    params["elem_type"] = f"P{int(spec.degree)}"
    params["element_degree"] = int(spec.degree)
    return _build_level_space(
        level=int(spec.level),
        params=params,
        adjacency=adjacency,
        reorder_mode=reorder_mode,
        comm=comm,
    )


def build_pmg_hierarchy(
    *,
    finest_level: int,
    coarsest_level: int,
    finest_params: dict[str, object],
    finest_adjacency,
    finest_perm: np.ndarray,
    reorder_mode: str,
    comm: MPI.Comm,
    level_build_mode: str = "replicated",
    transfer_build_mode: str = "replicated",
) -> SlopeStabilityMGHierarchy | None:
    finest_level = int(finest_level)
    coarsest_level = int(coarsest_level)
    if finest_level <= 1:
        return None
    if coarsest_level < 1 or coarsest_level > finest_level - 1:
        raise ValueError(
            f"mg coarsest level must lie in [1, {finest_level - 1}] for finest level {finest_level}"
        )

    levels: list[MGLevelSpace] = []
    level_records: list[dict[str, object]] = []
    t_levels0 = time.perf_counter()
    for level in range(coarsest_level, finest_level):
        t_level0 = time.perf_counter()
        level_space = _load_level_from_mesh(level, reorder_mode, comm)
        levels.append(level_space)
        level_records.append(
            {
                "level": int(level),
                "degree": 2,
                "build_time": float(time.perf_counter() - t_level0),
                "n_free": int(level_space.n_free),
                "ownership_block_size": int(level_space.ownership_block_size),
            }
        )
    t_finest0 = time.perf_counter()
    levels.append(
        _build_level_space(
            level=finest_level,
            params=finest_params,
            adjacency=finest_adjacency,
            reorder_mode=reorder_mode,
            comm=comm,
            perm_override=np.asarray(finest_perm, dtype=np.int64),
        )
    )
    level_records.append(
        {
            "level": int(finest_level),
            "degree": int(finest_params.get("element_degree", 2)),
            "build_time": float(time.perf_counter() - t_finest0),
            "n_free": int(levels[-1].n_free),
            "ownership_block_size": int(levels[-1].ownership_block_size),
        }
    )
    level_build_time = float(time.perf_counter() - t_levels0)

    prolongations: list[PETSc.Mat] = []
    restrictions: list[PETSc.Mat] = []
    transfer_records: list[dict[str, object]] = []
    transfer_meta_total = {
        "cache_hits": 0,
        "cache_io_time": 0.0,
        "cache_build_time": 0.0,
        "mapping_time": 0.0,
        "matrix_build_time": 0.0,
    }
    t_transfers0 = time.perf_counter()
    for coarse, fine in zip(levels[:-1], levels[1:]):
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
                "coarse_level": int(coarse.level),
                "coarse_degree": int(coarse.degree),
                "fine_level": int(fine.level),
                "fine_degree": int(fine.degree),
                "cache_hits": int(transfer_meta["cache_hits"]),
                "cache_io_time": float(transfer_meta["cache_io_time"]),
                "cache_build_time": float(transfer_meta["cache_build_time"]),
                "mapping_time": float(transfer_meta["mapping_time"]),
                "matrix_build_time": float(transfer_meta["matrix_build_time"]),
            }
        )
        for key in transfer_meta_total:
            transfer_meta_total[key] = float(transfer_meta_total[key]) + float(
                transfer_meta[key]
            )
    transfer_build_time = float(time.perf_counter() - t_transfers0)
    return SlopeStabilityMGHierarchy(
        levels=levels,
        prolongations=prolongations,
        restrictions=restrictions,
        injection_indices=[],
        build_metadata={
            "level_build_time": float(level_build_time),
            "transfer_build_time": float(transfer_build_time),
            "transfer_cache_hits": int(transfer_meta_total["cache_hits"]),
            "transfer_cache_io_time": float(transfer_meta_total["cache_io_time"]),
            "transfer_cache_build_time": float(transfer_meta_total["cache_build_time"]),
            "transfer_mapping_time": float(transfer_meta_total["mapping_time"]),
            "transfer_matrix_build_time": float(transfer_meta_total["matrix_build_time"]),
            "level_records": level_records,
            "transfer_records": transfer_records,
        },
    )


def mixed_hierarchy_specs(
    *,
    finest_level: int,
    finest_degree: int,
    strategy: str,
    custom_hierarchy: str | None = None,
) -> list[MGHierarchySpec]:
    finest_level = int(finest_level)
    finest_degree = int(finest_degree)
    tail_level = max(1, finest_level - 1)
    if strategy == "same_mesh_p2_p1":
        if finest_degree != 2:
            raise ValueError("same_mesh_p2_p1 requires finest degree 2")
        return [MGHierarchySpec(finest_level, 1), MGHierarchySpec(finest_level, 2)]
    if strategy == "same_mesh_p2_p1_lminus1_p1":
        if finest_degree != 2:
            raise ValueError("same_mesh_p2_p1_lminus1_p1 requires finest degree 2")
        specs = [MGHierarchySpec(finest_level, 1), MGHierarchySpec(finest_level, 2)]
        if tail_level < finest_level:
            specs.insert(0, MGHierarchySpec(tail_level, 1))
        return specs
    if strategy == "same_mesh_p4_p1":
        if finest_degree != 4:
            raise ValueError("same_mesh_p4_p1 requires finest degree 4")
        return [MGHierarchySpec(finest_level, 1), MGHierarchySpec(finest_level, 4)]
    if strategy == "same_mesh_p4_p2_p1":
        if finest_degree != 4:
            raise ValueError("same_mesh_p4_p2_p1 requires finest degree 4")
        return [
            MGHierarchySpec(finest_level, 1),
            MGHierarchySpec(finest_level, 2),
            MGHierarchySpec(finest_level, 4),
        ]
    if strategy == "same_mesh_p4_p1_lminus1_p1":
        if finest_degree != 4:
            raise ValueError("same_mesh_p4_p1_lminus1_p1 requires finest degree 4")
        specs = [MGHierarchySpec(finest_level, 1), MGHierarchySpec(finest_level, 4)]
        if tail_level < finest_level:
            specs.insert(0, MGHierarchySpec(tail_level, 1))
        return specs
    if strategy == "same_mesh_p4_p2_p1_lminus1_p1":
        if finest_degree != 4:
            raise ValueError("same_mesh_p4_p2_p1_lminus1_p1 requires finest degree 4")
        specs = [
            MGHierarchySpec(finest_level, 1),
            MGHierarchySpec(finest_level, 2),
            MGHierarchySpec(finest_level, 4),
        ]
        if tail_level < finest_level:
            specs.insert(0, MGHierarchySpec(tail_level, 1))
        return specs
    if strategy == "custom_mixed":
        if custom_hierarchy is None or not str(custom_hierarchy).strip():
            raise ValueError(
                "custom_mixed requires --mg_custom_hierarchy with entries like "
                "'1:1,2:1,6:2,6:4'"
            )
        specs = parse_custom_mg_hierarchy_specs(str(custom_hierarchy))
        finest_spec = specs[-1]
        if int(finest_spec.level) != finest_level or int(finest_spec.degree) != finest_degree:
            raise ValueError(
                "custom mixed hierarchy must end at the active finest space "
                f"L{finest_level} P{finest_degree}; got "
                f"L{int(finest_spec.level)} P{int(finest_spec.degree)}"
            )
        return specs
    raise ValueError(f"Unsupported mixed hierarchy strategy {strategy!r}")


def build_mixed_pmg_hierarchy(
    *,
    specs: list[MGHierarchySpec],
    finest_params: dict[str, object],
    finest_adjacency,
    finest_perm: np.ndarray,
    reorder_mode: str,
    comm: MPI.Comm,
    level_build_mode: str = "replicated",
    transfer_build_mode: str = "replicated",
) -> SlopeStabilityMGHierarchy:
    if len(specs) < 2:
        raise ValueError("mixed mg hierarchy requires at least two spaces")

    levels: list[MGLevelSpace] = []
    level_records: list[dict[str, object]] = []
    t_levels0 = time.perf_counter()
    for spec in specs[:-1]:
        t_level0 = time.perf_counter()
        level_space = _load_level_from_spec(
                spec,
                reorder_mode,
                comm,
                build_mode=str(level_build_mode),
            )
        levels.append(level_space)
        level_records.append(
            {
                "level": int(spec.level),
                "degree": int(spec.degree),
                "build_time": float(time.perf_counter() - t_level0),
                "n_free": int(level_space.n_free),
                "ownership_block_size": int(level_space.ownership_block_size),
            }
        )
    finest_spec = specs[-1]
    t_finest0 = time.perf_counter()
    levels.append(
        _build_level_space(
            level=int(finest_spec.level),
            params=finest_params,
            adjacency=finest_adjacency,
            reorder_mode=reorder_mode,
            comm=comm,
            perm_override=np.asarray(finest_perm, dtype=np.int64),
        )
    )
    level_records.append(
        {
            "level": int(finest_spec.level),
            "degree": int(finest_spec.degree),
            "build_time": float(time.perf_counter() - t_finest0),
            "n_free": int(levels[-1].n_free),
            "ownership_block_size": int(levels[-1].ownership_block_size),
        }
    )
    level_build_time = float(time.perf_counter() - t_levels0)

    prolongations: list[PETSc.Mat] = []
    restrictions: list[PETSc.Mat] = []
    transfer_records: list[dict[str, object]] = []
    transfer_meta_total = {
        "cache_hits": 0,
        "cache_io_time": 0.0,
        "cache_build_time": 0.0,
        "mapping_time": 0.0,
        "matrix_build_time": 0.0,
    }
    t_transfers0 = time.perf_counter()
    for coarse, fine in zip(levels[:-1], levels[1:]):
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
                "coarse_level": int(coarse.level),
                "coarse_degree": int(coarse.degree),
                "fine_level": int(fine.level),
                "fine_degree": int(fine.degree),
                "cache_hits": int(transfer_meta["cache_hits"]),
                "cache_io_time": float(transfer_meta["cache_io_time"]),
                "cache_build_time": float(transfer_meta["cache_build_time"]),
                "mapping_time": float(transfer_meta["mapping_time"]),
                "matrix_build_time": float(transfer_meta["matrix_build_time"]),
            }
        )
        for key in transfer_meta_total:
            transfer_meta_total[key] = float(transfer_meta_total[key]) + float(
                transfer_meta[key]
            )
    transfer_build_time = float(time.perf_counter() - t_transfers0)
    return SlopeStabilityMGHierarchy(
        levels=levels,
        prolongations=prolongations,
        restrictions=restrictions,
        injection_indices=[],
        build_metadata={
            "level_build_time": float(level_build_time),
            "transfer_build_time": float(transfer_build_time),
            "transfer_cache_hits": int(transfer_meta_total["cache_hits"]),
            "transfer_cache_io_time": float(transfer_meta_total["cache_io_time"]),
            "transfer_cache_build_time": float(transfer_meta_total["cache_build_time"]),
            "transfer_mapping_time": float(transfer_meta_total["mapping_time"]),
            "transfer_matrix_build_time": float(transfer_meta_total["matrix_build_time"]),
            "level_records": level_records,
            "transfer_records": transfer_records,
        },
    )


def configure_pmg(
    ksp: PETSc.KSP,
    hierarchy: SlopeStabilityMGHierarchy,
    *,
    smoother_steps: int = 3,
    smoother_pc_type: str = "sor",
    level_smoothers: dict[str, LegacyPMGLevelSmootherConfig] | None = None,
    coarse_backend: str = "hypre",
    coarse_ksp_type: str | None = None,
    coarse_pc_type: str | None = None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
) -> PMGObserverSuite:
    def _configure_level_ksp(
        level_ksp: PETSc.KSP,
        *,
        cfg: LegacyPMGLevelSmootherConfig,
    ) -> None:
        level_ksp.setType(str(cfg.ksp_type))
        level_ksp.setTolerances(max_it=int(cfg.steps))
        level_pc = level_ksp.getPC()
        level_pc.setType(str(cfg.pc_type))
        if str(cfg.pc_type) == "hypre":
            level_pc.setHYPREType("boomeramg")

    observed_handles: set[int] = set()

    def _maybe_attach_observer(
        *,
        level_ksp: PETSc.KSP,
        label: str,
        level_index: int,
        mesh_level: int,
        degree: int,
        family: str,
        sweep_role: str,
        entries: list[dict[str, object]],
    ) -> None:
        handle = int(level_ksp.handle)
        if handle in observed_handles:
            return
        observed_handles.add(handle)
        observer = _KSPInvocationObserver(label)
        level_ksp.setMonitor(observer.monitor)
        level_ksp.setPreSolve(observer.preSolve)
        level_ksp.setPostSolve(observer.postSolve)
        entries.append(
            {
                "observer": observer,
                "ksp": level_ksp,
                "level_index": int(level_index),
                "mesh_level": int(mesh_level),
                "degree": int(degree),
                "family": str(family),
                "sweep_role": str(sweep_role),
            }
        )

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.MG)
    pc.setMGLevels(len(hierarchy.levels))
    pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
    pc.setMGCycleType(PETSc.PC.MGCycleType.V)
    keepalive_objects: list[object] = []
    for level_idx, (prolong, restrict) in enumerate(
        zip(hierarchy.prolongations, hierarchy.restrictions),
        start=1,
    ):
        pc.setMGInterpolation(level_idx, prolong)
        pc.setMGRestriction(level_idx, restrict)

    finest_level_idx = len(hierarchy.levels) - 1
    for level_idx, space in enumerate(hierarchy.levels):
        if level_idx < finest_level_idx:
            x_vec = _create_level_template_vec(space, ksp.comm)
            rhs_vec = _create_level_template_vec(space, ksp.comm)
            pc.setMGX(level_idx, x_vec)
            pc.setMGRhs(level_idx, rhs_vec)
            keepalive_objects.extend([x_vec, rhs_vec])
        if 0 < level_idx < finest_level_idx:
            r_vec = _create_level_template_vec(space, ksp.comm)
            pc.setMGR(level_idx, r_vec)
            keepalive_objects.append(r_vec)

    smoother_defaults = dict(level_smoothers or {})
    default_cfg = LegacyPMGLevelSmootherConfig(
        ksp_type="richardson",
        pc_type=str(smoother_pc_type),
        steps=int(smoother_steps),
    )
    fine_cfg = smoother_defaults.get("fine", default_cfg)
    p2_cfg = smoother_defaults.get("degree2", default_cfg)
    p1_cfg = smoother_defaults.get("degree1", default_cfg)
    observer_entries: list[dict[str, object]] = []
    for level_idx in range(1, len(hierarchy.levels)):
        level_space = hierarchy.levels[level_idx]
        cfg = fine_cfg if level_idx == finest_level_idx else (
            p2_cfg if int(level_space.degree) == 2 else p1_cfg
        )
        for variant_name, level_ksp in (
            ("smoother", pc.getMGSmoother(level_idx)),
            ("smoother_down", pc.getMGSmootherDown(level_idx)),
            ("smoother_up", pc.getMGSmootherUp(level_idx)),
        ):
            _configure_level_ksp(level_ksp, cfg=cfg)
            _maybe_attach_observer(
                level_ksp=level_ksp,
                label=f"{variant_name}_level{level_idx}_deg{level_space.degree}",
                level_index=int(level_idx),
                mesh_level=int(level_space.level),
                degree=int(level_space.degree),
                family="fine"
                if level_idx == finest_level_idx
                else ("degree2" if int(level_space.degree) == 2 else "degree1"),
                sweep_role=str(variant_name),
                entries=observer_entries,
            )

    if coarse_ksp_type is None:
        coarse_ksp_type = "cg"
    if coarse_pc_type is None:
        coarse_pc_type = "hypre"
    coarse = pc.getMGCoarseSolve()
    _configure_coarse_solver(
        coarse,
        backend=str(coarse_backend),
        ksp_type=str(coarse_ksp_type),
        pc_type=str(coarse_pc_type),
        hypre_nodal_coarsen=int(coarse_hypre_nodal_coarsen),
        hypre_vec_interp_variant=int(coarse_hypre_vec_interp_variant),
        hypre_strong_threshold=coarse_hypre_strong_threshold,
        hypre_coarsen_type=coarse_hypre_coarsen_type,
        hypre_max_iter=int(coarse_hypre_max_iter),
        hypre_tol=float(coarse_hypre_tol),
        hypre_relax_type_all=coarse_hypre_relax_type_all,
        coordinates=_build_level_coordinates(hierarchy.levels[0]),
    )
    _maybe_attach_observer(
        level_ksp=coarse,
        label="coarse",
        level_index=0,
        mesh_level=int(hierarchy.levels[0].level),
        degree=int(hierarchy.levels[0].degree),
        family="coarse",
        sweep_role="coarse",
        entries=observer_entries,
    )
    return PMGObserverSuite(observer_entries, keepalive_objects=keepalive_objects)


def configure_explicit_pmg(
    ksp: PETSc.KSP,
    hierarchy: SlopeStabilityMGHierarchy,
    *,
    intermediate_smoother_steps: int = 3,
    intermediate_smoother_pc_type: str = "jacobi",
    intermediate_degree_pc_types: dict[int, str] | None = None,
    finest_smoother_ksp_type: str = "richardson",
    finest_smoother_pc_type: str = "none",
    finest_smoother_steps: int = 2,
    finest_smoother_down_ksp_type: str | None = None,
    finest_smoother_down_pc_type: str | None = None,
    finest_smoother_down_steps: int | None = None,
    finest_smoother_up_ksp_type: str | None = None,
    finest_smoother_up_pc_type: str | None = None,
    finest_smoother_up_steps: int | None = None,
    finest_smoother_pc_context=None,
    finest_smoother_down_pc_context=None,
    finest_smoother_up_pc_context=None,
    coarse_ksp_type: str | None = None,
    coarse_pc_type: str | None = None,
    coarse_hypre_nodal_coarsen: int = 6,
    coarse_hypre_vec_interp_variant: int = 3,
    coarse_hypre_strong_threshold: float | None = None,
    coarse_hypre_coarsen_type: str | None = None,
    coarse_hypre_max_iter: int = 2,
    coarse_hypre_tol: float = 0.0,
    coarse_hypre_relax_type_all: str | None = "symmetric-SOR/Jacobi",
    observe_transfers: bool = True,
) -> PMGObserverSuite:
    def _configure_level_ksp(
        level_ksp: PETSc.KSP,
        *,
        ksp_type: str,
        pc_type: str,
        steps: int,
        pc_context=None,
    ) -> None:
        level_ksp.setType(str(ksp_type))
        level_ksp.setTolerances(max_it=int(steps))
        level_pc = level_ksp.getPC()
        level_pc.setType(str(pc_type))
        if str(pc_type) == "hypre":
            level_pc.setHYPREType("boomeramg")
        elif str(pc_type) == "python":
            if pc_context is None:
                raise ValueError("python PC requested without a PETSc Python context")
            level_pc.setPythonContext(pc_context)

    observed_handles: set[int] = set()

    def _maybe_attach_observer(
        *,
        level_ksp: PETSc.KSP,
        label: str,
        level_index: int,
        mesh_level: int,
        degree: int,
        family: str,
        sweep_role: str,
        entries: list[dict[str, object]],
    ) -> None:
        handle = int(level_ksp.handle)
        if handle in observed_handles:
            return
        observed_handles.add(handle)
        observer = _KSPInvocationObserver(label)
        level_ksp.setMonitor(observer.monitor)
        level_ksp.setPreSolve(observer.preSolve)
        level_ksp.setPostSolve(observer.postSolve)
        entries.append(
            {
                "observer": observer,
                "ksp": level_ksp,
                "level_index": int(level_index),
                "mesh_level": int(mesh_level),
                "degree": int(degree),
                "family": str(family),
                "sweep_role": str(sweep_role),
            }
        )

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.MG)
    pc.setMGLevels(len(hierarchy.levels))
    pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
    pc.setMGCycleType(PETSc.PC.MGCycleType.V)
    observer_entries: list[dict[str, object]] = []
    keepalive_transfer_mats: list[PETSc.Mat] = []
    keepalive_objects: list[object] = []
    for level_idx, (prolong, restrict) in enumerate(
        zip(hierarchy.prolongations, hierarchy.restrictions),
        start=1,
    ):
        prolong_to_use = prolong
        restrict_to_use = restrict
        coarse_space = hierarchy.levels[level_idx - 1]
        fine_space = hierarchy.levels[level_idx]
        if observe_transfers:
            prolong_observer = _TransferApplyObserver(
                f"prolongation_level{level_idx-1}_to_{level_idx}"
            )
            prolong_to_use = _wrap_transfer_operator(prolong, observer=prolong_observer)
            observer_entries.append(
                {
                    "kind": "transfer",
                    "observer": prolong_observer,
                    "level_index": int(level_idx),
                    "mesh_level": int(fine_space.level),
                    "degree": int(fine_space.degree),
                    "family": "transfer",
                    "sweep_role": "prolongation",
                    "target_mesh_level": int(coarse_space.level),
                    "target_degree": int(coarse_space.degree),
                }
            )
            keepalive_transfer_mats.append(prolong_to_use)

            restrict_observer = _TransferApplyObserver(
                f"restriction_level{level_idx}_to_{level_idx-1}"
            )
            restrict_to_use = _wrap_transfer_operator(restrict, observer=restrict_observer)
            observer_entries.append(
                {
                    "kind": "transfer",
                    "observer": restrict_observer,
                    "level_index": int(level_idx),
                    "mesh_level": int(fine_space.level),
                    "degree": int(fine_space.degree),
                    "family": "transfer",
                    "sweep_role": "restriction",
                    "target_mesh_level": int(coarse_space.level),
                    "target_degree": int(coarse_space.degree),
                }
            )
            keepalive_transfer_mats.append(restrict_to_use)
        pc.setMGInterpolation(level_idx, prolong_to_use)
        pc.setMGRestriction(level_idx, restrict_to_use)

    finest_level_idx = len(hierarchy.levels) - 1
    for level_idx, space in enumerate(hierarchy.levels):
        if level_idx < finest_level_idx:
            x_vec = _create_level_template_vec(space, ksp.comm)
            rhs_vec = _create_level_template_vec(space, ksp.comm)
            pc.setMGX(level_idx, x_vec)
            pc.setMGRhs(level_idx, rhs_vec)
            keepalive_objects.extend([x_vec, rhs_vec])
        if 0 < level_idx < finest_level_idx:
            r_vec = _create_level_template_vec(space, ksp.comm)
            pc.setMGR(level_idx, r_vec)
            keepalive_objects.append(r_vec)

    degree_pc_types = dict(intermediate_degree_pc_types or {})
    fine_down_ksp_type = str(finest_smoother_down_ksp_type or finest_smoother_ksp_type)
    fine_down_pc_type = str(finest_smoother_down_pc_type or finest_smoother_pc_type)
    fine_down_steps = int(
        finest_smoother_down_steps
        if finest_smoother_down_steps is not None
        else finest_smoother_steps
    )
    fine_up_ksp_type = str(finest_smoother_up_ksp_type or fine_down_ksp_type)
    fine_up_pc_type = str(finest_smoother_up_pc_type or fine_down_pc_type)
    fine_up_steps = int(
        finest_smoother_up_steps
        if finest_smoother_up_steps is not None
        else fine_down_steps
    )

    for level_idx in range(1, len(hierarchy.levels)):
        level_space = hierarchy.levels[level_idx]
        if level_idx == finest_level_idx:
            configs = (
                (
                    "smoother_down",
                    pc.getMGSmootherDown(level_idx),
                    fine_down_ksp_type,
                    fine_down_pc_type,
                    fine_down_steps,
                    finest_smoother_down_pc_context or finest_smoother_pc_context,
                ),
                (
                    "smoother_up",
                    pc.getMGSmootherUp(level_idx),
                    fine_up_ksp_type,
                    fine_up_pc_type,
                    fine_up_steps,
                    finest_smoother_up_pc_context or finest_smoother_pc_context,
                ),
                (
                    "smoother",
                    pc.getMGSmoother(level_idx),
                    fine_down_ksp_type,
                    fine_down_pc_type,
                    fine_down_steps,
                    finest_smoother_pc_context or finest_smoother_down_pc_context,
                ),
            )
            for variant_name, level_ksp, ksp_type, pc_type, steps, pc_context in configs:
                _configure_level_ksp(
                    level_ksp,
                    ksp_type=ksp_type,
                    pc_type=pc_type,
                    steps=steps,
                    pc_context=pc_context,
                )
                _maybe_attach_observer(
                    level_ksp=level_ksp,
                    label=f"{variant_name}_level{level_idx}_deg{level_space.degree}",
                    level_index=int(level_idx),
                    mesh_level=int(level_space.level),
                    degree=int(level_space.degree),
                    family="fine",
                    sweep_role=str(variant_name),
                    entries=observer_entries,
                )
        else:
            level_pc_type = str(
                degree_pc_types.get(int(level_space.degree), intermediate_smoother_pc_type)
            )
            for variant_name, level_ksp in (
                ("smoother_down", pc.getMGSmootherDown(level_idx)),
                ("smoother_up", pc.getMGSmootherUp(level_idx)),
                ("smoother", pc.getMGSmoother(level_idx)),
            ):
                _configure_level_ksp(
                    level_ksp,
                    ksp_type="richardson",
                    pc_type=level_pc_type,
                    steps=int(intermediate_smoother_steps),
                )
                _maybe_attach_observer(
                    level_ksp=level_ksp,
                    label=f"{variant_name}_level{level_idx}_deg{level_space.degree}",
                    level_index=int(level_idx),
                    mesh_level=int(level_space.level),
                    degree=int(level_space.degree),
                    family="degree2" if int(level_space.degree) == 2 else "degree1",
                    sweep_role=str(variant_name),
                    entries=observer_entries,
                )

    if coarse_ksp_type is None:
        coarse_ksp_type = "cg"
    if coarse_pc_type is None:
        coarse_pc_type = "hypre"
    coarse = pc.getMGCoarseSolve()
    coarse.setType(str(coarse_ksp_type))
    coarse.setTolerances(rtol=1.0e-10, max_it=200)
    coarse.getPC().setType(str(coarse_pc_type))
    if str(coarse_pc_type) == "hypre":
        _apply_hypre_system_amg_settings(
            coarse,
            nodal_coarsen=int(coarse_hypre_nodal_coarsen),
            vec_interp_variant=int(coarse_hypre_vec_interp_variant),
            strong_threshold=coarse_hypre_strong_threshold,
            coarsen_type=coarse_hypre_coarsen_type,
            max_iter=int(coarse_hypre_max_iter),
            tol=float(coarse_hypre_tol),
            relax_type_all=coarse_hypre_relax_type_all,
            coordinates=_build_level_coordinates(hierarchy.levels[0]),
        )
    _maybe_attach_observer(
        level_ksp=coarse,
        label="coarse",
        level_index=0,
        mesh_level=int(hierarchy.levels[0].level),
        degree=int(hierarchy.levels[0].degree),
        family="coarse",
        sweep_role="coarse",
        entries=observer_entries,
    )
    return PMGObserverSuite(
        observer_entries,
        keepalive_mats=keepalive_transfer_mats,
        keepalive_objects=keepalive_objects,
    )


def update_explicit_pmg_operators(
    ksp: PETSc.KSP,
    hierarchy: SlopeStabilityMGHierarchy,
    level_operators: list[PETSc.Mat],
) -> None:
    if len(level_operators) != len(hierarchy.levels):
        raise ValueError(
            "Expected one operator per MG level; "
            f"received {len(level_operators)} for {len(hierarchy.levels)} levels"
        )
    pc = ksp.getPC()
    coarse = pc.getMGCoarseSolve()
    coarse.setOperators(level_operators[0])
    for level_idx in range(1, len(level_operators)):
        smoother = pc.getMGSmoother(level_idx)
        smoother_down = pc.getMGSmootherDown(level_idx)
        smoother_up = pc.getMGSmootherUp(level_idx)
        smoother.setOperators(level_operators[level_idx])
        smoother_down.setOperators(level_operators[level_idx])
        smoother_up.setOperators(level_operators[level_idx])
