"""Experimental single-step JAX+PETSc slope-stability solver."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.benchmark.state_export import export_planestrain_state_npz
from src.core.petsc.minimizers import newton
from src.core.petsc.reasons import ksp_reason_name
from src.core.petsc.trust_ksp import ksp_cg_set_radius
from src.problems.slope_stability.jax.mesh import MeshSlopeStability2D
from src.problems.slope_stability.jax_petsc.multigrid import (
    LegacyPMGLevelSmootherConfig,
    attach_pmg_level_metadata,
    build_mixed_pmg_hierarchy,
    build_pmg_hierarchy,
    configure_pmg,
    configure_explicit_pmg,
    mixed_hierarchy_specs,
    update_explicit_pmg_operators,
)
from src.problems.slope_stability.jax_petsc.reordered_element_assembler import (
    SlopeStabilityReorderedElementAssembler,
)
from src.problems.slope_stability.support import (
    DEFAULT_LEVEL,
    build_near_nullspace_modes,
    build_refined_p1_case_data,
    build_same_mesh_lagrange_case_data,
    case_name_for_level,
    davis_b_reduction,
    ensure_same_mesh_case_hdf5,
    load_same_mesh_case_hdf5_rank_local,
)


PROFILE_DEFAULTS = {
    "reference": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-3,
        "ksp_max_it": 200,
        "hypre_nodal_coarsen": -1,
        "hypre_vec_interp_variant": -1,
        "hypre_strong_threshold": None,
        "hypre_coarsen_type": "",
        "hypre_max_iter": 1,
        "hypre_tol": 0.0,
        "hypre_relax_type_all": "",
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
    "performance": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-2,
        "ksp_max_it": 50,
        "hypre_nodal_coarsen": -1,
        "hypre_vec_interp_variant": -1,
        "hypre_strong_threshold": None,
        "hypre_coarsen_type": "",
        "hypre_max_iter": 1,
        "hypre_tol": 0.0,
        "hypre_relax_type_all": "",
        "pc_setup_on_ksp_cap": False,
        "gamg_threshold": 0.05,
        "gamg_agg_nsmooths": 1,
        "use_near_nullspace": True,
        "gamg_set_coordinates": True,
        "reorder": False,
    },
}

SUCCESS_MESSAGE_PREFIXES = (
    "Converged",
    "Gradient norm converged",
    "Stopping condition for f is satisfied",
    "Stopping condition for step size is satisfied",
    "Trust-region step converged",
)


@dataclass(frozen=True)
class _LinearSolveFailure(RuntimeError):
    reason_code: int
    reason_name: str
    ksp_its: int
    true_residual_norm: float
    true_relative_residual: float

    def __str__(self) -> str:
        return (
            "Linear solve failed with "
            f"{self.reason_name} after {self.ksp_its} iterations "
            f"(true rel residual={self.true_relative_residual:.3e})"
        )


def _resolve_linear_settings(args):
    settings = dict(PROFILE_DEFAULTS[args.profile])
    overrides = {
        "ksp_type": args.ksp_type,
        "pc_type": args.pc_type,
        "ksp_rtol": args.ksp_rtol,
        "ksp_max_it": args.ksp_max_it,
        "hypre_nodal_coarsen": args.hypre_nodal_coarsen,
        "hypre_vec_interp_variant": args.hypre_vec_interp_variant,
        "hypre_strong_threshold": args.hypre_strong_threshold,
        "hypre_coarsen_type": args.hypre_coarsen_type,
        "hypre_max_iter": args.hypre_max_iter,
        "hypre_tol": args.hypre_tol,
        "hypre_relax_type_all": args.hypre_relax_type_all,
        "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
        "gamg_threshold": args.gamg_threshold,
        "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
        "use_near_nullspace": args.use_near_nullspace,
        "gamg_set_coordinates": args.gamg_set_coordinates,
        "reorder": args.reorder,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _pc_options(settings, *, operator_mode: str = "assembled"):
    opts = {}
    if settings["pc_type"] == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    if settings["pc_type"] == "mg":
        opts["pc_mg_galerkin"] = "none" if str(operator_mode) != "assembled" else "both"
    if settings["pc_type"] == "hypre":
        opts["pc_hypre_type"] = "boomeramg"
        if int(settings["hypre_nodal_coarsen"]) >= 0:
            opts["pc_hypre_boomeramg_nodal_coarsen"] = int(settings["hypre_nodal_coarsen"])
        if int(settings["hypre_vec_interp_variant"]) >= 0:
            opts["pc_hypre_boomeramg_vec_interp_variant"] = int(
                settings["hypre_vec_interp_variant"]
            )
        if settings["hypre_strong_threshold"] is not None:
            opts["pc_hypre_boomeramg_strong_threshold"] = float(
                settings["hypre_strong_threshold"]
            )
        if str(settings["hypre_coarsen_type"]):
            opts["pc_hypre_boomeramg_coarsen_type"] = str(settings["hypre_coarsen_type"])
        if int(settings["hypre_max_iter"]) >= 0:
            opts["pc_hypre_boomeramg_max_iter"] = int(settings["hypre_max_iter"])
        if settings["hypre_tol"] is not None:
            opts["pc_hypre_boomeramg_tol"] = float(settings["hypre_tol"])
        if str(settings["hypre_relax_type_all"]):
            opts["pc_hypre_boomeramg_relax_type_all"] = str(
                settings["hypre_relax_type_all"]
            )
    return opts


def _is_success_message(message: str) -> bool:
    return any(str(message).startswith(prefix) for prefix in SUCCESS_MESSAGE_PREFIXES)


def _gather_full_free_original(assembler, vec) -> np.ndarray:
    full_reordered, _ = assembler._allgather_full_owned(np.asarray(vec.array[:], dtype=np.float64))
    full_original = np.empty_like(full_reordered)
    full_original[np.asarray(assembler.layout.perm, dtype=np.int64)] = full_reordered
    return full_original


def _build_gamg_coordinates_partial_blocks(assembler, params: dict[str, object]) -> np.ndarray:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64).ravel()
    nodes = np.asarray(params["nodes"], dtype=np.float64)
    owned_orig_free = np.asarray(
        assembler.part.perm[assembler.part.lo : assembler.part.hi],
        dtype=np.int64,
    )
    owned_total_dofs = freedofs[owned_orig_free]
    node_ids = owned_total_dofs // 2
    return np.asarray(nodes[node_ids], dtype=np.float64)


def _reduced_material(params: dict[str, object], lambda_target: float) -> tuple[float, float]:
    return davis_b_reduction(
        float(params["c0"]),
        float(params["phi_deg"]),
        float(params["psi_deg"]),
        float(lambda_target),
    )


def _element_degree(params: dict[str, object], default: int = 2) -> int:
    degree = params.get("element_degree")
    if degree is not None:
        return int(degree)
    n_scalar = int(np.asarray(params["elems_scalar"]).shape[1])
    mapping = {3: 1, 6: 2, 15: 4}
    return int(mapping.get(n_scalar, default))


def _is_explicit_mg_strategy(params: dict[str, object], mg_strategy: str) -> bool:
    return int(_element_degree(params)) == 4 and str(mg_strategy) in {
        "same_mesh_p4_p2_p1",
        "same_mesh_p4_p2_p1_lminus1_p1",
    }


def _resolve_mg_variant(
    *,
    settings: dict[str, object],
    operator_mode: str,
    params: dict[str, object],
    mg_strategy: str,
    preconditioner_operator: str,
    mg_variant: str,
) -> str:
    if str(settings["pc_type"]) != "mg":
        return "none"
    variant = str(mg_variant)
    if variant == "auto":
        if (
            _is_explicit_mg_strategy(params, mg_strategy)
            and str(preconditioner_operator) == "same_operator"
            and str(operator_mode) != "assembled"
        ):
            return "explicit_pmg"
        return "legacy_pmg"
    if variant == "legacy_pmg":
        return variant
    if variant in {"explicit_pmg", "outer_pcksp"}:
        if not _is_explicit_mg_strategy(params, mg_strategy):
            raise ValueError(
                f"{variant} requires a same-mesh P4 mixed hierarchy strategy; got {mg_strategy!r}"
            )
        if str(preconditioner_operator) != "same_operator":
            raise ValueError(f"{variant} requires --preconditioner_operator same_operator")
        return variant
    raise ValueError(f"Unsupported MG variant {variant!r}")


def _mg_operator_policy_name(
    *,
    mg_variant: str,
    operator_mode: str,
    mg_lower_operator_policy: str,
) -> str:
    if mg_variant == "none":
        return "default"
    return (
        f"{mg_variant}_"
        f"{'assembled' if str(operator_mode) == 'assembled' else 'matrixfree'}_fine_"
        f"{str(mg_lower_operator_policy)}"
    )


def _build_explicit_mg_settings(args) -> dict[str, object]:
    coarse_backend = str(getattr(args, "mg_coarse_backend", None) or "hypre")
    coarse_pc_type = getattr(args, "mg_coarse_pc_type", None)
    if coarse_pc_type is None:
        if coarse_backend in {"redundant_lu", "redundant_hypre"}:
            coarse_pc_type = "redundant"
        elif coarse_backend in {"rank0_lu_broadcast", "rank0_hypre_broadcast"}:
            coarse_pc_type = "telescope"
        else:
            coarse_pc_type = coarse_backend
    if coarse_pc_type is None:
        coarse_pc_type = coarse_backend
    coarse_ksp_type = getattr(args, "mg_coarse_ksp_type", None)
    if coarse_ksp_type is None:
        if coarse_backend in {
            "redundant_lu",
            "redundant_hypre",
            "rank0_lu_broadcast",
            "rank0_hypre_broadcast",
        }:
            coarse_ksp_type = "preonly"
    if coarse_ksp_type is None:
        coarse_ksp_type = "cg"
    return {
        "fine_down": {
            "ksp_type": str(getattr(args, "mg_fine_down_ksp_type", None) or getattr(args, "mg_fine_ksp_type", "richardson")),
            "pc_type": str(getattr(args, "mg_fine_down_pc_type", None) or getattr(args, "mg_fine_pc_type", "none")),
            "python_pc_variant": str(getattr(args, "mg_fine_python_pc_variant", "none")),
            "steps": int(
                getattr(args, "mg_fine_down_steps", None)
                if getattr(args, "mg_fine_down_steps", None) is not None
                else getattr(args, "mg_fine_steps", 2)
            ),
        },
        "fine_up": {
            "ksp_type": str(getattr(args, "mg_fine_up_ksp_type", None) or getattr(args, "mg_fine_ksp_type", "richardson")),
            "pc_type": str(getattr(args, "mg_fine_up_pc_type", None) or getattr(args, "mg_fine_pc_type", "none")),
            "python_pc_variant": str(getattr(args, "mg_fine_python_pc_variant", "none")),
            "steps": int(
                getattr(args, "mg_fine_up_steps", None)
                if getattr(args, "mg_fine_up_steps", None) is not None
                else getattr(args, "mg_fine_steps", 2)
            ),
        },
        "intermediate_steps": int(getattr(args, "mg_intermediate_steps", 3)),
        "intermediate_pc_type": str(getattr(args, "mg_intermediate_pc_type", "jacobi")),
        "intermediate_degree_pc_types": {
            1: str(getattr(args, "mg_degree1_pc_type", None) or getattr(args, "mg_intermediate_pc_type", "jacobi")),
            2: str(getattr(args, "mg_degree2_pc_type", None) or getattr(args, "mg_intermediate_pc_type", "jacobi")),
        },
        "coarse_backend": str(coarse_backend),
        "coarse_ksp_type": str(coarse_ksp_type),
        "coarse_pc_type": str(coarse_pc_type),
        "coarse_hypre_nodal_coarsen": int(getattr(args, "mg_coarse_hypre_nodal_coarsen", 6)),
        "coarse_hypre_vec_interp_variant": int(getattr(args, "mg_coarse_hypre_vec_interp_variant", 3)),
        "coarse_hypre_strong_threshold": getattr(args, "mg_coarse_hypre_strong_threshold", None),
        "coarse_hypre_coarsen_type": str(getattr(args, "mg_coarse_hypre_coarsen_type", None) or ""),
        "coarse_hypre_max_iter": int(getattr(args, "mg_coarse_hypre_max_iter", 2)),
        "coarse_hypre_tol": float(getattr(args, "mg_coarse_hypre_tol", 0.0)),
        "coarse_hypre_relax_type_all": str(
            getattr(args, "mg_coarse_hypre_relax_type_all", None)
            or "symmetric-SOR/Jacobi"
        ),
        "outer_pcksp_inner_ksp_type": str(getattr(args, "outer_pcksp_inner_ksp_type", "fgmres")),
        "outer_pcksp_inner_ksp_rtol": float(getattr(args, "outer_pcksp_inner_ksp_rtol", 1.0e-2)),
        "outer_pcksp_inner_ksp_max_it": int(getattr(args, "outer_pcksp_inner_ksp_max_it", 20)),
        "python_pc_variant": str(getattr(args, "python_pc_variant", "none")),
    }


def _build_legacy_mg_settings(args) -> dict[str, object]:
    def _cfg(
        *,
        ksp_arg: str,
        pc_arg: str,
        steps_arg: str,
        default_pc: str = "sor",
        default_steps: int = 3,
    ) -> LegacyPMGLevelSmootherConfig:
        return LegacyPMGLevelSmootherConfig(
            ksp_type=str(getattr(args, ksp_arg, None) or "richardson"),
            pc_type=str(getattr(args, pc_arg, None) or default_pc),
            steps=int(getattr(args, steps_arg, None) or default_steps),
        )

    return {
        "fine": _cfg(
            ksp_arg="mg_p4_smoother_ksp_type",
            pc_arg="mg_p4_smoother_pc_type",
            steps_arg="mg_p4_smoother_steps",
            default_pc="sor",
            default_steps=3,
        ),
        "degree2": _cfg(
            ksp_arg="mg_p2_smoother_ksp_type",
            pc_arg="mg_p2_smoother_pc_type",
            steps_arg="mg_p2_smoother_steps",
            default_pc="sor",
            default_steps=3,
        ),
        "degree1": _cfg(
            ksp_arg="mg_p1_smoother_ksp_type",
            pc_arg="mg_p1_smoother_pc_type",
            steps_arg="mg_p1_smoother_steps",
            default_pc="sor",
            default_steps=3,
        ),
    }


def _summarize_linear_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "n_solves": 0,
            "n_converged": 0,
            "n_failed": 0,
            "n_accepted_via_maxit_direction": 0,
            "all_converged": True,
            "worst_true_relative_residual": 0.0,
            "reason_names": [],
        }
    reasons = [str(record.get("ksp_reason_name", "UNKNOWN")) for record in records]
    converged_flags = [bool(record.get("ksp_converged", False)) for record in records]
    accepted_via_maxit = [
        bool(record.get("ksp_accepted_via_maxit_direction", False)) for record in records
    ]
    return {
        "n_solves": int(len(records)),
        "n_converged": int(sum(int(flag) for flag in converged_flags)),
        "n_failed": int(sum(int(not flag) for flag in converged_flags)),
        "n_accepted_via_maxit_direction": int(
            sum(int(flag) for flag in accepted_via_maxit)
        ),
        "all_converged": bool(all(converged_flags)),
        "worst_true_relative_residual": float(
            max(float(record.get("true_relative_residual", 0.0)) for record in records)
        ),
        "reason_names": reasons,
        "last_reason_name": reasons[-1],
        "last_true_relative_residual": float(records[-1].get("true_relative_residual", 0.0)),
    }


def _build_mg_level_problem_data(
    *,
    level: int,
    degree: int,
    case_name: str | None,
    lambda_target: float,
    reg: float,
    build_mode: str,
    comm: MPI.Comm,
) -> tuple[dict[str, object], object]:
    params, adjacency, _ = _build_same_mesh_problem_data(
        level=int(level),
        degree=int(degree),
        case_name=case_name,
        build_mode=str(build_mode),
        comm=comm,
    )
    reduced_cohesion, reduced_phi_deg = _reduced_material(params, float(lambda_target))
    params["cohesion"] = float(reduced_cohesion)
    params["phi_deg"] = float(reduced_phi_deg)
    params["reg"] = float(reg)
    return params, adjacency


def _build_same_mesh_problem_data(
    *,
    level: int,
    degree: int,
    case_name: str | None,
    build_mode: str = "replicated",
    comm: MPI.Comm | None = None,
) -> tuple[dict[str, object], object, np.ndarray]:
    ensure_same_mesh_case_hdf5(int(level), int(degree))
    if str(build_mode) == "rank_local":
        if comm is None:
            raise ValueError("rank_local same-mesh problem build requires an MPI communicator")
        params = load_same_mesh_case_hdf5_rank_local(
            int(level),
            int(degree),
            reorder_mode="block_xyz",
            comm=comm,
            block_size=2,
        )
        u_init = np.zeros(int(np.asarray(params["freedofs"], dtype=np.int64).size), dtype=np.float64)
        return params, None, u_init
    case_data = build_same_mesh_lagrange_case_data(
        case_name_for_level(int(level)) if not case_name else str(case_name),
        degree=int(degree),
        build_mode=str(build_mode),
        comm=comm,
    )
    params = dict(case_data.__dict__)
    params["elastic_kernel"] = build_near_nullspace_modes(
        np.asarray(case_data.nodes, dtype=np.float64),
        np.asarray(case_data.freedofs, dtype=np.int64),
    )
    params["elem_type"] = f"P{int(degree)}"
    params["element_degree"] = int(degree)
    u_init = np.zeros(case_data.freedofs.size, dtype=np.float64)
    return params, case_data.adjacency, u_init


def _build_preconditioner_problem_data(
    *,
    params: dict[str, object],
    lambda_target: float,
    reg: float,
    preconditioner_operator: str,
):
    if str(preconditioner_operator) == "same_operator":
        return None
    if str(preconditioner_operator) != "refined_p1_same_nodes":
        raise ValueError(
            f"Unsupported preconditioner operator {preconditioner_operator!r}"
        )
    if str(params.get("elem_type", "P2")) != "P2":
        raise ValueError(
            "refined_p1_same_nodes preconditioner is currently supported only for the P2 operator"
        )

    case_data = build_refined_p1_case_data(case_name_for_level(int(params["level"])))
    pc_params = dict(case_data.__dict__)
    pc_params["elastic_kernel"] = build_near_nullspace_modes(
        np.asarray(case_data.nodes, dtype=np.float64),
        np.asarray(case_data.freedofs, dtype=np.int64),
    )
    reduced_cohesion, reduced_phi_deg = _reduced_material(pc_params, float(lambda_target))
    pc_params["cohesion"] = float(reduced_cohesion)
    pc_params["phi_deg"] = float(reduced_phi_deg)
    pc_params["reg"] = float(reg)
    pc_params["elem_type"] = "P1_refined_same_nodes"
    return pc_params, case_data.adjacency


def _is_frozen_fine_pmat_policy(policy: str) -> bool:
    return str(policy) in {"elastic_frozen", "initial_tangent_frozen"}


def _is_staggered_whole_pmat_policy(policy: str) -> bool:
    return str(policy) == "staggered_whole"


def _is_staggered_smoother_pmat_policy(policy: str) -> bool:
    return str(policy) == "staggered_smoother_only"


def run(args, problem_data=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    total_start = time.perf_counter()
    main_problem_build_time = 0.0
    preconditioner_problem_build_time = 0.0
    mg_hierarchy_build_time = 0.0
    mg_level_assembler_build_time = 0.0
    mg_configure_time = 0.0
    mg_hierarchy_metadata: dict[str, float | int] = {}

    if str(args.assembly_mode) != "element":
        raise ValueError("Slope-stability PETSc solver currently supports only --assembly_mode element")
    if not bool(args.local_coloring):
        raise ValueError("Slope-stability PETSc element mode requires --local_coloring")

    settings = _resolve_linear_settings(args)
    operator_mode = str(getattr(args, "operator_mode", "assembled"))
    pc_reuse_preconditioner = bool(getattr(args, "pc_reuse_preconditioner", False))
    pc_options = _pc_options(settings, operator_mode=operator_mode)
    preconditioner_operator = str(
        getattr(args, "preconditioner_operator", "same_operator")
    )
    mg_strategy = str(getattr(args, "mg_strategy", "legacy_p2_h"))
    mg_variant = str(getattr(args, "mg_variant", "auto"))
    fine_pmat_policy = str(getattr(args, "fine_pmat_policy", "same_operator"))
    mg_lower_operator_policy = str(
        getattr(args, "mg_lower_operator_policy", "refresh_each_newton")
    )
    explicit_mg_settings = _build_explicit_mg_settings(args)
    legacy_mg_settings = _build_legacy_mg_settings(args)
    if (
        str(explicit_mg_settings["fine_down"]["python_pc_variant"]) != "none"
        or str(explicit_mg_settings["fine_up"]["python_pc_variant"]) != "none"
    ) and (
        str(explicit_mg_settings["fine_down"]["pc_type"]) != "python"
        or str(explicit_mg_settings["fine_up"]["pc_type"]) != "python"
    ):
        raise ValueError(
            "Matrix-free fine Python PC variants require --mg_fine_pc_type python "
            "for both down and up smoothers"
        )
    if (
        str(explicit_mg_settings["python_pc_variant"]) != "none"
        and str(settings["pc_type"]) != "python"
    ):
        raise ValueError(
            "--python_pc_variant is only valid when --pc_type python"
        )
    if (
        str(explicit_mg_settings["python_pc_variant"]) != "none"
        or str(explicit_mg_settings["fine_down"]["python_pc_variant"]) != "none"
    ) and operator_mode == "assembled":
        raise ValueError(
            "Matrix-free Python PC variants require --operator_mode matfree_overlap "
            "or --operator_mode matfree_element"
        )

    if problem_data is None:
        t_pb0 = time.perf_counter()
        elem_degree = int(getattr(args, "elem_degree", 2))
        if elem_degree == 2:
            mesh_obj = MeshSlopeStability2D(
                level=int(getattr(args, "level", DEFAULT_LEVEL)),
                case=(args.case or None),
            )
            params, adjacency, u_init = mesh_obj.get_data()
            params = dict(params)
            params.setdefault("elem_type", "P2")
            params.setdefault("element_degree", 2)
        else:
            params, adjacency, u_init = _build_same_mesh_problem_data(
                level=int(getattr(args, "level", DEFAULT_LEVEL)),
                degree=elem_degree,
                case_name=(args.case or None),
                build_mode=str(getattr(args, "problem_build_mode", "root_bcast")),
                comm=comm,
            )
        main_problem_build_time = time.perf_counter() - t_pb0
    else:
        params, adjacency, u_init = problem_data
        params = dict(params)
        u_init = np.asarray(u_init, dtype=np.float64)
        main_problem_build_time = time.perf_counter() - total_start
    raw_phi_deg = float(params["phi_deg"])
    raw_c0 = float(params["c0"])
    reduced_cohesion, reduced_phi_deg = _reduced_material(params, float(args.lambda_target))
    params["cohesion"] = float(reduced_cohesion)
    params["phi_deg"] = float(reduced_phi_deg)
    params["reg"] = float(getattr(args, "reg", 1.0e-12))
    mg_variant = _resolve_mg_variant(
        settings=settings,
        operator_mode=operator_mode,
        params=params,
        mg_strategy=mg_strategy,
        preconditioner_operator=preconditioner_operator,
        mg_variant=mg_variant,
    )
    if (
        str(settings["pc_type"]) == "mg"
        and str(mg_strategy) == "same_mesh_p4_p2_p1"
        and str(explicit_mg_settings["coarse_pc_type"]) == "hypre"
        and int(explicit_mg_settings["coarse_hypre_vec_interp_variant"]) >= 0
    ):
        # This same-mesh P1 coarse solve is already block-structured, but the
        # BoomerAMG vector interpolation option can segfault in PETSc/Hypre for
        # this 2D hierarchy. Keep nodal coarsening and disable only that option.
        explicit_mg_settings["coarse_hypre_vec_interp_variant"] = -1
        explicit_mg_settings["coarse_hypre_vec_interp_variant_guarded"] = True
    else:
        explicit_mg_settings["coarse_hypre_vec_interp_variant_guarded"] = False
    frozen_fine_pmat = _is_frozen_fine_pmat_policy(fine_pmat_policy)
    staggered_whole_fine_pmat = _is_staggered_whole_pmat_policy(fine_pmat_policy)
    staggered_smoother_fine_pmat = _is_staggered_smoother_pmat_policy(fine_pmat_policy)
    fine_pmat_stagger_period = int(getattr(args, "fine_pmat_stagger_period", 2))
    if fine_pmat_stagger_period < 1:
        raise ValueError("--fine_pmat_stagger_period must be >= 1")
    if frozen_fine_pmat:
        if str(settings["pc_type"]) != "mg":
            raise ValueError("Frozen fine Pmat policies require --pc_type mg")
        if str(operator_mode) == "assembled":
            raise ValueError("Frozen fine Pmat policies require a matrix-free operator mode")
        if str(mg_variant) != "legacy_pmg":
            raise ValueError("Frozen fine Pmat policies require --mg_variant legacy_pmg")
        if str(preconditioner_operator) != "same_operator":
            raise ValueError("Frozen fine Pmat policies require --preconditioner_operator same_operator")
        if int(_element_degree(params)) != 4 or str(mg_strategy) != "same_mesh_p4_p2_p1":
            raise ValueError(
                "Frozen fine Pmat policies currently support only "
                "--elem_degree 4 with --mg_strategy same_mesh_p4_p2_p1"
            )
    if staggered_whole_fine_pmat:
        if str(settings["pc_type"]) != "mg":
            raise ValueError("staggered_whole fine Pmat policy requires --pc_type mg")
        if str(operator_mode) == "assembled":
            raise ValueError("staggered_whole fine Pmat policy requires a matrix-free operator mode")
        if str(mg_variant) != "legacy_pmg":
            raise ValueError("staggered_whole fine Pmat policy requires --mg_variant legacy_pmg")
        if str(preconditioner_operator) != "same_operator":
            raise ValueError("staggered_whole fine Pmat policy requires --preconditioner_operator same_operator")
        if int(_element_degree(params)) != 4 or str(mg_strategy) != "same_mesh_p4_p2_p1":
            raise ValueError(
                "staggered_whole fine Pmat policy currently supports only "
                "--elem_degree 4 with --mg_strategy same_mesh_p4_p2_p1"
            )
    if staggered_smoother_fine_pmat:
        if str(settings["pc_type"]) != "mg":
            raise ValueError("staggered_smoother_only fine Pmat policy requires --pc_type mg")
        if str(operator_mode) == "assembled":
            raise ValueError("staggered_smoother_only fine Pmat policy requires a matrix-free operator mode")
        if str(mg_variant) != "explicit_pmg":
            raise ValueError("staggered_smoother_only fine Pmat policy requires --mg_variant explicit_pmg")
        if str(preconditioner_operator) != "same_operator":
            raise ValueError("staggered_smoother_only fine Pmat policy requires --preconditioner_operator same_operator")
        if int(_element_degree(params)) != 4 or str(mg_strategy) != "same_mesh_p4_p2_p1":
            raise ValueError(
                "staggered_smoother_only fine Pmat policy currently supports only "
                "--elem_degree 4 with --mg_strategy same_mesh_p4_p2_p1"
            )
    uses_explicit_mg = mg_variant in {"explicit_pmg", "outer_pcksp"}
    if uses_explicit_mg and str(settings["ksp_type"]) not in {"fgmres", "gcr", "richardson"}:
        settings["ksp_type"] = "fgmres"
        if getattr(args, "ksp_type", None) is None:
            args.ksp_type = "fgmres"
    if mg_variant == "outer_pcksp":
        settings["ksp_type"] = "fgmres"
        if getattr(args, "ksp_type", None) is None:
            args.ksp_type = "fgmres"
    if (
        str(settings["pc_type"]) == "mg"
        and operator_mode != "assembled"
        and mg_variant == "legacy_pmg"
        and not frozen_fine_pmat
        and not staggered_whole_fine_pmat
    ):
        raise ValueError(
            "Matrix-free PCMG is currently implemented only for same-mesh P4->P2->P1 "
            "hierarchies with --preconditioner_operator same_operator and either "
            "an explicit MG variant or a frozen fine Pmat policy."
        )
    if str(settings["pc_type"]) == "mg" and uses_explicit_mg:
        pc_options["pc_mg_galerkin"] = "none"
    elif str(settings["pc_type"]) == "mg" and (frozen_fine_pmat or staggered_whole_fine_pmat):
        pc_options["pc_mg_galerkin"] = "both"

    setup_start = time.perf_counter()
    distribution_strategy = str(
        getattr(args, "distribution_strategy", "overlap_p2p")
    )
    reuse_hessian_value_buffers = bool(
        getattr(args, "reuse_hessian_value_buffers", True)
    )
    assembler = SlopeStabilityReorderedElementAssembler(
        params=params,
        comm=comm,
        adjacency=adjacency,
        ksp_rtol=float(settings["ksp_rtol"]),
        ksp_type=str(settings["ksp_type"]),
        pc_type=str(settings["pc_type"]),
        ksp_max_it=int(settings["ksp_max_it"]),
        use_near_nullspace=bool(settings["use_near_nullspace"]),
        pc_options=pc_options,
        reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
        local_hessian_mode=str(getattr(args, "local_hessian_mode", None) or "element"),
        distribution_strategy=distribution_strategy,
        reuse_hessian_value_buffers=reuse_hessian_value_buffers,
    )
    pc_assembler = None
    pc_params = None
    pc_adjacency = None
    t_pb0 = time.perf_counter()
    pc_problem_data = _build_preconditioner_problem_data(
        params=params,
        lambda_target=float(args.lambda_target),
        reg=float(params["reg"]),
        preconditioner_operator=preconditioner_operator,
    )
    preconditioner_problem_build_time = time.perf_counter() - t_pb0
    if pc_problem_data is not None:
        pc_params, pc_adjacency = pc_problem_data
        pc_assembler = SlopeStabilityReorderedElementAssembler(
            params=pc_params,
            comm=comm,
            adjacency=pc_adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            use_near_nullspace=bool(settings["use_near_nullspace"]),
            pc_options=pc_options,
            reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
            local_hessian_mode=str(getattr(args, "local_hessian_mode", None) or "element"),
            perm_override=np.asarray(assembler.layout.perm, dtype=np.int64),
            distribution_strategy=distribution_strategy,
            reuse_hessian_value_buffers=reuse_hessian_value_buffers,
        )
    assembler_setup_time = time.perf_counter() - setup_start
    problem_build_time = float(main_problem_build_time + preconditioner_problem_build_time)
    post_assembler_setup_start = time.perf_counter()

    u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    u_init_owned = np.asarray(
        u_init_reordered[assembler.layout.lo : assembler.layout.hi], dtype=np.float64
    )
    frozen_fine_pmat_mat = None
    fine_pmat_source = "same_operator"
    fine_pmat_setup_assembly_time = 0.0
    if frozen_fine_pmat:
        t_fpmat0 = time.perf_counter()
        if fine_pmat_policy == "elastic_frozen":
            assembler.assemble_hessian_with_mode(
                u_init_owned,
                constitutive_mode="elastic",
            )
            fine_pmat_source = "elastic_frozen_p4_matrix"
        elif fine_pmat_policy == "initial_tangent_frozen":
            assembler.assemble_hessian(u_init_owned)
            fine_pmat_source = "initial_tangent_frozen_p4_matrix"
        else:
            raise ValueError(f"Unsupported frozen fine Pmat policy {fine_pmat_policy!r}")
        fine_pmat_setup_assembly_time = time.perf_counter() - t_fpmat0
        frozen_fine_pmat_mat = assembler.A
    elif staggered_whole_fine_pmat:
        fine_pmat_source = "staggered_whole_p4_matrix"
    elif staggered_smoother_fine_pmat:
        fine_pmat_source = "staggered_smoother_p4_matrix"
    staggered_fine_pmat_mat = assembler.A if (staggered_whole_fine_pmat or staggered_smoother_fine_pmat) else None
    staggered_fine_pmat_last_update = 0
    x_initial = None
    x = None
    mg_hierarchy = None
    mg_level_assemblers = None
    mg_target_ksp = None
    mg_fixed_level_operators = None
    mg_level_operator_states = None
    mg_level_work_vecs = None
    mg_level_nullspaces: list[PETSc.NullSpace] = []
    mg_level_metadata: list[dict[str, object]] = []
    mg_runtime_observers = None
    mg_level_records_policy = str(mg_lower_operator_policy)
    x_initial = assembler.create_vec(u_init_reordered)
    x = assembler.create_vec(u_init_reordered)

    ksp = assembler.ksp
    A = assembler.A
    active_operator_mat = A
    pc = ksp.getPC()
    pc.setReusePreconditioner(bool(pc_reuse_preconditioner))
    top_python_pc_context = None
    mg_fine_python_pc_context = None
    if str(settings["pc_type"]) == "python" and str(explicit_mg_settings["python_pc_variant"]) != "none":
        top_python_pc_context = assembler.make_matrix_free_python_pc_context(
            str(explicit_mg_settings["python_pc_variant"])
        )
        pc.setPythonContext(top_python_pc_context)
    if str(settings["pc_type"]) == "mg" and (
        frozen_fine_pmat or staggered_whole_fine_pmat or staggered_smoother_fine_pmat
    ):
        pc.setUseAmat(False)
    if uses_explicit_mg and str(explicit_mg_settings["fine_down"]["pc_type"]) == "python":
        mg_fine_python_pc_context = assembler.make_matrix_free_python_pc_context(
            str(explicit_mg_settings["fine_down"]["python_pc_variant"])
        )
    if settings["pc_type"] == "mg":
        t_mg0 = time.perf_counter()
        if mg_strategy == "legacy_p2_h":
            mg_hierarchy = build_pmg_hierarchy(
                finest_level=int(params["level"]),
                coarsest_level=int(getattr(args, "mg_coarsest_level", 1)),
                finest_params=params,
                finest_adjacency=adjacency,
                finest_perm=np.asarray(assembler.layout.perm, dtype=np.int64),
                reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                comm=comm,
                level_build_mode=str(getattr(args, "mg_level_build_mode", "root_bcast")),
                transfer_build_mode=str(
                    getattr(args, "mg_transfer_build_mode", "root_bcast")
                ),
            )
        else:
            specs = mixed_hierarchy_specs(
                finest_level=int(params["level"]),
                finest_degree=_element_degree(params),
                strategy=mg_strategy,
                custom_hierarchy=getattr(args, "mg_custom_hierarchy", None),
            )
            mg_hierarchy = build_mixed_pmg_hierarchy(
                specs=specs,
                finest_params=params,
                finest_adjacency=adjacency,
                finest_perm=np.asarray(assembler.layout.perm, dtype=np.int64),
                reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                comm=comm,
                level_build_mode=str(getattr(args, "mg_level_build_mode", "root_bcast")),
                transfer_build_mode=str(
                    getattr(args, "mg_transfer_build_mode", "root_bcast")
                ),
            )
            if mg_hierarchy is None:
                raise ValueError("PCMG requires at least level 2")
        mg_hierarchy_build_time = time.perf_counter() - t_mg0
        mg_hierarchy_metadata = dict(getattr(mg_hierarchy, "build_metadata", {}) or {})
        t_mg_cfg0 = time.perf_counter()
        if mg_variant == "legacy_pmg":
            mg_runtime_observers = configure_pmg(
                ksp,
                mg_hierarchy,
                level_smoothers=dict(legacy_mg_settings),
                coarse_backend=str(explicit_mg_settings["coarse_backend"]),
                coarse_ksp_type=str(explicit_mg_settings["coarse_ksp_type"]),
                coarse_pc_type=str(explicit_mg_settings["coarse_pc_type"]),
                coarse_hypre_nodal_coarsen=int(
                    explicit_mg_settings["coarse_hypre_nodal_coarsen"]
                ),
                coarse_hypre_vec_interp_variant=int(
                    explicit_mg_settings["coarse_hypre_vec_interp_variant"]
                ),
                coarse_hypre_strong_threshold=explicit_mg_settings[
                    "coarse_hypre_strong_threshold"
                ],
                coarse_hypre_coarsen_type=str(
                    explicit_mg_settings["coarse_hypre_coarsen_type"]
                ),
                coarse_hypre_max_iter=int(explicit_mg_settings["coarse_hypre_max_iter"]),
                coarse_hypre_tol=float(explicit_mg_settings["coarse_hypre_tol"]),
                coarse_hypre_relax_type_all=str(
                    explicit_mg_settings["coarse_hypre_relax_type_all"]
                ),
            )
        else:
            if mg_variant == "outer_pcksp":
                pc.setType(PETSc.PC.Type.KSP)
                mg_target_ksp = pc.getKSP()
                mg_target_ksp.setType(str(explicit_mg_settings["outer_pcksp_inner_ksp_type"]))
                mg_target_ksp.setTolerances(
                    rtol=float(explicit_mg_settings["outer_pcksp_inner_ksp_rtol"]),
                    max_it=int(explicit_mg_settings["outer_pcksp_inner_ksp_max_it"]),
                )
                mg_target_ksp.getPC().setReusePreconditioner(bool(pc_reuse_preconditioner))
            else:
                mg_target_ksp = ksp
            mg_runtime_observers = configure_explicit_pmg(
                mg_target_ksp,
                mg_hierarchy,
                intermediate_smoother_steps=int(explicit_mg_settings["intermediate_steps"]),
                intermediate_smoother_pc_type=str(explicit_mg_settings["intermediate_pc_type"]),
                intermediate_degree_pc_types=dict(explicit_mg_settings["intermediate_degree_pc_types"]),
                finest_smoother_ksp_type=str(explicit_mg_settings["fine_down"]["ksp_type"]),
                finest_smoother_pc_type=str(explicit_mg_settings["fine_down"]["pc_type"]),
                finest_smoother_steps=int(explicit_mg_settings["fine_down"]["steps"]),
                finest_smoother_down_ksp_type=str(explicit_mg_settings["fine_down"]["ksp_type"]),
                finest_smoother_down_pc_type=str(explicit_mg_settings["fine_down"]["pc_type"]),
                finest_smoother_down_steps=int(explicit_mg_settings["fine_down"]["steps"]),
                finest_smoother_up_ksp_type=str(explicit_mg_settings["fine_up"]["ksp_type"]),
                finest_smoother_up_pc_type=str(explicit_mg_settings["fine_up"]["pc_type"]),
                finest_smoother_up_steps=int(explicit_mg_settings["fine_up"]["steps"]),
                finest_smoother_pc_context=mg_fine_python_pc_context,
                finest_smoother_down_pc_context=mg_fine_python_pc_context,
                finest_smoother_up_pc_context=mg_fine_python_pc_context,
                coarse_ksp_type=explicit_mg_settings["coarse_ksp_type"],
                coarse_pc_type=explicit_mg_settings["coarse_pc_type"],
                coarse_hypre_nodal_coarsen=int(
                    explicit_mg_settings["coarse_hypre_nodal_coarsen"]
                ),
                coarse_hypre_vec_interp_variant=int(
                    explicit_mg_settings["coarse_hypre_vec_interp_variant"]
                ),
                coarse_hypre_strong_threshold=explicit_mg_settings[
                    "coarse_hypre_strong_threshold"
                ],
                coarse_hypre_coarsen_type=str(
                    explicit_mg_settings["coarse_hypre_coarsen_type"]
                ),
                coarse_hypre_max_iter=int(explicit_mg_settings["coarse_hypre_max_iter"]),
                coarse_hypre_tol=float(explicit_mg_settings["coarse_hypre_tol"]),
                coarse_hypre_relax_type_all=str(
                    explicit_mg_settings["coarse_hypre_relax_type_all"]
                ),
            )
            mg_fixed_level_operators = [None] * len(mg_hierarchy.levels)
            mg_level_operator_states = [None] * len(mg_hierarchy.levels)
            if mg_lower_operator_policy == "galerkin_refresh":
                mg_level_assemblers = None
                mg_level_work_vecs = None
                mg_level_assembler_build_time = 0.0
            else:
                mg_level_assemblers = []
                mg_level_work_vecs = []
                t_lvl0 = time.perf_counter()
                for level_space in mg_hierarchy.levels[:-1]:
                    level_params, level_adjacency = _build_mg_level_problem_data(
                        level=int(level_space.level),
                        degree=int(level_space.degree),
                        case_name=(args.case or None),
                        lambda_target=float(args.lambda_target),
                        reg=float(params["reg"]),
                        build_mode=str(getattr(args, "mg_level_build_mode", "root_bcast")),
                        comm=comm,
                    )
                    level_assembler = SlopeStabilityReorderedElementAssembler(
                        params=level_params,
                        comm=comm,
                        adjacency=level_adjacency,
                        ksp_rtol=float(settings["ksp_rtol"]),
                        ksp_type="cg",
                        pc_type="none",
                        ksp_max_it=int(settings["ksp_max_it"]),
                        use_near_nullspace=bool(settings["use_near_nullspace"]),
                        pc_options=None,
                        reorder_mode=str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                        local_hessian_mode=str(getattr(args, "local_hessian_mode", None) or "element"),
                        perm_override=np.asarray(level_space.perm, dtype=np.int64),
                        block_size_override=int(level_space.ownership_block_size),
                        distribution_strategy=distribution_strategy,
                        reuse_hessian_value_buffers=reuse_hessian_value_buffers,
                    )
                    mg_level_assemblers.append(level_assembler)
                    mg_level_work_vecs.append(level_assembler.create_vec())
                mg_level_assembler_build_time = time.perf_counter() - t_lvl0
                if mg_lower_operator_policy == "fixed_setup":
                    for level_idx, level_assembler in enumerate(mg_level_assemblers):
                        zero_owned = np.zeros(level_assembler.part.n_owned, dtype=np.float64)
                        level_assembler.assemble_hessian(zero_owned)
                        mg_fixed_level_operators[level_idx] = level_assembler.A
                        mg_level_operator_states[level_idx] = "fixed_setup"
        mg_configure_time = time.perf_counter() - t_mg_cfg0

    gamg_coords = None
    if settings["pc_type"] == "gamg" and settings["gamg_set_coordinates"]:
        gamg_coords = _build_gamg_coordinates_partial_blocks(assembler, params)

    use_trust_region = bool(getattr(args, "use_trust_region", False))
    trust_radius_init = float(getattr(args, "trust_radius_init", 1.0))
    trust_radius_min = float(getattr(args, "trust_radius_min", 1.0e-8))
    trust_radius_max = float(getattr(args, "trust_radius_max", 1.0e6))
    trust_shrink = float(getattr(args, "trust_shrink", 0.5))
    trust_expand = float(getattr(args, "trust_expand", 1.5))
    trust_eta_shrink = float(getattr(args, "trust_eta_shrink", 0.05))
    trust_eta_expand = float(getattr(args, "trust_eta_expand", 0.75))
    trust_max_reject = int(getattr(args, "trust_max_reject", 6))
    trust_subproblem_line_search = bool(
        getattr(args, "trust_subproblem_line_search", False)
    )
    step_time_limit_s = getattr(args, "step_time_limit_s", None)
    line_search = str(getattr(args, "line_search", "golden_fixed"))
    benchmark_mode = str(getattr(args, "benchmark_mode", "end_to_end"))
    linesearch_interval = (float(args.linesearch_a), float(args.linesearch_b))
    trust_ksp_subproblem = bool(
        use_trust_region and str(settings["ksp_type"]).lower() in {"stcg", "nash", "gltr"}
    )

    linear_timing_records: list[dict[str, object]] = []
    linear_iters_this_attempt: list[int] = []
    force_pc_setup_next = True
    used_ksp_rtol = float(settings["ksp_rtol"])
    used_ksp_max_it = int(settings["ksp_max_it"])
    ksp_accept_true_rel = getattr(args, "ksp_accept_true_rel", None)
    residual_ax = x.duplicate()
    residual_vec = x.duplicate()

    def _assemble_and_solve(vec, rhs, sol, trust_radius=None):
        nonlocal force_pc_setup_next, gamg_coords, active_operator_mat, staggered_fine_pmat_last_update, mg_level_nullspaces, mg_level_metadata

        operator_mat = assembler.A
        operator_prepare_total_time = 0.0
        operator_prepare_details = {}
        fine_operator_source = "assembled_matrix"
        step_fine_pmat_source = str(fine_pmat_source)
        fine_pmat_updated_this_step = False
        linear_step_index = int(len(linear_timing_records) + 1)
        should_update_staggered_fine_pmat = (
            int(((linear_step_index - 1) % fine_pmat_stagger_period) == 0)
            if (staggered_whole_fine_pmat or staggered_smoother_fine_pmat)
            else False
        )

        if operator_mode == "assembled":
            t_asm0 = time.perf_counter()
            assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
            asm_total_time = time.perf_counter() - t_asm0
            asm_details = dict(assembler.iter_timings[-1]) if assembler.iter_timings else {}
            asm_details["assembly_total_time"] = float(asm_total_time)
        else:
            operator_prepare_start = time.perf_counter()
            operator_mat = assembler.prepare_matrix_free_operator(
                np.asarray(vec.array[:], dtype=np.float64),
                mode=operator_mode,
            )
            operator_prepare_total_time = time.perf_counter() - operator_prepare_start
            operator_prepare_details = assembler.matrix_free_summary()
            fine_operator_source = str(operator_prepare_details.get("mode", operator_mode))
            asm_total_time = 0.0
            asm_details = {
                "assembly_mode": operator_mode,
                "hvp_compute": 0.0,
                "extraction": 0.0,
                "coo_assembly": 0.0,
                "p2p_exchange": 0.0,
                "n_hvps": 0,
                "assembly_total_time": 0.0,
            }

        pc_asm_total_time = 0.0
        pc_asm_details = {}
        pmat = None
        pmg_level_operators = None
        fine_pmat_step_assembly_time = 0.0
        if mg_variant in {"explicit_pmg", "outer_pcksp"}:
            if (
                mg_hierarchy is None
                or mg_fixed_level_operators is None
                or mg_level_operator_states is None
            ):
                raise RuntimeError("Explicit MG requested without a configured hierarchy")
            level_operators = [None] * len(mg_hierarchy.levels)
            if staggered_smoother_fine_pmat:
                if staggered_fine_pmat_mat is None:
                    raise RuntimeError("staggered fine smoother policy is missing the reusable Pmat")
                if should_update_staggered_fine_pmat:
                    t_fpmat0 = time.perf_counter()
                    assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
                    fine_pmat_step_assembly_time = time.perf_counter() - t_fpmat0
                    staggered_fine_pmat_last_update = int(linear_step_index)
                    fine_pmat_updated_this_step = True
                    step_fine_pmat_source = "staggered_smoother_updated"
                else:
                    step_fine_pmat_source = "staggered_smoother_reused"
                level_operators[-1] = staggered_fine_pmat_mat
            else:
                level_operators[-1] = operator_mat
            mg_level_records = []
            restriction_transfer_time = 0.0

            if mg_lower_operator_policy == "fixed_setup":
                for level_idx, level_assembler in enumerate(mg_level_assemblers):
                    level_operators[level_idx] = mg_fixed_level_operators[level_idx]
                    mg_level_records.append(
                        {
                            "level_index": int(level_idx),
                            "mesh_level": int(mg_hierarchy.levels[level_idx].level),
                            "degree": int(mg_hierarchy.levels[level_idx].degree),
                            "assembly_total_time": 0.0,
                            "operator_source": str(mg_level_operator_states[level_idx] or "fixed_setup"),
                        }
                    )
            elif mg_lower_operator_policy == "galerkin_refresh":
                current_operator = level_operators[-1]
                for fine_level_idx in range(len(mg_hierarchy.levels) - 1, 0, -1):
                    coarse_level_idx = fine_level_idx - 1
                    prolong = mg_hierarchy.prolongations[coarse_level_idx]
                    t_level0 = time.perf_counter()
                    coarse_operator = current_operator.ptap(
                        prolong,
                        result=mg_fixed_level_operators[coarse_level_idx],
                    )
                    level_assembly_time = time.perf_counter() - t_level0
                    mg_fixed_level_operators[coarse_level_idx] = coarse_operator
                    level_operators[coarse_level_idx] = coarse_operator
                    mg_level_operator_states[coarse_level_idx] = "galerkin_refresh"
                    coarse_space = mg_hierarchy.levels[coarse_level_idx]
                    mg_level_records.append(
                        {
                            "level_index": int(coarse_level_idx),
                            "mesh_level": int(coarse_space.level),
                            "degree": int(coarse_space.degree),
                            "assembly_total_time": float(level_assembly_time),
                            "operator_source": "galerkin_refresh",
                        }
                    )
                    current_operator = coarse_operator
                mg_level_records.reverse()
                pc_asm_total_time = float(
                    sum(float(entry["assembly_total_time"]) for entry in mg_level_records)
                )
            else:
                if mg_level_assemblers is None or mg_level_work_vecs is None:
                    raise RuntimeError(
                        f"Explicit MG operator policy {mg_lower_operator_policy!r} requires level assemblers"
                    )
                t_restrict0 = time.perf_counter()
                current_level_vec = vec
                for restrict_idx in range(len(mg_hierarchy.restrictions) - 1, -1, -1):
                    coarse_vec = mg_level_work_vecs[restrict_idx]
                    mg_hierarchy.restrictions[restrict_idx].mult(current_level_vec, coarse_vec)
                    current_level_vec = coarse_vec
                restriction_transfer_time = time.perf_counter() - t_restrict0
                for level_idx, level_assembler in enumerate(mg_level_assemblers):
                    level_owned = np.asarray(
                        mg_level_work_vecs[level_idx].array[:],
                        dtype=np.float64,
                    )
                    t_level0 = time.perf_counter()
                    level_assembler.assemble_hessian(level_owned)
                    level_assembly_time = time.perf_counter() - t_level0
                    level_operators[level_idx] = level_assembler.A
                    mg_level_operator_states[level_idx] = "refresh_each_newton"
                    mg_level_records.append(
                        {
                            "level_index": int(level_idx),
                            "mesh_level": int(mg_hierarchy.levels[level_idx].level),
                            "degree": int(mg_hierarchy.levels[level_idx].degree),
                            "assembly_total_time": float(level_assembly_time),
                            "operator_source": "refresh_each_newton",
                        }
                    )
                pc_asm_total_time = float(
                    sum(float(entry["assembly_total_time"]) for entry in mg_level_records)
                )

            pc_asm_details = {
                "assembly_total_time": float(pc_asm_total_time + fine_pmat_step_assembly_time),
                "restriction_transfer_time": float(restriction_transfer_time),
                "mg_level_records": mg_level_records,
            }
            pmg_level_operators = level_operators
            pmat = level_operators[-1] if staggered_smoother_fine_pmat else operator_mat
        elif staggered_whole_fine_pmat and staggered_fine_pmat_mat is not None:
            if should_update_staggered_fine_pmat:
                t_fpmat0 = time.perf_counter()
                assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
                fine_pmat_step_assembly_time = time.perf_counter() - t_fpmat0
                staggered_fine_pmat_last_update = int(linear_step_index)
                fine_pmat_updated_this_step = True
                step_fine_pmat_source = "staggered_whole_updated"
            else:
                step_fine_pmat_source = "staggered_whole_reused"
            pmat = staggered_fine_pmat_mat
            pc_asm_details = {
                "assembly_total_time": float(fine_pmat_step_assembly_time),
                "restriction_transfer_time": 0.0,
                "mg_level_records": [],
            }
        elif frozen_fine_pmat and frozen_fine_pmat_mat is not None:
            pmat = frozen_fine_pmat_mat
            pc_asm_details = {
                "assembly_total_time": 0.0,
                "restriction_transfer_time": 0.0,
                "mg_level_records": [],
            }
        elif pc_assembler is not None:
            t_pasm0 = time.perf_counter()
            pc_assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
            pc_asm_total_time = time.perf_counter() - t_pasm0
            pc_asm_details = (
                dict(pc_assembler.iter_timings[-1]) if pc_assembler.iter_timings else {}
            )
            pc_asm_details["assembly_total_time"] = float(pc_asm_total_time)
            pmat = pc_assembler.A
        elif str(settings["pc_type"]) == "python":
            pmat = None
        elif operator_mode != "assembled":
            t_pasm0 = time.perf_counter()
            assembler.assemble_hessian(np.asarray(vec.array[:], dtype=np.float64))
            pc_asm_total_time = time.perf_counter() - t_pasm0
            fine_pmat_step_assembly_time = float(pc_asm_total_time)
            pc_asm_details = (
                dict(assembler.iter_timings[-1]) if assembler.iter_timings else {}
            )
            pc_asm_details["assembly_total_time"] = float(pc_asm_total_time)
            pmat = assembler.A

        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))

        t_setop0 = time.perf_counter()
        if mg_variant == "outer_pcksp":
            ksp.setOperators(operator_mat)
            if mg_target_ksp is None:
                raise RuntimeError("outer_pcksp requested without an inner KSP")
            mg_target_ksp.setOperators(operator_mat, pmat if pmat is not None else operator_mat)
        else:
            if pmat is None:
                ksp.setOperators(operator_mat)
            else:
                ksp.setOperators(operator_mat, pmat)
        if pmg_level_operators is not None:
            update_explicit_pmg_operators(
                mg_target_ksp if mg_variant == "outer_pcksp" else ksp,
                mg_hierarchy,
                pmg_level_operators,
            )
        active_operator_mat = operator_mat
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        t_setop = time.perf_counter() - t_setop0

        t_tol0 = time.perf_counter()
        ksp.setTolerances(rtol=float(used_ksp_rtol), max_it=int(used_ksp_max_it))
        t_tol = time.perf_counter() - t_tol0

        t_setup0 = time.perf_counter()
        if settings["pc_setup_on_ksp_cap"]:
            if force_pc_setup_next:
                if mg_variant == "outer_pcksp" and mg_target_ksp is not None:
                    mg_target_ksp.setUp()
                ksp.setUp()
                force_pc_setup_next = False
        else:
            if mg_variant == "outer_pcksp" and mg_target_ksp is not None:
                mg_target_ksp.setUp()
            ksp.setUp()
        t_setup = time.perf_counter() - t_setup0
        if mg_hierarchy is not None:
            mg_metadata_bundle = attach_pmg_level_metadata(
                mg_target_ksp if mg_variant == "outer_pcksp" and mg_target_ksp is not None else ksp,
                mg_hierarchy,
                use_near_nullspace=bool(settings["use_near_nullspace"]),
                block_size=2,
                coarse_pc_type=str(explicit_mg_settings["coarse_pc_type"]),
                coarse_hypre_nodal_coarsen=int(
                    explicit_mg_settings["coarse_hypre_nodal_coarsen"]
                ),
                coarse_hypre_vec_interp_variant=int(
                    explicit_mg_settings["coarse_hypre_vec_interp_variant"]
                ),
                coarse_hypre_strong_threshold=explicit_mg_settings[
                    "coarse_hypre_strong_threshold"
                ],
                coarse_hypre_coarsen_type=str(
                    explicit_mg_settings["coarse_hypre_coarsen_type"]
                ),
                coarse_hypre_max_iter=int(explicit_mg_settings["coarse_hypre_max_iter"]),
                coarse_hypre_tol=float(explicit_mg_settings["coarse_hypre_tol"]),
                coarse_hypre_relax_type_all=str(
                    explicit_mg_settings["coarse_hypre_relax_type_all"]
                ),
            )
            mg_level_nullspaces = list(mg_metadata_bundle.get("nullspaces", []))
            mg_level_metadata = list(mg_metadata_bundle.get("levels", []))
            if mg_runtime_observers is not None:
                mg_runtime_observers.reset()

        t_solve0 = time.perf_counter()
        ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t_solve0
        ksp_its = int(ksp.getIterationNumber())
        linear_iters_this_attempt.append(ksp_its)
        operator_record = (
            assembler.matrix_free_summary() if operator_mode != "assembled" else {}
        )
        reason_code = int(ksp.getConvergedReason())
        reason_name = ksp_reason_name(reason_code)
        rhs_norm = float(rhs.norm(PETSc.NormType.NORM_2))
        operator_mat.mult(sol, residual_ax)
        rhs.copy(residual_vec)
        residual_vec.axpy(-1.0, residual_ax)
        true_residual_norm = float(residual_vec.norm(PETSc.NormType.NORM_2))
        true_relative_residual = true_residual_norm / max(rhs_norm, 1.0e-16)
        directional_derivative = float(-rhs.dot(sol))
        accepted_via_true_residual = bool(
            reason_code <= 0
            and ksp_accept_true_rel is not None
            and np.isfinite(float(true_relative_residual))
            and float(true_relative_residual) <= float(ksp_accept_true_rel)
        )
        maxit_direction_true_rel_cap = float(
            getattr(args, "ksp_maxit_direction_true_rel_cap", 6.0e-2)
        )
        guard_ksp_maxit_direction = bool(
            getattr(args, "guard_ksp_maxit_direction", False)
        )
        accepted_via_maxit_direction = bool(
            reason_code <= 0
            and bool(getattr(args, "accept_ksp_maxit_direction", True))
            and str(reason_name) == "DIVERGED_MAX_IT"
            and np.isfinite(float(true_relative_residual))
            and (
                not guard_ksp_maxit_direction
                or (
                    float(true_relative_residual) <= maxit_direction_true_rel_cap
                    and np.isfinite(float(directional_derivative))
                    and float(directional_derivative) < 0.0
                )
            )
        )
        mg_runtime_diagnostics = (
            list(mg_runtime_observers.snapshot()) if mg_runtime_observers is not None else []
        )

        if settings["pc_setup_on_ksp_cap"] and ksp_its >= int(used_ksp_max_it):
            force_pc_setup_next = True

        record = {
            "assemble_total_time": float(asm_total_time),
            "assemble_p2p_exchange": float(asm_details.get("p2p_exchange", 0.0)),
            "assemble_hvp_compute": float(asm_details.get("hvp_compute", 0.0)),
            "assemble_extraction": float(asm_details.get("extraction", 0.0)),
            "assemble_coo_assembly": float(asm_details.get("coo_assembly", 0.0)),
            "assemble_n_hvps": int(asm_details.get("n_hvps", 0)),
            "operator_mode": str(operator_mode),
            "operator_source": fine_operator_source,
            "operator_prepare_total_time": float(operator_prepare_total_time),
            "operator_prepare_allgatherv": float(
                operator_prepare_details.get("prepare_allgatherv", 0.0)
            ),
            "operator_prepare_ghost_exchange": float(
                operator_prepare_details.get("prepare_ghost_exchange", 0.0)
            ),
            "operator_prepare_build_v_local": float(
                operator_prepare_details.get("prepare_build_v_local", 0.0)
            ),
            "operator_prepare_linearize": float(
                operator_prepare_details.get("prepare_linearize", 0.0)
            ),
            "operator_diagonal_source": str(
                operator_record.get(
                    "diagonal_source",
                    operator_prepare_details.get("diagonal_source", "not_requested"),
                )
            ),
            "operator_diagonal_prepare_total": float(
                operator_record.get(
                    "diagonal_prepare_total",
                    operator_prepare_details.get("diagonal_prepare_total", 0.0),
                )
            ),
            "fine_pmat_policy": str(fine_pmat_policy),
            "fine_pmat_source": str(step_fine_pmat_source),
            "fine_pmat_setup_assembly_time": float(fine_pmat_setup_assembly_time),
            "fine_pmat_step_assembly_time": float(fine_pmat_step_assembly_time),
            "fine_pmat_updated_this_step": bool(fine_pmat_updated_this_step),
            "fine_pmat_lag_steps": int(max(0, linear_step_index - max(1, staggered_fine_pmat_last_update)))
            if (staggered_whole_fine_pmat or staggered_smoother_fine_pmat)
            else 0,
            "operator_apply_calls": int(operator_record.get("mult_calls", 0)),
            "operator_apply_total_time": float(operator_record.get("mult_total", 0.0)),
            "operator_apply_allgatherv": float(
                operator_record.get("mult_allgatherv", 0.0)
            ),
            "operator_apply_ghost_exchange": float(
                operator_record.get("mult_ghost_exchange", 0.0)
            ),
            "operator_apply_build_v_local": float(
                operator_record.get("mult_build_v_local", 0.0)
            ),
            "operator_apply_kernel": float(operator_record.get("mult_apply", 0.0)),
            "operator_apply_scatter": float(operator_record.get("mult_scatter", 0.0)),
            "python_pc_variant": str(
                operator_record.get(
                    "python_pc_variant",
                    explicit_mg_settings.get("python_pc_variant", "none"),
                )
            ),
            "python_pc_prepare_total_time": float(
                operator_record.get("python_pc_prepare_total", 0.0)
            ),
            "python_pc_apply_calls": int(
                operator_record.get("python_pc_apply_calls", 0)
            ),
            "python_pc_apply_total_time": float(
                operator_record.get("python_pc_apply_total", 0.0)
            ),
            "pc_operator": str(preconditioner_operator),
            "pc_operator_assemble_total_time": float(pc_asm_total_time),
            "pc_operator_assemble_p2p_exchange": float(
                pc_asm_details.get("p2p_exchange", 0.0)
            ),
            "pc_operator_assemble_hvp_compute": float(
                pc_asm_details.get("hvp_compute", 0.0)
            ),
            "pc_operator_assemble_extraction": float(
                pc_asm_details.get("extraction", 0.0)
            ),
            "pc_operator_assemble_coo_assembly": float(
                pc_asm_details.get("coo_assembly", 0.0)
            ),
            "pc_operator_assemble_n_hvps": int(pc_asm_details.get("n_hvps", 0)),
            "pc_operator_state_transfer_time": float(
                pc_asm_details.get("restriction_transfer_time", 0.0)
            ),
            "pc_operator_mg_level_records": list(pc_asm_details.get("mg_level_records", [])),
            "mg_runtime_diagnostics": mg_runtime_diagnostics,
            "mg_variant": str(mg_variant),
            "mg_lower_operator_policy": str(mg_lower_operator_policy),
            "mg_legacy_level_smoothers": {
                "fine": {
                    "ksp_type": str(legacy_mg_settings["fine"].ksp_type),
                    "pc_type": str(legacy_mg_settings["fine"].pc_type),
                    "steps": int(legacy_mg_settings["fine"].steps),
                },
                "degree2": {
                    "ksp_type": str(legacy_mg_settings["degree2"].ksp_type),
                    "pc_type": str(legacy_mg_settings["degree2"].pc_type),
                    "steps": int(legacy_mg_settings["degree2"].steps),
                },
                "degree1": {
                    "ksp_type": str(legacy_mg_settings["degree1"].ksp_type),
                    "pc_type": str(legacy_mg_settings["degree1"].pc_type),
                    "steps": int(legacy_mg_settings["degree1"].steps),
                },
            },
            "mg_fine_down_ksp_type": str(explicit_mg_settings["fine_down"]["ksp_type"]),
            "mg_fine_down_pc_type": str(explicit_mg_settings["fine_down"]["pc_type"]),
            "mg_fine_down_steps": int(explicit_mg_settings["fine_down"]["steps"]),
            "mg_fine_up_ksp_type": str(explicit_mg_settings["fine_up"]["ksp_type"]),
            "mg_fine_up_pc_type": str(explicit_mg_settings["fine_up"]["pc_type"]),
            "mg_fine_up_steps": int(explicit_mg_settings["fine_up"]["steps"]),
            "setop_time": float(t_setop),
            "set_tolerances_time": float(t_tol),
            "pc_setup_time": float(t_setup),
            "solve_time": float(t_solve),
            "linear_total_time": float(
                asm_total_time
                + operator_prepare_total_time
                + pc_asm_total_time
                + t_setop
                + t_tol
                + t_setup
                + t_solve
            ),
            "ksp_its": int(ksp_its),
            "ksp_reason_code": int(reason_code),
            "ksp_reason_name": str(reason_name),
            "ksp_converged": bool(
                reason_code > 0 or accepted_via_true_residual or accepted_via_maxit_direction
            ),
            "ksp_accepted_via_true_residual": bool(accepted_via_true_residual),
            "ksp_accepted_via_maxit_direction": bool(accepted_via_maxit_direction),
            "ksp_residual_norm": float(ksp.getResidualNorm()),
            "rhs_norm": float(rhs_norm),
            "true_residual_norm": float(true_residual_norm),
            "true_relative_residual": float(true_relative_residual),
            "directional_derivative": float(directional_derivative),
            "guard_ksp_maxit_direction": bool(guard_ksp_maxit_direction),
            "ksp_maxit_direction_true_rel_cap": float(maxit_direction_true_rel_cap),
        }
        if trust_radius is not None:
            record["trust_radius"] = float(trust_radius)
        linear_timing_records.append(record)

        if reason_code <= 0 and not accepted_via_true_residual and not accepted_via_maxit_direction:
            raise _LinearSolveFailure(
                reason_code=int(reason_code),
                reason_name=str(reason_name),
                ksp_its=int(ksp_its),
                true_residual_norm=float(true_residual_norm),
                true_relative_residual=float(true_relative_residual),
            )

        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=None)

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(vec, rhs, sol, trust_radius=float(trust_radius))

    solve_phase_start = time.perf_counter()
    solver_bootstrap_time = float(solve_phase_start - post_assembler_setup_start)
    one_time_setup_time = float(problem_build_time + assembler_setup_time + solver_bootstrap_time)

    attempts = build_retry_attempts(
        retry_on_failure=bool(args.retry_on_failure),
        linesearch_interval=linesearch_interval,
        ksp_rtol=float(settings["ksp_rtol"]),
        ksp_max_it=int(settings["ksp_max_it"]),
    )

    result = None
    solve_time = 0.0
    used_attempt = "primary"
    used_linesearch = linesearch_interval
    linear_failure_message = ""

    payload = None
    try:
        for attempt_name, attempt_ls, attempt_rtol, attempt_ksp_max_it in attempts:
            x_initial.copy(x)
            force_pc_setup_next = True
            linear_timing_records.clear()
            linear_iters_this_attempt.clear()
            used_ksp_rtol = float(attempt_rtol)
            used_ksp_max_it = int(attempt_ksp_max_it)
            used_attempt = str(attempt_name)
            used_linesearch = (float(attempt_ls[0]), float(attempt_ls[1]))

            solve_start = time.perf_counter()
            linear_failure_message = ""
            try:
                result = newton(
                    energy_fn=assembler.energy_fn,
                    gradient_fn=assembler.gradient_fn,
                    hessian_solve_fn=hessian_solve_fn,
                    x=x,
                    tolf=float(args.tolf),
                    tolg=float(args.tolg),
                    tolg_rel=float(args.tolg_rel),
                    linesearch_tol=float(args.linesearch_tol),
                    linesearch_interval=used_linesearch,
                    line_search=line_search,
                    armijo_alpha0=float(getattr(args, "armijo_alpha0", 1.0)),
                    armijo_c1=float(getattr(args, "armijo_c1", 1.0e-4)),
                    armijo_shrink=float(getattr(args, "armijo_shrink", 0.5)),
                    armijo_max_ls=int(getattr(args, "armijo_max_ls", 40)),
                    maxit=int(args.maxit),
                    tolx_rel=float(args.tolx_rel),
                    tolx_abs=float(args.tolx_abs),
                    require_all_convergence=True,
                    fail_on_nonfinite=True,
                    verbose=(not args.quiet),
                    comm=comm,
                    hessian_matvec_fn=lambda _x, vin, vout: active_operator_mat.mult(vin, vout),
                    trust_subproblem_solve_fn=(
                        trust_subproblem_solve_fn if trust_ksp_subproblem else None
                    ),
                    trust_subproblem_line_search=trust_subproblem_line_search,
                    save_history=bool(args.save_history),
                    trust_region=use_trust_region,
                    trust_radius_init=trust_radius_init,
                    trust_radius_min=trust_radius_min,
                    trust_radius_max=trust_radius_max,
                    trust_shrink=trust_shrink,
                    trust_expand=trust_expand,
                    trust_eta_shrink=trust_eta_shrink,
                    trust_eta_expand=trust_eta_expand,
                    trust_max_reject=trust_max_reject,
                    step_time_limit_s=step_time_limit_s,
                )
            except _LinearSolveFailure as exc:
                linear_failure_message = str(exc)
                result = {
                    "nit": int(len(linear_timing_records)),
                    "fun": float(assembler.energy_fn(x)),
                    "message": linear_failure_message,
                    "history": [],
                }
            solve_time += time.perf_counter() - solve_start
            if not needs_solver_repair(
                result,
                retry_on_nonfinite=bool(args.retry_on_failure),
                retry_on_maxit=bool(args.retry_on_failure),
            ):
                break
        assert result is not None
        finalize_start = time.perf_counter()
        full_free_original = _gather_full_free_original(assembler, x)
        u_full = np.asarray(params["u_0"], dtype=np.float64).copy()
        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        u_full[freedofs] = full_free_original
        coords_ref = np.asarray(params["nodes"], dtype=np.float64)
        coords_final = coords_ref + u_full.reshape((-1, 2))
        displacement = coords_final - coords_ref
        u_max = float(np.max(np.linalg.norm(displacement, axis=1)))
        omega = float(np.dot(np.asarray(params["force"], dtype=np.float64), u_full))
        total_linear_iters = int(sum(int(v) for v in linear_iters_this_attempt))
        linear_summary = _summarize_linear_records(linear_timing_records)
        callback_summary = assembler.callback_summary()
        final_grad_vec = x.duplicate()
        try:
            assembler.gradient_fn(x, final_grad_vec)
            final_grad_norm = float(final_grad_vec.norm(PETSc.NormType.NORM_2))
        finally:
            final_grad_vec.destroy()
        message = str(result["message"])
        solver_success = bool(
            _is_success_message(message)
            and np.isfinite(float(result["fun"]))
            and np.all(np.isfinite(full_free_original))
            and bool(linear_summary["all_converged"])
        )
        result_status = "completed" if solver_success else "failed"

        if getattr(args, "state_out", "") and rank == 0:
            export_planestrain_state_npz(
                args.state_out,
                coords_ref=coords_ref,
                x_final=coords_final,
                triangles=np.asarray(params["elems_scalar"], dtype=np.int32),
                case_name=str(params["case_name"]),
                lambda_target=float(args.lambda_target),
                energy=float(result["fun"]),
                metadata={
                    "solver_family": "jax_petsc",
                    "prototype_mode": "zero_history_endpoint",
                    "davis_type": str(params["davis_type"]),
                    "mpi_ranks": int(nprocs),
                },
            )

        finalize_time = float(time.perf_counter() - finalize_start)
        total_time = float(time.perf_counter() - total_start)
        warmup_time = float(assembler.setup_summary().get("warmup", 0.0))
        steady_state_setup_time = max(0.0, float(one_time_setup_time - warmup_time))
        steady_state_total_time = float(steady_state_setup_time + solve_time + finalize_time)
        benchmark_total_time = (
            float(steady_state_total_time)
            if benchmark_mode == "warmup_once_then_solve"
            else float(total_time)
        )
        step_record = {
            "step": 1,
            "time": float(round(solve_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": total_linear_iters,
            "energy": float(result["fun"]),
            "omega": float(omega),
            "u_max": float(u_max),
            "message": message,
            "attempt": used_attempt,
            "ksp_rtol_used": float(used_ksp_rtol),
            "ksp_max_it_used": int(used_ksp_max_it),
            "linesearch_interval_used": [float(used_linesearch[0]), float(used_linesearch[1])],
            "final_grad_norm": float(final_grad_norm),
            "accepted_capped_step_count": int(
                linear_summary.get("n_accepted_via_maxit_direction", 0)
            ),
            "linear_summary": dict(linear_summary),
        }
        if args.save_history:
            history = list(result.get("history", []))
            for hist_entry, linear_entry in zip(history, linear_timing_records):
                hist_entry["linear_ksp"] = {
                    "ksp_reason_name": str(linear_entry.get("ksp_reason_name", "UNKNOWN")),
                    "ksp_reason_code": int(linear_entry.get("ksp_reason_code", 0)),
                    "true_relative_residual": float(
                        linear_entry.get("true_relative_residual", 0.0)
                    ),
                    "true_residual_norm": float(linear_entry.get("true_residual_norm", 0.0)),
                    "ksp_converged": bool(linear_entry.get("ksp_converged", False)),
                    "accepted_via_maxit_direction": bool(
                        linear_entry.get("ksp_accepted_via_maxit_direction", False)
                    ),
                }
                hist_entry["directional_derivative"] = float(
                    linear_entry.get("directional_derivative", np.nan)
                )
                hist_entry["accepted_via_maxit_direction"] = bool(
                    linear_entry.get("ksp_accepted_via_maxit_direction", False)
                )
            step_record["history"] = history
        if args.save_linear_timing:
            step_record["linear_timing"] = list(linear_timing_records)

        payload = {
            "family": "slope_stability",
            "solver": "jax_petsc",
            "prototype_mode": "zero_history_endpoint",
            "case": {
                "name": str(params["case_name"]),
                "backend": "element",
                "analysis": "ssr_endpoint",
                "elem_type": str(params.get("elem_type", "P2")),
                "element_degree": int(_element_degree(params)),
                "lambda_target": float(args.lambda_target),
                "davis_type": str(params["davis_type"]),
                "profile": str(args.profile),
                "level": int(params["level"]),
                "h": float(params["h"]),
                "linesearch_a": float(args.linesearch_a),
                "linesearch_b": float(args.linesearch_b),
                "linesearch_tol": float(args.linesearch_tol),
                "line_search": str(line_search),
                "armijo_alpha0": float(getattr(args, "armijo_alpha0", 1.0)),
                "armijo_c1": float(getattr(args, "armijo_c1", 1.0e-4)),
                "armijo_shrink": float(getattr(args, "armijo_shrink", 0.5)),
                "armijo_max_ls": int(getattr(args, "armijo_max_ls", 40)),
                "benchmark_mode": str(benchmark_mode),
                "use_trust_region": bool(use_trust_region),
                "trust_radius_init": float(trust_radius_init),
                "trust_radius_min": float(trust_radius_min),
                "trust_radius_max": float(trust_radius_max),
                "trust_shrink": float(trust_shrink),
                "trust_expand": float(trust_expand),
                "trust_eta_shrink": float(trust_eta_shrink),
                "trust_eta_expand": float(trust_eta_expand),
                "trust_max_reject": int(trust_max_reject),
                "trust_subproblem_line_search": bool(trust_subproblem_line_search),
                "step_time_limit_s": (
                    None if step_time_limit_s is None else float(step_time_limit_s)
                ),
                "ksp_type": str(settings["ksp_type"]),
                "pc_type": str(settings["pc_type"]),
                "pc_type_effective": (
                    "ksp" if mg_variant == "outer_pcksp" else str(settings["pc_type"])
                ),
                "operator_mode": str(operator_mode),
                "ksp_rtol": float(settings["ksp_rtol"]),
                "ksp_max_it": int(settings["ksp_max_it"]),
                "ksp_accept_true_rel": (
                    None if ksp_accept_true_rel is None else float(ksp_accept_true_rel)
                ),
                "accept_ksp_maxit_direction": bool(
                    getattr(args, "accept_ksp_maxit_direction", True)
                ),
                "guard_ksp_maxit_direction": bool(
                    getattr(args, "guard_ksp_maxit_direction", False)
                ),
                "ksp_maxit_direction_true_rel_cap": float(
                    getattr(args, "ksp_maxit_direction_true_rel_cap", 6.0e-2)
                ),
                "preconditioner_operator": str(preconditioner_operator),
                "fine_pmat_policy": str(fine_pmat_policy),
                "fine_pmat_source": str(fine_pmat_source),
                "fine_pmat_setup_assembly_time": float(fine_pmat_setup_assembly_time),
                "fine_pmat_stagger_period": int(fine_pmat_stagger_period),
                "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                "pc_reuse_preconditioner": bool(pc_reuse_preconditioner),
                "pc_use_amat": bool(pc.getUseAmat()),
                "pc_hypre_type": str(pc_options.get("pc_hypre_type", "")),
                "hypre_nodal_coarsen": int(settings["hypre_nodal_coarsen"]),
                "hypre_vec_interp_variant": int(settings["hypre_vec_interp_variant"]),
                "hypre_strong_threshold": (
                    None
                    if settings["hypre_strong_threshold"] is None
                    else float(settings["hypre_strong_threshold"])
                ),
                "hypre_coarsen_type": str(settings["hypre_coarsen_type"]),
                "hypre_max_iter": int(settings["hypre_max_iter"]),
                "hypre_tol": float(settings["hypre_tol"]),
                "hypre_relax_type_all": str(settings["hypre_relax_type_all"]),
                "gamg_threshold": float(settings["gamg_threshold"]),
                "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                "mg_coarsest_level": int(getattr(args, "mg_coarsest_level", 1)),
                "mg_variant": str(mg_variant),
                "mg_legacy_level_smoothers": {
                    "fine": {
                        "ksp_type": str(legacy_mg_settings["fine"].ksp_type),
                        "pc_type": str(legacy_mg_settings["fine"].pc_type),
                        "steps": int(legacy_mg_settings["fine"].steps),
                    },
                    "degree2": {
                        "ksp_type": str(legacy_mg_settings["degree2"].ksp_type),
                        "pc_type": str(legacy_mg_settings["degree2"].pc_type),
                        "steps": int(legacy_mg_settings["degree2"].steps),
                    },
                    "degree1": {
                        "ksp_type": str(legacy_mg_settings["degree1"].ksp_type),
                        "pc_type": str(legacy_mg_settings["degree1"].pc_type),
                        "steps": int(legacy_mg_settings["degree1"].steps),
                    },
                },
                "mg_coarse_backend": str(explicit_mg_settings["coarse_backend"]),
                "mg_coarse_ksp_type": str(explicit_mg_settings["coarse_ksp_type"]),
                "mg_coarse_pc_type": str(explicit_mg_settings["coarse_pc_type"]),
                "mg_coarse_hypre_nodal_coarsen": int(
                    explicit_mg_settings["coarse_hypre_nodal_coarsen"]
                ),
                "mg_coarse_hypre_vec_interp_variant": int(
                    explicit_mg_settings["coarse_hypre_vec_interp_variant"]
                ),
                "mg_coarse_hypre_vec_interp_variant_guarded": bool(
                    explicit_mg_settings["coarse_hypre_vec_interp_variant_guarded"]
                ),
                "mg_coarse_hypre_strong_threshold": (
                    None
                    if explicit_mg_settings["coarse_hypre_strong_threshold"] is None
                    else float(explicit_mg_settings["coarse_hypre_strong_threshold"])
                ),
                "mg_coarse_hypre_coarsen_type": str(
                    explicit_mg_settings["coarse_hypre_coarsen_type"]
                ),
                "mg_coarse_hypre_max_iter": int(
                    explicit_mg_settings["coarse_hypre_max_iter"]
                ),
                "mg_coarse_hypre_tol": float(explicit_mg_settings["coarse_hypre_tol"]),
                "mg_coarse_hypre_relax_type_all": str(
                    explicit_mg_settings["coarse_hypre_relax_type_all"]
                ),
                "mg_lower_operator_policy": str(mg_lower_operator_policy),
                "mg_fine_python_pc_variant": str(
                    explicit_mg_settings["fine_down"]["python_pc_variant"]
                ),
                "python_pc_variant": str(explicit_mg_settings["python_pc_variant"]),
                "use_near_nullspace": bool(settings["use_near_nullspace"]),
                "element_reorder_mode": str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                "local_hessian_mode": str(getattr(args, "local_hessian_mode", None) or "element"),
                "distribution_strategy": str(distribution_strategy),
                "problem_build_mode": str(getattr(args, "problem_build_mode", "root_bcast")),
                "mg_level_build_mode": str(getattr(args, "mg_level_build_mode", "root_bcast")),
                "mg_transfer_build_mode": str(getattr(args, "mg_transfer_build_mode", "root_bcast")),
                "reuse_hessian_value_buffers": bool(reuse_hessian_value_buffers),
            },
            "mesh": {
                "level": int(params["level"]),
                "h": float(params["h"]),
                "nodes": int(params["nodes"].shape[0]),
                "elements": int(params["elems_scalar"].shape[0]),
                "free_dofs": int(freedofs.size),
                "free_x_dofs": int(np.asarray(params["q_mask"], dtype=bool)[:, 0].sum()),
                "free_y_dofs": int(np.asarray(params["q_mask"], dtype=bool)[:, 1].sum()),
            },
            "material": {
                "raw": {
                    "c0": float(raw_c0),
                    "phi_deg": float(raw_phi_deg),
                    "psi_deg": float(params["psi_deg"]),
                    "E": float(params["E"]),
                    "nu": float(params["nu"]),
                    "gamma": float(params["gamma"]),
                },
                "reduced": {
                    "cohesion": float(reduced_cohesion),
                    "phi_deg": float(reduced_phi_deg),
                },
            },
            "metadata": {
                "profile": str(args.profile),
                "nprocs": int(nprocs),
                "nproc_threads": int(args.nproc),
                "linear_solver": {
                    "ksp_type": str(settings["ksp_type"]),
                    "pc_type": str(settings["pc_type"]),
                    "pc_type_effective": (
                        "ksp" if mg_variant == "outer_pcksp" else str(settings["pc_type"])
                    ),
                    "operator_mode": str(operator_mode),
                    "ksp_rtol": float(settings["ksp_rtol"]),
                    "ksp_max_it": int(settings["ksp_max_it"]),
                    "ksp_accept_true_rel": (
                        None if ksp_accept_true_rel is None else float(ksp_accept_true_rel)
                    ),
                    "accept_ksp_maxit_direction": bool(
                        getattr(args, "accept_ksp_maxit_direction", True)
                    ),
                    "guard_ksp_maxit_direction": bool(
                        getattr(args, "guard_ksp_maxit_direction", False)
                    ),
                    "ksp_maxit_direction_true_rel_cap": float(
                        getattr(args, "ksp_maxit_direction_true_rel_cap", 6.0e-2)
                    ),
                    "preconditioner_operator": str(preconditioner_operator),
                    "fine_pmat_policy": str(fine_pmat_policy),
                    "fine_pmat_source": str(fine_pmat_source),
                    "fine_pmat_setup_assembly_time": float(fine_pmat_setup_assembly_time),
                    "fine_pmat_stagger_period": int(fine_pmat_stagger_period),
                    "mg_strategy": str(getattr(args, "mg_strategy", "legacy_p2_h")),
                    "mg_custom_hierarchy": getattr(args, "mg_custom_hierarchy", None),
                    "mg_variant": str(mg_variant),
                    "mg_level_metadata": list(mg_level_metadata),
                    "mg_coarse_backend": str(explicit_mg_settings["coarse_backend"]),
                    "mg_coarse_ksp_type": str(explicit_mg_settings["coarse_ksp_type"]),
                    "mg_coarse_pc_type": str(explicit_mg_settings["coarse_pc_type"]),
                    "mg_coarse_hypre_nodal_coarsen": int(
                        explicit_mg_settings["coarse_hypre_nodal_coarsen"]
                    ),
                    "mg_coarse_hypre_vec_interp_variant": int(
                        explicit_mg_settings["coarse_hypre_vec_interp_variant"]
                    ),
                    "mg_coarse_hypre_vec_interp_variant_guarded": bool(
                        explicit_mg_settings["coarse_hypre_vec_interp_variant_guarded"]
                    ),
                    "mg_coarse_hypre_strong_threshold": (
                        None
                        if explicit_mg_settings["coarse_hypre_strong_threshold"] is None
                        else float(explicit_mg_settings["coarse_hypre_strong_threshold"])
                    ),
                    "mg_coarse_hypre_coarsen_type": str(
                        explicit_mg_settings["coarse_hypre_coarsen_type"]
                    ),
                    "mg_coarse_hypre_max_iter": int(
                        explicit_mg_settings["coarse_hypre_max_iter"]
                    ),
                    "mg_coarse_hypre_tol": float(explicit_mg_settings["coarse_hypre_tol"]),
                    "mg_coarse_hypre_relax_type_all": str(
                        explicit_mg_settings["coarse_hypre_relax_type_all"]
                    ),
                    "mg_lower_operator_policy": str(mg_lower_operator_policy),
                    "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                    "pc_reuse_preconditioner": bool(pc_reuse_preconditioner),
                    "pc_use_amat": bool(pc.getUseAmat()),
                    "pc_hypre_type": str(pc_options.get("pc_hypre_type", "")),
                    "hypre_nodal_coarsen": int(settings["hypre_nodal_coarsen"]),
                    "hypre_vec_interp_variant": int(settings["hypre_vec_interp_variant"]),
                    "hypre_strong_threshold": (
                        None
                        if settings["hypre_strong_threshold"] is None
                        else float(settings["hypre_strong_threshold"])
                    ),
                    "hypre_coarsen_type": str(settings["hypre_coarsen_type"]),
                    "hypre_max_iter": int(settings["hypre_max_iter"]),
                    "hypre_tol": float(settings["hypre_tol"]),
                    "hypre_relax_type_all": str(settings["hypre_relax_type_all"]),
                    "gamg_threshold": float(settings["gamg_threshold"]),
                    "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                    "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                    "mg_coarsest_level": int(getattr(args, "mg_coarsest_level", 1)),
                    "mg_operator_policy": _mg_operator_policy_name(
                        mg_variant=mg_variant,
                        operator_mode=operator_mode,
                        mg_lower_operator_policy=mg_lower_operator_policy,
                    ),
                    "mg_legacy_level_smoothers": {
                        "fine": {
                            "ksp_type": str(legacy_mg_settings["fine"].ksp_type),
                            "pc_type": str(legacy_mg_settings["fine"].pc_type),
                            "steps": int(legacy_mg_settings["fine"].steps),
                        },
                        "degree2": {
                            "ksp_type": str(legacy_mg_settings["degree2"].ksp_type),
                            "pc_type": str(legacy_mg_settings["degree2"].pc_type),
                            "steps": int(legacy_mg_settings["degree2"].steps),
                        },
                        "degree1": {
                            "ksp_type": str(legacy_mg_settings["degree1"].ksp_type),
                            "pc_type": str(legacy_mg_settings["degree1"].pc_type),
                            "steps": int(legacy_mg_settings["degree1"].steps),
                        },
                    },
                    "mg_fine_down": dict(explicit_mg_settings["fine_down"]),
                    "mg_fine_up": dict(explicit_mg_settings["fine_up"]),
                    "mg_fine_python_pc_variant": str(
                        explicit_mg_settings["fine_down"]["python_pc_variant"]
                    ),
                    "mg_intermediate_steps": int(explicit_mg_settings["intermediate_steps"]),
                    "mg_intermediate_pc_type": str(explicit_mg_settings["intermediate_pc_type"]),
                    "mg_intermediate_degree_pc_types": dict(
                        explicit_mg_settings["intermediate_degree_pc_types"]
                    ),
                    "mg_hierarchy_levels": list(mg_hierarchy_metadata.get("level_records", [])),
                    "mg_hierarchy_transfers": list(
                        mg_hierarchy_metadata.get("transfer_records", [])
                    ),
                    "python_pc_variant": str(explicit_mg_settings["python_pc_variant"]),
                    "outer_pcksp_inner_ksp_type": str(
                        explicit_mg_settings["outer_pcksp_inner_ksp_type"]
                    ),
                    "outer_pcksp_inner_ksp_rtol": float(
                        explicit_mg_settings["outer_pcksp_inner_ksp_rtol"]
                    ),
                    "outer_pcksp_inner_ksp_max_it": int(
                        explicit_mg_settings["outer_pcksp_inner_ksp_max_it"]
                    ),
                    "use_near_nullspace": bool(settings["use_near_nullspace"]),
                    "matrix_block_size": 2,
                    "reorder": bool(settings["reorder"]),
                    "assembly_mode": "element",
                    "element_reorder_mode": str(getattr(args, "element_reorder_mode", None) or "block_xyz"),
                    "local_hessian_mode": str(getattr(args, "local_hessian_mode", None) or "element"),
                    "distribution_strategy": str(getattr(assembler, "distribution_strategy", "overlap_allgather")),
                    "problem_build_mode": str(getattr(args, "problem_build_mode", "root_bcast")),
                    "mg_level_build_mode": str(getattr(args, "mg_level_build_mode", "root_bcast")),
                    "mg_transfer_build_mode": str(getattr(args, "mg_transfer_build_mode", "root_bcast")),
                    "reuse_hessian_value_buffers": bool(reuse_hessian_value_buffers),
                    "assembler": assembler.__class__.__name__,
                    "preconditioner_case_name": (
                        None if pc_params is None else str(pc_params["case_name"])
                    ),
                    "preconditioner_elem_type": (
                        None if pc_params is None else str(pc_params.get("elem_type", ""))
                    ),
                    "preconditioner_elements": (
                        None
                        if pc_params is None
                        else int(np.asarray(pc_params["elems_scalar"]).shape[0])
                    ),
                    "trust_subproblem_solver": "petsc_ksp" if trust_ksp_subproblem else "reduced_2d",
                    "trust_subproblem_line_search": bool(trust_subproblem_line_search),
                },
                "newton": {
                    "tolf": float(args.tolf),
                    "tolg": float(args.tolg),
                    "tolg_rel": float(args.tolg_rel),
                    "tolx_rel": float(args.tolx_rel),
                    "tolx_abs": float(args.tolx_abs),
                    "maxit": int(args.maxit),
                    "step_time_limit_s": (
                        None if step_time_limit_s is None else float(step_time_limit_s)
                    ),
                    "require_all_convergence": True,
                    "fail_on_nonfinite": True,
                },
            },
            "timings": {
                "setup_time": float(assembler_setup_time),
                "problem_build_time": float(problem_build_time),
                "main_problem_build_time": float(main_problem_build_time),
                "preconditioner_problem_build_time": float(preconditioner_problem_build_time),
                "assembler_setup_time": float(assembler_setup_time),
                "assembler_setup_breakdown": assembler.setup_summary(),
                "solver_bootstrap_time": float(solver_bootstrap_time),
                "solver_bootstrap_breakdown": {
                    "mg_hierarchy_build_time": float(mg_hierarchy_build_time),
                    "mg_level_build_time": float(mg_hierarchy_metadata.get("level_build_time", 0.0)),
                    "mg_level_records": list(mg_hierarchy_metadata.get("level_records", [])),
                    "mg_transfer_build_time": float(
                        mg_hierarchy_metadata.get("transfer_build_time", 0.0)
                    ),
                    "mg_transfer_records": list(
                        mg_hierarchy_metadata.get("transfer_records", [])
                    ),
                    "mg_transfer_cache_hits": int(
                        mg_hierarchy_metadata.get("transfer_cache_hits", 0)
                    ),
                    "mg_transfer_cache_io_time": float(
                        mg_hierarchy_metadata.get("transfer_cache_io_time", 0.0)
                    ),
                    "mg_transfer_cache_build_time": float(
                        mg_hierarchy_metadata.get("transfer_cache_build_time", 0.0)
                    ),
                    "mg_transfer_mapping_time": float(
                        mg_hierarchy_metadata.get("transfer_mapping_time", 0.0)
                    ),
                    "mg_transfer_matrix_build_time": float(
                        mg_hierarchy_metadata.get("transfer_matrix_build_time", 0.0)
                    ),
                    "mg_level_assembler_build_time": float(mg_level_assembler_build_time),
                    "mg_configure_time": float(mg_configure_time),
                },
                "one_time_setup_time": float(one_time_setup_time),
                "steady_state_setup_time": float(steady_state_setup_time),
                "fine_pmat_setup_assembly_time": float(fine_pmat_setup_assembly_time),
                "solve_time": float(solve_time),
                "finalize_time": float(finalize_time),
                "steady_state_total_time": float(steady_state_total_time),
                "benchmark_total_time": float(benchmark_total_time),
                "total_time": total_time,
                "callback_summary": dict(callback_summary),
            },
            "result": {
                "mesh_level": int(params["level"]),
                "total_dofs": int(len(np.asarray(params["u_0"], dtype=np.float64))),
                "free_dofs": int(freedofs.size),
                "setup_time": float(assembler_setup_time),
                "one_time_setup_time": float(one_time_setup_time),
                "steady_state_setup_time": float(steady_state_setup_time),
                "solve_time_total": float(solve_time),
                "steady_state_total_time": float(steady_state_total_time),
                "benchmark_total_time": float(benchmark_total_time),
                "total_time": total_time,
                "status": result_status,
                "solver_success": bool(solver_success),
                "final_grad_norm": float(final_grad_norm),
                "accepted_capped_step_count": int(
                    linear_summary.get("n_accepted_via_maxit_direction", 0)
                ),
                "steps": [step_record],
            },
        }
        return payload
    finally:
        if x_initial is not None:
            x_initial.destroy()
        if x is not None:
            x.destroy()
        if residual_ax is not None:
            residual_ax.destroy()
        if residual_vec is not None:
            residual_vec.destroy()
        if pc_assembler is not None:
            pc_assembler.cleanup()
        if mg_level_work_vecs is not None:
            for work_vec in mg_level_work_vecs:
                work_vec.destroy()
        if mg_level_assemblers is not None:
            for level_assembler in mg_level_assemblers:
                level_assembler.cleanup()
        if mg_lower_operator_policy == "galerkin_refresh" and mg_fixed_level_operators is not None:
            for mat in mg_fixed_level_operators[:-1]:
                if mat is not None:
                    mat.destroy()
        if mg_hierarchy is not None:
            mg_hierarchy.cleanup()
        assembler.cleanup()
