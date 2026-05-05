"""Shared scalar JAX + PETSc nonlinear driver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.core.benchmark.repair import build_retry_attempts, needs_solver_repair
from src.core.petsc.gamg import build_gamg_coordinates
from src.core.petsc.minimizers import newton
from src.core.petsc.trust_ksp import ksp_cg_set_radius


LinearRecordExtras = Callable[[PETSc.KSP, PETSc.Vec], dict[str, object]]


@dataclass(frozen=True)
class ScalarProblemDriverSpec:
    problem_name: str
    mesh_loader: Callable[[int], tuple[dict, object, np.ndarray]]
    assembler_factories: Mapping[str, Callable[..., object]]
    default_profile_defaults: Mapping[str, Mapping[str, object]]
    line_search_defaults: Mapping[str, object]
    trust_region_defaults: Mapping[str, object]
    repair_policy: Callable[[object, Mapping[str, object]], list[tuple[str, tuple[float, float], float, int]]] | None
    result_formatter: Callable[[dict], dict]
    linear_record_extras: LinearRecordExtras | None = None


def resolve_linear_settings(args, profile_defaults: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    settings = dict(profile_defaults[args.profile])
    overrides = {
        "ksp_type": args.ksp_type,
        "pc_type": args.pc_type,
        "ksp_rtol": args.ksp_rtol,
        "ksp_max_it": args.ksp_max_it,
        "pc_setup_on_ksp_cap": args.pc_setup_on_ksp_cap,
        "gamg_threshold": args.gamg_threshold,
        "gamg_agg_nsmooths": args.gamg_agg_nsmooths,
        "gamg_set_coordinates": args.gamg_set_coordinates,
        "reorder": args.reorder,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def _pc_options(settings: Mapping[str, object]) -> dict[str, object]:
    opts: dict[str, object] = {}
    if settings["pc_type"] == "gamg":
        opts["pc_gamg_threshold"] = float(settings["gamg_threshold"])
        opts["pc_gamg_agg_nsmooths"] = int(settings["gamg_agg_nsmooths"])
    return opts


def run_scalar_problem(spec: ScalarProblemDriverSpec, args) -> dict:
    """Run a scalar JAX + PETSc problem with the shared driver."""
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    total_runtime_start = time.perf_counter()

    settings = resolve_linear_settings(args, spec.default_profile_defaults)
    pc_options = _pc_options(settings)
    use_element_assembly = args.assembly_mode == "element"
    element_reorder_mode = str(
        getattr(args, "element_reorder_mode", None) or "block_xyz"
    )
    local_hessian_mode = str(
        getattr(args, "local_hessian_mode", None) or "element"
    )

    params, adjacency, u_init = spec.mesh_loader(int(args.level))

    setup_start = time.perf_counter()
    if use_element_assembly:
        assembler = spec.assembler_factories["element"](
            params=params,
            comm=comm,
            adjacency=adjacency,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
            ksp_max_it=int(settings["ksp_max_it"]),
            pc_options=pc_options,
            reorder_mode=element_reorder_mode,
            local_hessian_mode=local_hessian_mode,
        )
    else:
        factory_key = "local_coloring" if args.local_coloring else "parallel_dof"
        assembler_kwargs = dict(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=args.coloring_trials,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_type=str(settings["ksp_type"]),
            pc_type=str(settings["pc_type"]),
        )
        if args.local_coloring:
            assembler_kwargs["hvp_eval_mode"] = str(args.hvp_eval_mode)
        assembler = spec.assembler_factories[factory_key](**assembler_kwargs)
    setup_time = time.perf_counter() - setup_start

    u_init_reordered = np.asarray(u_init, dtype=np.float64)[assembler.part.perm]
    x = assembler.create_vec(u_init_reordered)
    x_initial = x.duplicate()
    x.copy(x_initial)

    ksp = assembler.ksp
    A = assembler.A
    pc = ksp.getPC()
    gamg_coords = None
    if settings["pc_type"] == "gamg" and settings["gamg_set_coordinates"]:
        gamg_coords = build_gamg_coordinates(
            assembler.part,
            np.asarray(params["freedofs"], dtype=np.int64),
            np.asarray(params["nodes"], dtype=np.float64),
        )

    linesearch_interval = (float(args.linesearch_a), float(args.linesearch_b))
    line_search = str(getattr(args, "line_search", "golden_fixed"))
    use_trust_region = bool(getattr(args, "use_trust_region", False))
    trust_radius_init = float(getattr(args, "trust_radius_init", spec.trust_region_defaults.get("trust_radius_init", 1.0)))
    trust_radius_min = float(getattr(args, "trust_radius_min", spec.trust_region_defaults.get("trust_radius_min", 1e-8)))
    trust_radius_max = float(getattr(args, "trust_radius_max", spec.trust_region_defaults.get("trust_radius_max", 1e6)))
    trust_shrink = float(getattr(args, "trust_shrink", spec.trust_region_defaults.get("trust_shrink", 0.5)))
    trust_expand = float(getattr(args, "trust_expand", spec.trust_region_defaults.get("trust_expand", 1.5)))
    trust_eta_shrink = float(getattr(args, "trust_eta_shrink", spec.trust_region_defaults.get("trust_eta_shrink", 0.05)))
    trust_eta_expand = float(getattr(args, "trust_eta_expand", spec.trust_region_defaults.get("trust_eta_expand", 0.75)))
    trust_max_reject = int(getattr(args, "trust_max_reject", spec.trust_region_defaults.get("trust_max_reject", 6)))
    trust_subproblem_line_search = bool(
        getattr(
            args,
            "trust_subproblem_line_search",
            spec.trust_region_defaults.get("trust_subproblem_line_search", False),
        )
    )
    trust_ksp_subproblem = bool(
        use_trust_region and str(settings["ksp_type"]).lower() in {"stcg", "nash", "gltr"}
    )
    step_time_limit_s = getattr(args, "step_time_limit_s", None)

    linear_timing_records: list[dict[str, object]] = []
    linear_iters_this_attempt: list[int] = []
    force_pc_setup_next = True
    used_ksp_rtol = float(settings["ksp_rtol"])
    used_ksp_max_it = int(settings["ksp_max_it"])

    def _assemble_and_solve(vec, rhs, sol, ksp_rtol_attempt, ksp_max_it_attempt, trust_radius=None):
        nonlocal force_pc_setup_next, gamg_coords

        t_asm0 = time.perf_counter()
        u_owned = np.array(vec.array[:], dtype=np.float64)
        if use_element_assembly:
            assembler.assemble_hessian(u_owned)
        else:
            assembler.assemble_hessian(u_owned, variant=2)
        asm_total_time = time.perf_counter() - t_asm0

        asm_details = {}
        if assembler.iter_timings:
            asm_details = dict(assembler.iter_timings[-1])
        asm_details["assembly_total_time"] = float(asm_total_time)

        if trust_radius is not None:
            ksp_cg_set_radius(ksp, float(trust_radius))

        t_setop0 = time.perf_counter()
        ksp.setOperators(A)
        if gamg_coords is not None:
            pc.setCoordinates(gamg_coords)
            gamg_coords = None
        t_setop = time.perf_counter() - t_setop0

        t_tol0 = time.perf_counter()
        ksp.setTolerances(
            rtol=float(ksp_rtol_attempt), max_it=int(ksp_max_it_attempt)
        )
        t_tol = time.perf_counter() - t_tol0

        t_setup0 = time.perf_counter()
        if settings["pc_setup_on_ksp_cap"]:
            if force_pc_setup_next:
                ksp.setUp()
                force_pc_setup_next = False
        else:
            ksp.setUp()
        t_setup = time.perf_counter() - t_setup0

        t_solve0 = time.perf_counter()
        ksp.solve(rhs, sol)
        t_solve = time.perf_counter() - t_solve0
        ksp_its = int(ksp.getIterationNumber())
        linear_iters_this_attempt.append(ksp_its)

        if settings["pc_setup_on_ksp_cap"] and ksp_its >= int(ksp_max_it_attempt):
            force_pc_setup_next = True

        if args.save_linear_timing:
            record: dict[str, object] = {
                "assemble_total_time": float(asm_total_time),
                "assemble_p2p_exchange": float(asm_details.get("p2p_exchange", 0.0)),
                "assemble_hvp_compute": float(asm_details.get("hvp_compute", 0.0)),
                "assemble_extraction": float(asm_details.get("extraction", 0.0)),
                "assemble_coo_assembly": float(asm_details.get("coo_assembly", 0.0)),
                "assemble_n_hvps": int(asm_details.get("n_hvps", 0)),
                "setop_time": float(t_setop),
                "set_tolerances_time": float(t_tol),
                "pc_setup_time": float(t_setup),
                "solve_time": float(t_solve),
                "linear_total_time": float(
                    asm_total_time + t_setop + t_tol + t_setup + t_solve
                ),
                "ksp_its": int(ksp_its),
            }
            if spec.linear_record_extras is not None:
                record.update(spec.linear_record_extras(ksp, rhs))
            if trust_radius is not None:
                record["trust_radius"] = float(trust_radius)
            linear_timing_records.append(record)

        return ksp_its

    def hessian_solve_fn(vec, rhs, sol):
        return _assemble_and_solve(
            vec, rhs, sol, used_ksp_rtol, used_ksp_max_it, trust_radius=None
        )

    def trust_subproblem_solve_fn(vec, rhs, sol, trust_radius):
        return _assemble_and_solve(
            vec,
            rhs,
            sol,
            used_ksp_rtol,
            used_ksp_max_it,
            trust_radius=float(trust_radius),
        )

    if spec.repair_policy is not None:
        attempt_specs = spec.repair_policy(args, settings)
    else:
        attempt_specs = build_retry_attempts(
            retry_on_failure=bool(args.retry_on_failure),
            linesearch_interval=linesearch_interval,
            ksp_rtol=float(settings["ksp_rtol"]),
            ksp_max_it=int(settings["ksp_max_it"]),
        )

    step_records: list[dict[str, object]] = []
    result = None
    used_attempt = "primary"
    used_linesearch = linesearch_interval
    solve_time = 0.0

    try:
        for idx, (attempt_name, ls_interval, ksp_rtol_attempt, ksp_max_it_attempt) in enumerate(
            attempt_specs
        ):
            x_initial.copy(x)
            force_pc_setup_next = True
            linear_iters_this_attempt = []
            if args.save_linear_timing:
                linear_timing_records = []

            used_attempt = attempt_name
            used_linesearch = ls_interval
            used_ksp_rtol = float(ksp_rtol_attempt)
            used_ksp_max_it = int(ksp_max_it_attempt)

            t0 = time.perf_counter()
            result = newton(
                energy_fn=assembler.energy_fn,
                gradient_fn=assembler.gradient_fn,
                hessian_solve_fn=hessian_solve_fn,
                x=x,
                tolf=float(args.tolf),
                tolg=float(args.tolg),
                tolg_rel=float(args.tolg_rel),
                linesearch_tol=float(args.linesearch_tol),
                linesearch_interval=ls_interval,
                line_search=line_search,
                maxit=int(args.maxit),
                tolx_rel=float(args.tolx_rel),
                tolx_abs=float(args.tolx_abs),
                require_all_convergence=True,
                fail_on_nonfinite=True,
                verbose=(not args.quiet),
                comm=comm,
                ghost_update_fn=None,
                hessian_matvec_fn=lambda _x, vin, vout: assembler.A.mult(vin, vout),
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
            solve_time = time.perf_counter() - t0

            if needs_solver_repair(result) and idx + 1 < len(attempt_specs):
                continue
            break

        if result is None:
            raise RuntimeError("Newton solver did not return a result")

        step_record: dict[str, object] = {
            "step": 1,
            "time": float(round(solve_time, 6)),
            "nit": int(result["nit"]),
            "linear_iters": int(sum(linear_iters_this_attempt)),
            "energy": float(result["fun"]),
            "message": str(result["message"]),
            "attempt": used_attempt,
            "ksp_rtol_used": float(used_ksp_rtol),
            "ksp_max_it_used": int(used_ksp_max_it),
            "linesearch_interval_used": [
                float(used_linesearch[0]),
                float(used_linesearch[1]),
            ],
        }
        if step_time_limit_s is not None:
            step_record["step_time_limit_s"] = float(step_time_limit_s)
            step_record["kill_switch_exceeded"] = bool(
                solve_time > float(step_time_limit_s)
            )
        if args.save_history:
            step_record["history"] = result.get("history", [])
        if args.save_linear_timing:
            step_record["linear_timing"] = list(linear_timing_records)
        step_records.append(step_record)

    finally:
        x_initial.destroy()
        x.destroy()
        assembler.cleanup()

    return {
        "mesh_level": int(args.level),
        "total_dofs": int(len(params["u_0"])),
        "free_dofs": int(assembler.part.n_free),
        "setup_time": float(round(setup_time, 6)),
        "solve_time_total": float(round(sum(step["time"] for step in step_records), 6)),
        "total_time": float(round(time.perf_counter() - total_runtime_start, 6)),
        "steps": step_records,
        "metadata": {
            "profile": args.profile,
            "nprocs": nprocs,
            "nproc_threads": max(1, int(args.nproc)),
            "problem": {
                "name": spec.problem_name,
            },
            "linear_solver": {
                "ksp_type": str(settings["ksp_type"]),
                "pc_type": str(settings["pc_type"]),
                "ksp_rtol": float(settings["ksp_rtol"]),
                "ksp_max_it": int(settings["ksp_max_it"]),
                "pc_setup_on_ksp_cap": bool(settings["pc_setup_on_ksp_cap"]),
                "gamg_threshold": float(settings["gamg_threshold"]),
                "gamg_agg_nsmooths": int(settings["gamg_agg_nsmooths"]),
                "gamg_set_coordinates": bool(settings["gamg_set_coordinates"]),
                "reorder": bool(settings["reorder"]),
                "hvp_eval_mode": str(getattr(assembler, "_hvp_eval_mode", "batched")),
                "assembly_mode": str(args.assembly_mode),
                "element_reorder_mode": (
                    element_reorder_mode if use_element_assembly else None
                ),
                "local_hessian_mode": (
                    local_hessian_mode if use_element_assembly else None
                ),
                "distribution_strategy": str(
                    getattr(assembler, "distribution_strategy", "reduced_free_dofs")
                ),
                "assembler": assembler.__class__.__name__,
                "trust_subproblem_solver": (
                    "petsc_ksp" if trust_ksp_subproblem else "direct_linear_solve"
                ),
                "trust_subproblem_line_search": bool(trust_subproblem_line_search),
            },
            "newton": {
                "tolf": float(args.tolf),
                "tolg": float(args.tolg),
                "tolg_rel": float(args.tolg_rel),
                "tolx_rel": float(args.tolx_rel),
                "tolx_abs": float(args.tolx_abs),
                "maxit": int(args.maxit),
                "require_all_convergence": True,
                "fail_on_nonfinite": True,
                "linesearch_interval": [float(args.linesearch_a), float(args.linesearch_b)],
                "linesearch_tol": float(args.linesearch_tol),
                "line_search": str(line_search),
                "trust_region": bool(use_trust_region),
                "trust_radius_init": float(trust_radius_init),
                "trust_radius_min": float(trust_radius_min),
                "trust_radius_max": float(trust_radius_max),
                "trust_subproblem_line_search": bool(trust_subproblem_line_search),
                "step_time_limit_s": (
                    None if step_time_limit_s is None else float(step_time_limit_s)
                ),
            },
        },
    }
