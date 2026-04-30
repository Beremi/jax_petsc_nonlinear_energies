#!/usr/bin/env python3
"""HyperElasticity 3D pure-JAX solver with the final serial trust policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.core.benchmark.state_export import export_hyperelasticity_state_npz
from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.hyperelasticity.jax.jax_energy import J
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D
from src.problems.hyperelasticity.support.rotate_boundary import (
    rotate_right_face_from_reference,
)


def _result_from_step_message(message: str) -> str:
    message_lower = str(message).lower()
    if "converged" in message_lower or "satisfied" in message_lower:
        return "completed"
    return "failed"


def run_level(
    level: int,
    steps: int,
    total_steps: int,
    *,
    maxit: int = 100,
    linesearch_interval: tuple[float, float] = (-0.5, 2.0),
    linesearch_tol: float = 1e-1,
    ksp_rtol: float = 1e-1,
    ksp_max_it: int = 30,
    tolf: float = 1e-4,
    tolg: float = 1e-3,
    tolg_rel: float = 1e-3,
    tolx_rel: float = 1e-3,
    tolx_abs: float = 1e-10,
    require_all_convergence: bool = True,
    use_trust_region: bool = True,
    trust_radius_init: float = 0.5,
    trust_radius_min: float = 1e-8,
    trust_radius_max: float = 1e6,
    trust_shrink: float = 0.5,
    trust_expand: float = 1.5,
    trust_eta_shrink: float = 0.05,
    trust_eta_expand: float = 0.75,
    trust_max_reject: int = 6,
    trust_subproblem_line_search: bool = True,
    start_step: int = 1,
    verbose: bool = False,
    state_out: str = "",
) -> dict:
    total_runtime_start = time.perf_counter()
    setup_start = time.perf_counter()

    mesh = MeshHyperElasticity3D(mesh_level=level)
    params, adjacency, u_init = mesh.get_data_jax()
    energy = EnergyDerivator(J, params, adjacency, u_init)

    setup_time = time.perf_counter() - setup_start
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    nodes2coord = np.asarray(mesh.params["nodes2coord"], dtype=np.float64)
    u0_reference = np.asarray(params["u_0"], dtype=np.float64).copy()

    rotation_per_iter = 4.0 * 2.0 * np.pi / float(total_steps)

    step_results = []
    current_u = np.asarray(u_init, dtype=np.float64).copy()

    for step in range(start_step, start_step + steps):
        angle = float(step * rotation_per_iter)
        u0_step = rotate_right_face_from_reference(
            u0_reference,
            nodes2coord,
            angle,
        )
        params["u_0"] = jnp.asarray(u0_step, dtype=jnp.float64)
        energy.params = params

        F, dF, ddF = energy.get_derivatives()
        ddf_with_solver = HessSolverGenerator(
            ddf=ddF,
            solver_type="amg",
            elastic_kernel=mesh.elastic_kernel,
            verbose=verbose,
            tol=ksp_rtol,
            maxiter=ksp_max_it,
        )

        step_start = time.perf_counter()
        res = newton(
            F,
            dF,
            ddf_with_solver,
            current_u,
            tolf=tolf,
            tolg=tolg,
            tolg_rel=tolg_rel,
            linesearch_tol=linesearch_tol,
            linesearch_interval=linesearch_interval,
            maxit=maxit,
            tolx_rel=tolx_rel,
            tolx_abs=tolx_abs,
            require_all_convergence=require_all_convergence,
            fail_on_nonfinite=True,
            verbose=verbose,
            trust_region=use_trust_region,
            trust_radius_init=trust_radius_init,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            trust_shrink=trust_shrink,
            trust_expand=trust_expand,
            trust_eta_shrink=trust_eta_shrink,
            trust_eta_expand=trust_eta_expand,
            trust_max_reject=trust_max_reject,
            trust_subproblem_line_search=trust_subproblem_line_search,
            save_history=True,
            save_linear_timing=True,
        )
        step_time = time.perf_counter() - step_start
        current_u = np.asarray(res["x"], dtype=np.float64).copy()

        step_results.append(
            {
                "step": int(step),
                "angle": angle,
                "time": float(step_time),
                "iters": int(res["nit"]),
                "energy": float(res["fun"]),
                "message": str(res["message"]),
                "history": res.get("history", []),
                "linear_timing": res.get("linear_timing", []),
            }
        )

    total_time = time.perf_counter() - total_runtime_start
    total_newton = int(sum(int(step["iters"]) for step in step_results))
    total_linear = int(
        sum(
            int(rec.get("ksp_its", 0))
            for step in step_results
            for rec in step.get("linear_timing", [])
        )
    )
    result = "completed"
    if step_results:
        result = _result_from_step_message(str(step_results[-1]["message"]))

    if state_out:
        freedofs = np.asarray(mesh.params["dofsMinim"], dtype=np.int64).ravel()
        coords_ref = np.asarray(mesh.params["nodes2coord"], dtype=np.float64).reshape((-1, 3))
        x_full = np.asarray(params["u_0"], dtype=np.float64).copy()
        x_full[freedofs] = current_u
        export_hyperelasticity_state_npz(
            state_out,
            coords_ref=coords_ref,
            x_final=x_full,
            tetrahedra=np.asarray(mesh.params["elems2nodes"], dtype=np.int32),
            mesh_level=int(level),
            total_steps=int(total_steps),
            energy=(None if not step_results else float(step_results[-1]["energy"])),
            metadata={"solver_family": "pure_jax"},
        )

    return {
        "solver": "pure_jax",
        "backend": "serial",
        "level": int(level),
        "nprocs": 1,
        "total_dofs": int(np.asarray(mesh.params["u0"]).size),
        "free_dofs": int(len(u_init)),
        "start_step": int(start_step),
        "steps_requested": int(steps),
        "total_steps": int(total_steps),
        "setup_time": float(setup_time),
        "time": float(total_time),
        "total_newton_iters": total_newton,
        "total_linear_iters": total_linear,
        "result": result,
        "solver_options": {
            "use_trust_region": bool(use_trust_region),
            "trust_subproblem_solver": "serial_stcg",
            "trust_subproblem_line_search": bool(trust_subproblem_line_search),
            "linesearch_interval": [float(linesearch_interval[0]), float(linesearch_interval[1])],
            "linesearch_tol": float(linesearch_tol),
            "trust_radius_init": float(trust_radius_init),
            "trust_radius_min": float(trust_radius_min),
            "trust_radius_max": float(trust_radius_max),
            "trust_shrink": float(trust_shrink),
            "trust_expand": float(trust_expand),
            "trust_eta_shrink": float(trust_eta_shrink),
            "trust_eta_expand": float(trust_eta_expand),
            "trust_max_reject": int(trust_max_reject),
            "ksp_rtol": float(ksp_rtol),
            "ksp_max_it": int(ksp_max_it),
            "require_all_convergence": bool(require_all_convergence),
        },
        "steps": step_results,
        "jax_setup_timing": energy.timings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--total_steps", type=int, default=24)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument("--linesearch_a", type=float, default=-0.5)
    parser.add_argument("--linesearch_b", type=float, default=2.0)
    parser.add_argument("--linesearch_tol", type=float, default=1e-1)
    parser.add_argument("--ksp_rtol", type=float, default=1e-1)
    parser.add_argument("--ksp_max_it", type=int, default=30)
    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_rel", type=float, default=1e-3)
    parser.add_argument("--tolx_abs", type=float, default=1e-10)
    parser.add_argument("--no_require_all_convergence", action="store_true")
    parser.add_argument("--no_use_trust_region", action="store_true")
    parser.add_argument("--trust_radius_init", type=float, default=0.5)
    parser.add_argument("--trust_radius_min", type=float, default=1e-8)
    parser.add_argument("--trust_radius_max", type=float, default=1e6)
    parser.add_argument("--trust_shrink", type=float, default=0.5)
    parser.add_argument("--trust_expand", type=float, default=1.5)
    parser.add_argument("--trust_eta_shrink", type=float, default=0.05)
    parser.add_argument("--trust_eta_expand", type=float, default=0.75)
    parser.add_argument("--trust_max_reject", type=int, default=6)
    parser.add_argument("--no_trust_subproblem_line_search", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--state-out", type=str, default="")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = run_level(
        args.level,
        args.steps,
        args.total_steps,
        maxit=args.maxit,
        start_step=args.start_step,
        linesearch_interval=(args.linesearch_a, args.linesearch_b),
        linesearch_tol=args.linesearch_tol,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        require_all_convergence=not args.no_require_all_convergence,
        use_trust_region=not args.no_use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=not args.no_trust_subproblem_line_search,
        verbose=not args.quiet,
        state_out=args.state_out,
    )

    print(json.dumps(result, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
