#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver

from src.core.benchmark.state_export import export_hyperelasticity_state_npz
from experiments.analysis.hyperelastic_companion_common import (
    centerline_profile,
    compute_energy_from_full_coordinates,
    displacement_from_full_coordinates,
    load_hyperelastic_case,
    max_displacement_norm,
    step_schedule,
)


class TensileHyperelasticity(Problem):
    def custom_init(self, mu, lam):
        self.mu = float(mu)
        self.lam = float(lam)

    def get_tensor_map(self):
        mu = float(self.mu)
        lam = float(self.lam)

        def first_piola(u_grad):
            f = u_grad + jnp.eye(self.dim, dtype=u_grad.dtype)
            j = jnp.linalg.det(f)
            j = jnp.maximum(j, 1.0e-12)
            finv_t = jnp.linalg.inv(f).T
            return mu * (f - finv_t) + lam * jnp.log(j) * finv_t

        return first_piola


def _build_problem(case: dict[str, object], displacement_x: float) -> TensileHyperelasticity:
    coords_ref = np.asarray(case["coords_ref"], dtype=np.float64)
    tetrahedra = np.asarray(case["tetrahedra"], dtype=np.int32)
    mesh = Mesh(coords_ref, tetrahedra, ele_type="TET4")
    x_min = float(case["x_min"])
    x_max = float(case["x_max"])
    atol = 1.0e-12

    def on_left(point):
        return jnp.isclose(point[0], x_min, atol=atol)

    def on_right(point):
        return jnp.isclose(point[0], x_max, atol=atol)

    zero_value = lambda point: 0.0
    dirichlet_bc_info = [
        [on_left, on_left, on_left, on_right, on_right, on_right],
        [0, 1, 2, 0, 1, 2],
        [
            zero_value,
            zero_value,
            zero_value,
            lambda point, dx=float(displacement_x): dx,
            zero_value,
            zero_value,
        ],
    ]
    c1 = float(case["C1"])
    d1 = float(case["D1"])
    mu = 2.0 * c1
    lam = 2.0 * d1
    return TensileHyperelasticity(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type="TET4",
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(mu, lam),
    )


def run_case(
    *,
    level: int,
    schedule: list[float],
    tol: float,
    rel_tol: float,
    line_search: bool,
    out_json: Path,
    state_out: Path,
) -> None:
    case = load_hyperelastic_case(level)
    coords_ref = np.asarray(case["coords_ref"], dtype=np.float64)
    params = dict(case["params"])
    current_sol = None
    rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    final_coords = coords_ref.copy()

    for step_id, displacement_x in enumerate(schedule, start=1):
        problem = _build_problem(case, displacement_x)
        solver_options: dict[str, object] = {
            "umfpack_solver": {},
            "line_search_flag": bool(line_search),
            "tol": float(tol),
            "rel_tol": float(rel_tol),
        }
        if current_sol is not None:
            solver_options["initial_guess"] = current_sol

        step_start = time.perf_counter()
        sol_list = solver(problem, solver_options)
        step_wall = time.perf_counter() - step_start
        current_sol = [np.asarray(sol, dtype=np.float64).copy() for sol in sol_list]

        displacement = np.asarray(current_sol[0], dtype=np.float64)
        final_coords = coords_ref + displacement
        x_full = final_coords.reshape((-1,))
        energy = compute_energy_from_full_coordinates(x_full, params)
        rows.append(
            {
                "step": int(step_id),
                "displacement_x": float(displacement_x),
                "wall_time_s": float(step_wall),
                "energy": float(energy),
                "u_max": float(max_displacement_norm(displacement)),
            }
        )

    total_wall = time.perf_counter() - total_start
    final_displacement = displacement_from_full_coordinates(final_coords, coords_ref)
    centerline = centerline_profile(coords_ref, final_displacement)
    export_hyperelasticity_state_npz(
        state_out,
        coords_ref=coords_ref,
        x_final=final_coords,
        tetrahedra=np.asarray(case["tetrahedra"], dtype=np.int32),
        mesh_level=int(level),
        total_steps=len(schedule),
        energy=(None if not rows else float(rows[-1]["energy"])),
        metadata={"solver_family": "jax_fem_umfpack"},
    )
    payload = {
        "implementation": "jax_fem_umfpack_serial",
        "level": int(level),
        "mesh_path": str(case["mesh_path"]),
        "schedule": [float(value) for value in schedule],
        "case_contract": {
            "constitutive_law": "compressible_neo_hookean",
            "mesh_family": "HyperElasticity_level1_tet4",
            "boundary_conditions": "left_clamp_right_face_prescribed_translation_xyz",
        },
        "solver_options": {
            "tol": float(tol),
            "rel_tol": float(rel_tol),
            "line_search_flag": bool(line_search),
            "linear_solver": "umfpack",
        },
        "total_wall_time_s": float(total_wall),
        "free_dofs": int(np.asarray(case["freedofs"]).size),
        "state_npz": str(state_out),
        "rows": rows,
        "final_metrics": {
            "energy": (None if not rows else float(rows[-1]["energy"])),
            "u_max": (None if not rows else float(rows[-1]["u_max"])),
            "centerline_x": centerline["x"].tolist(),
            "centerline_ux": centerline["ux"].tolist(),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAX-FEM hyperelastic companion case.")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--schedule", type=float, nargs="+", default=list(step_schedule()))
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--rel_tol", type=float, default=1.0e-8)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, required=True)
    args = parser.parse_args()
    run_case(
        level=args.level,
        schedule=step_schedule(args.schedule),
        tol=args.tol,
        rel_tol=args.rel_tol,
        line_search=bool(args.line_search),
        out_json=args.out,
        state_out=args.state_out,
    )


if __name__ == "__main__":
    main()
