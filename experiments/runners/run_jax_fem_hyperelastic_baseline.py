#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.benchmark.state_export import export_hyperelasticity_state_npz
from src.core.serial.jax_diff import EnergyDerivator
from src.core.serial.minimizers import newton
from src.core.serial.sparse_solvers import HessSolverGenerator
from src.problems.hyperelasticity.jax.jax_energy import J
from src.problems.hyperelasticity.jax.mesh import MeshHyperElasticity3D

from experiments.analysis.hyperelastic_companion_common import (
    centerline_profile,
    compute_energy_from_full_coordinates,
    full_coordinates_from_free_vector,
    load_hyperelastic_case,
    max_displacement_norm,
    prescribe_right_face_translation,
    relative_l2,
    step_schedule,
)


DEFAULT_ENV_PYTHON = REPO_ROOT / "tmp_work" / "jax_fem_0_0_10_py312" / "bin" / "python"
DEFAULT_MAIN_SITE = (
    REPO_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "jax_fem_hyperelastic_baseline"
WORKER_SCRIPT = REPO_ROOT / "experiments" / "analysis" / "jax_fem_hyperelastic_worker.py"


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _suggest_env_setup() -> str:
    return (
        "Create the isolated baseline env with:\n"
        "./.venv/bin/python -m venv tmp_work/jax_fem_0_0_10_py312\n"
        "tmp_work/jax_fem_0_0_10_py312/bin/pip install --upgrade pip setuptools wheel\n"
        "tmp_work/jax_fem_0_0_10_py312/bin/pip install 'jax-fem==0.0.10' 'jax[cpu]' meshio h5py pyfiglet fenics-basix gmsh"
    )


def _maintained_run(
    *,
    level: int,
    schedule: list[float],
    out_json: Path,
    state_out: Path,
    tolf: float,
    tolg: float,
    tolg_rel: float,
    tolx_rel: float,
    tolx_abs: float,
    linesearch_tol: float,
    maxit: int,
) -> None:
    mesh = MeshHyperElasticity3D(mesh_level=level)
    params, adjacency, u_init = mesh.get_data_jax()
    params = dict(params)
    coords_ref = np.asarray(mesh.params["nodes2coord"], dtype=np.float64)
    tetrahedra = np.asarray(mesh.params["elems2nodes"], dtype=np.int32)
    freedofs = np.asarray(mesh.params["dofsMinim"], dtype=np.int64).reshape((-1,))
    u0_reference = np.asarray(params["u_0"], dtype=np.float64).reshape((-1,))
    right_nodes = np.asarray(load_hyperelastic_case(level)["right_nodes"], dtype=np.int64)
    energy = EnergyDerivator(J, params, adjacency, jnp.asarray(u_init, dtype=jnp.float64))
    current_u = np.asarray(u_init, dtype=np.float64).copy()
    rows: list[dict[str, object]] = []
    total_wall = 0.0
    final_coords = coords_ref.copy()

    for step_id, displacement_x in enumerate(schedule, start=1):
        u0_step = prescribe_right_face_translation(
            u0_reference,
            coords_ref,
            displacement_x,
            right_nodes=right_nodes,
        )
        params["u_0"] = jnp.asarray(u0_step, dtype=jnp.float64)
        energy.params = params
        f, df, ddf = energy.get_derivatives()
        hess_solver = HessSolverGenerator(ddf=ddf, solver_type="direct")
        result = newton(
            f,
            df,
            hess_solver,
            current_u,
            tolf=tolf,
            tolg=tolg,
            tolg_rel=tolg_rel,
            tolx_rel=tolx_rel,
            tolx_abs=tolx_abs,
            linesearch_tol=linesearch_tol,
            maxit=maxit,
            require_all_convergence=False,
            verbose=False,
            save_history=True,
            save_linear_timing=True,
        )
        current_u = np.asarray(result["x"], dtype=np.float64).copy()
        x_full = full_coordinates_from_free_vector(current_u, u0_step, freedofs)
        final_coords = x_full.reshape((-1, 3))
        displacement = final_coords - coords_ref
        linear_timing = result.get("linear_timing", [])
        rows.append(
            {
                "step": int(step_id),
                "displacement_x": float(displacement_x),
                "wall_time_s": float(result["time"]),
                "nit": int(result["nit"]),
                "linear_iterations": int(sum(int(rec.get("ksp_its", 0)) for rec in linear_timing)),
                "energy": float(compute_energy_from_full_coordinates(x_full, mesh.params)),
                "u_max": float(max_displacement_norm(displacement)),
                "message": str(result["message"]),
            }
        )
        total_wall += float(result["time"])

    final_displacement = final_coords - coords_ref
    centerline = centerline_profile(coords_ref, final_displacement)
    export_hyperelasticity_state_npz(
        state_out,
        coords_ref=coords_ref,
        x_final=final_coords,
        tetrahedra=tetrahedra,
        mesh_level=int(level),
        total_steps=len(schedule),
        energy=(None if not rows else float(rows[-1]["energy"])),
        metadata={"solver_family": "repo_serial_direct"},
    )
    payload = {
        "implementation": "repo_serial_direct",
        "level": int(level),
        "mesh_path": str(load_hyperelastic_case(level)["mesh_path"]),
        "schedule": [float(value) for value in schedule],
        "case_contract": {
            "constitutive_law": "compressible_neo_hookean",
            "mesh_family": "HyperElasticity_level1_tet4",
            "boundary_conditions": "left_clamp_right_face_prescribed_translation_xyz",
        },
        "solver_options": {
            "tolf": float(tolf),
            "tolg": float(tolg),
            "tolg_rel": float(tolg_rel),
            "tolx_rel": float(tolx_rel),
            "tolx_abs": float(tolx_abs),
            "linesearch_tol": float(linesearch_tol),
            "maxit": int(maxit),
            "linear_solver": "direct",
        },
        "total_wall_time_s": float(total_wall),
        "free_dofs": int(freedofs.size),
        "state_npz": str(state_out),
        "rows": rows,
        "final_metrics": {
            "energy": (None if not rows else float(rows[-1]["energy"])),
            "u_max": (None if not rows else float(rows[-1]["u_max"])),
            "centerline_x": centerline["x"].tolist(),
            "centerline_ux": centerline["ux"].tolist(),
        },
    }
    _json_write(out_json, payload)


def _run_subprocess(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _jax_fem_run(
    *,
    env_python: Path,
    main_site: Path,
    level: int,
    schedule: list[float],
    out_json: Path,
    state_out: Path,
    tol: float,
    rel_tol: float,
    line_search: bool,
) -> None:
    if not env_python.exists():
        raise FileNotFoundError(f"Missing JAX-FEM environment python: {env_python}\n{_suggest_env_setup()}")
    env = os.environ.copy()
    extra_paths = [str(REPO_ROOT)]
    if main_site.exists():
        extra_paths.append(str(main_site))
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    cmd = [
        str(env_python),
        str(WORKER_SCRIPT),
        "--level",
        str(level),
        "--out",
        str(out_json),
        "--state-out",
        str(state_out),
        "--tol",
        str(tol),
        "--rel_tol",
        str(rel_tol),
        "--schedule",
        *[str(value) for value in schedule],
    ]
    if line_search:
        cmd.append("--line-search")
    _run_subprocess(cmd, env, REPO_ROOT)


def _timing_medians(rows: list[dict[str, object]]) -> dict[str, float]:
    by_name: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["wall_time_s"])
        if np.isfinite(value):
            by_name.setdefault(str(row["implementation"]), []).append(value)
    return {name: float(np.median(values)) for name, values in by_name.items() if values}


def _build_comparison_summary(
    maintained: dict[str, object],
    jax_fem: dict[str, object],
    timing_rows: list[dict[str, object]],
) -> dict[str, object]:
    maintained_rows = [dict(row) for row in maintained["rows"]]
    jax_rows = [dict(row) for row in jax_fem["rows"]]
    if len(maintained_rows) != len(jax_rows):
        raise ValueError("Maintained and JAX-FEM schedules do not match in length.")

    maintained_state = np.load(maintained["state_npz"])
    jax_state = np.load(jax_fem["state_npz"])
    maintained_disp = np.asarray(maintained_state["displacement"], dtype=np.float64)
    jax_disp = np.asarray(jax_state["displacement"], dtype=np.float64)
    coords_ref = np.asarray(maintained_state["coords_ref"], dtype=np.float64)
    centerline_ref = centerline_profile(coords_ref, maintained_disp)
    centerline_jax = centerline_profile(coords_ref, jax_disp)

    step_rows = []
    for maintained_row, jax_row in zip(maintained_rows, jax_rows):
        step_rows.append(
            {
                "step": int(maintained_row["step"]),
                "displacement_x": float(maintained_row["displacement_x"]),
                "repo_energy": float(maintained_row["energy"]),
                "jax_fem_energy": float(jax_row["energy"]),
                "repo_u_max": float(maintained_row["u_max"]),
                "jax_fem_u_max": float(jax_row["u_max"]),
                "repo_wall_time_s": float(maintained_row["wall_time_s"]),
                "jax_fem_wall_time_s": float(jax_row["wall_time_s"]),
                "energy_rel_diff": abs(float(jax_row["energy"]) - float(maintained_row["energy"]))
                / max(abs(float(maintained_row["energy"])), 1.0e-12),
            }
        )

    repo_umax = np.asarray([row["u_max"] for row in maintained_rows], dtype=np.float64)
    jax_umax = np.asarray([row["u_max"] for row in jax_rows], dtype=np.float64)
    energy_ref = float(maintained_rows[-1]["energy"])
    energy_cmp = float(jax_rows[-1]["energy"])
    energy_rel_diff = abs(energy_cmp - energy_ref) / max(abs(energy_ref), 1.0e-12)
    field_l2 = relative_l2(maintained_disp, jax_disp)
    centerline_l2 = relative_l2(centerline_ref["ux"], centerline_jax["ux"])
    umax_curve_l2 = relative_l2(repo_umax, jax_umax)
    fairness_checks = {
        "same_mesh_path": str(maintained["mesh_path"]) == str(jax_fem["mesh_path"]),
        "same_schedule": [float(v) for v in maintained["schedule"]] == [float(v) for v in jax_fem["schedule"]],
        "same_constitutive_law": str(maintained["case_contract"]["constitutive_law"])
        == str(jax_fem["case_contract"]["constitutive_law"]),
        "energy_rel_diff_le_5pct": bool(energy_rel_diff <= 5.0e-2),
        "field_relative_l2_le_5pct": bool(field_l2 <= 5.0e-2),
        "centerline_relative_l2_le_5pct": bool(centerline_l2 <= 5.0e-2),
        "umax_curve_relative_l2_le_5pct": bool(umax_curve_l2 <= 5.0e-2),
    }
    fairness_passed = all(bool(value) for value in fairness_checks.values())
    return {
        "case": {
            "name": "hyperelasticity_tensile_companion",
            "level": int(maintained["level"]),
            "mesh_path": str(maintained["mesh_path"]),
            "schedule": [float(v) for v in maintained["schedule"]],
        },
        "implementations": [
            {
                "name": str(maintained["implementation"]),
                "state_npz": str(maintained["state_npz"]),
                "total_wall_time_s": float(maintained["total_wall_time_s"]),
            },
            {
                "name": str(jax_fem["implementation"]),
                "state_npz": str(jax_fem["state_npz"]),
                "total_wall_time_s": float(jax_fem["total_wall_time_s"]),
            },
        ],
        "timing_rows": timing_rows,
        "timing_medians_s": _timing_medians(timing_rows),
        "step_rows": step_rows,
        "final_metrics": {
            "energy_rel_diff": float(energy_rel_diff),
            "field_relative_l2": float(field_l2),
            "centerline_relative_l2": float(centerline_l2),
            "umax_curve_relative_l2": float(umax_curve_l2),
        },
        "fairness_gate": {
            "checks": fairness_checks,
            "passed": bool(fairness_passed),
            "policy": (
                "Include in the manuscript only if the exact mesh/material/schedule contract holds "
                "and the final energy, full-field displacement, centerline, and u_max curves all stay within 5%."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAX-FEM hyperelastic companion baseline package.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--schedule", type=float, nargs="+", default=list(step_schedule()))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--jax-fem-python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--main-site-packages", type=Path, default=DEFAULT_MAIN_SITE)
    parser.add_argument("--maintained-tolf", type=float, default=1.0e-6)
    parser.add_argument("--maintained-tolg", type=float, default=1.0e-6)
    parser.add_argument("--maintained-tolg-rel", type=float, default=0.0)
    parser.add_argument("--maintained-tolx-rel", type=float, default=1.0e-8)
    parser.add_argument("--maintained-tolx-abs", type=float, default=1.0e-10)
    parser.add_argument("--maintained-linesearch-tol", type=float, default=1.0e-4)
    parser.add_argument("--maintained-maxit", type=int, default=40)
    parser.add_argument("--jax-fem-tol", type=float, default=1.0e-6)
    parser.add_argument("--jax-fem-rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--jax-fem-line-search", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = step_schedule(args.schedule)

    manifest = {
        "case_name": "jax_fem_hyperelastic_baseline",
        "level": int(args.level),
        "schedule": schedule,
        "jax_fem_python": str(args.jax_fem_python),
        "main_site_packages": str(args.main_site_packages),
        "maintained_solver": "repo_serial_direct",
        "jax_fem_solver": "jax_fem_umfpack_serial",
        "warmup_runs": int(args.warmup_runs),
        "timing_repeats": int(args.timing_repeats),
        "artifacts": {
            "maintained_output": str(out_dir / "parity" / "repo_serial_direct.json"),
            "maintained_state": str(out_dir / "parity" / "repo_serial_direct_state.npz"),
            "jax_fem_output": str(out_dir / "parity" / "jax_fem_umfpack_serial.json"),
            "jax_fem_state": str(out_dir / "parity" / "jax_fem_umfpack_serial_state.npz"),
            "comparison_summary": str(out_dir / "comparison_summary.json"),
        },
    }
    _json_write(out_dir / "run_manifest.json", manifest)

    maintained_output = Path(manifest["artifacts"]["maintained_output"])
    maintained_state = Path(manifest["artifacts"]["maintained_state"])
    jax_output = Path(manifest["artifacts"]["jax_fem_output"])
    jax_state = Path(manifest["artifacts"]["jax_fem_state"])

    _maintained_run(
        level=args.level,
        schedule=schedule,
        out_json=maintained_output,
        state_out=maintained_state,
        tolf=args.maintained_tolf,
        tolg=args.maintained_tolg,
        tolg_rel=args.maintained_tolg_rel,
        tolx_rel=args.maintained_tolx_rel,
        tolx_abs=args.maintained_tolx_abs,
        linesearch_tol=args.maintained_linesearch_tol,
        maxit=args.maintained_maxit,
    )
    _jax_fem_run(
        env_python=args.jax_fem_python,
        main_site=args.main_site_packages,
        level=args.level,
        schedule=schedule,
        out_json=jax_output,
        state_out=jax_state,
        tol=args.jax_fem_tol,
        rel_tol=args.jax_fem_rel_tol,
        line_search=bool(args.jax_fem_line_search),
    )

    timing_rows: list[dict[str, object]] = []
    for implementation in ("repo_serial_direct", "jax_fem_umfpack_serial"):
        for repeat_idx in range(args.warmup_runs):
            timing_rows.append(
                {
                    "implementation": implementation,
                    "phase": "warmup",
                    "repeat": int(repeat_idx + 1),
                    "wall_time_s": float("nan"),
                }
            )

    for repeat_idx in range(args.timing_repeats):
        maintained_repeat = out_dir / "timing" / f"repo_serial_direct_repeat{repeat_idx + 1}.json"
        maintained_state_repeat = out_dir / "timing" / f"repo_serial_direct_repeat{repeat_idx + 1}.npz"
        _maintained_run(
            level=args.level,
            schedule=schedule,
            out_json=maintained_repeat,
            state_out=maintained_state_repeat,
            tolf=args.maintained_tolf,
            tolg=args.maintained_tolg,
            tolg_rel=args.maintained_tolg_rel,
            tolx_rel=args.maintained_tolx_rel,
            tolx_abs=args.maintained_tolx_abs,
            linesearch_tol=args.maintained_linesearch_tol,
            maxit=args.maintained_maxit,
        )
        timing_rows.append(
            {
                "implementation": "repo_serial_direct",
                "phase": "measured",
                "repeat": int(repeat_idx + 1),
                "wall_time_s": float(_load_json(maintained_repeat)["total_wall_time_s"]),
            }
        )

        jax_repeat = out_dir / "timing" / f"jax_fem_umfpack_serial_repeat{repeat_idx + 1}.json"
        jax_state_repeat = out_dir / "timing" / f"jax_fem_umfpack_serial_repeat{repeat_idx + 1}.npz"
        _jax_fem_run(
            env_python=args.jax_fem_python,
            main_site=args.main_site_packages,
            level=args.level,
            schedule=schedule,
            out_json=jax_repeat,
            state_out=jax_state_repeat,
            tol=args.jax_fem_tol,
            rel_tol=args.jax_fem_rel_tol,
            line_search=bool(args.jax_fem_line_search),
        )
        timing_rows.append(
            {
                "implementation": "jax_fem_umfpack_serial",
                "phase": "measured",
                "repeat": int(repeat_idx + 1),
                "wall_time_s": float(_load_json(jax_repeat)["total_wall_time_s"]),
            }
        )

    summary = _build_comparison_summary(_load_json(maintained_output), _load_json(jax_output), timing_rows)
    _json_write(out_dir / "comparison_summary.json", summary)


if __name__ == "__main__":
    main()
