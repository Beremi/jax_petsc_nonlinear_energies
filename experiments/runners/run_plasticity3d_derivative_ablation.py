#!/usr/bin/env python3
"""Run the Plasticity3D derivative-route ablation on one locked case."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
CASE_RUNNER = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_derivative_ablation"
SUMMARY_NAME = "comparison_summary.json"
RUNNER_NAME = "plasticity3d_derivative_ablation"

ROUTES = (
    {
        "name": "element_ad",
        "display_label": "Element AD",
        "assembly_backend": "local",
        "solver_backend": "local",
    },
    {
        "name": "constitutive_ad",
        "display_label": "Constitutive AD",
        "assembly_backend": "local_constitutiveAD",
        "solver_backend": "local",
    },
    {
        "name": "colored_sfd",
        "display_label": "Colored SFD",
        "assembly_backend": "local_sfd",
        "solver_backend": "local",
    },
)

NORMALIZED_ROW_KEYS = (
    "route",
    "display_label",
    "assembly_backend",
    "solver_backend",
    "ranks",
    "measured_runs",
    "status",
    "solver_success",
    "median_wall_time_s",
    "median_solve_time_s",
    "median_nit",
    "median_linear_iterations_total",
    "median_final_metric",
    "final_metric_name",
    "median_energy",
    "median_omega",
    "median_u_max",
    "run_rows",
)


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _run_command(
    *,
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fh:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=_base_env(),
            check=False,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
    return int(proc.returncode), float(perf_counter() - started)


def _build_case_command(
    *,
    source_root: Path,
    route: dict[str, str],
    case_dir: Path,
    output_json: Path,
    ranks: int,
    stop_tol: float,
    maxit: int,
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(ranks),
        str(PYTHON),
        "-u",
        str(CASE_RUNNER),
        "--assembly-backend",
        str(route["assembly_backend"]),
        "--solver-backend",
        str(route["solver_backend"]),
        "--source-root",
        str(source_root),
        "--mesh-name",
        "hetero_ssr_L1",
        "--elem-degree",
        "4",
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        "1.5",
        "--stop-tol",
        str(float(stop_tol)),
        "--maxit",
        str(int(maxit)),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(output_json),
    ]


def _normalize_run(
    *,
    payload: dict,
    run_id: str,
    case_dir: Path,
    output_json: Path,
    stdout_path: Path,
    stderr_path: Path,
    command: list[str],
    wall_time_s: float,
    exit_code: int,
) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "status": str(payload.get("status", "failed")),
        "solver_success": bool(payload.get("solver_success", False)),
        "exit_code": int(exit_code),
        "wall_time_s": float(wall_time_s),
        "solve_time_s": float(payload.get("solve_time", float("nan"))),
        "nit": int(payload.get("nit", 0)),
        "linear_iterations_total": int(payload.get("linear_iterations_total", 0)),
        "final_metric": float(payload.get("final_metric", float("nan"))),
        "final_metric_name": str(payload.get("final_metric_name", "")),
        "energy": float(payload.get("energy", float("nan"))),
        "omega": float(payload.get("omega", float("nan"))),
        "u_max": float(payload.get("u_max", float("nan"))),
        "output_json": _repo_rel(output_json),
        "case_dir": _repo_rel(case_dir),
        "stdout_path": _repo_rel(stdout_path),
        "stderr_path": _repo_rel(stderr_path),
        "command": shlex.join(command),
    }


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(statistics.median(finite))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for route in ROUTES:
        route_dir = out_dir / str(route["name"])
        measured_rows: list[dict[str, object]] = []
        total_runs = int(args.warmup_runs) + int(args.measured_runs)
        for idx in range(total_runs):
            run_kind = "warmup" if idx < int(args.warmup_runs) else "measure"
            run_number = idx + 1 - int(args.warmup_runs) if run_kind == "measure" else idx + 1
            run_slug = f"{run_kind}_{run_number:02d}"
            case_dir = route_dir / run_slug
            output_json = case_dir / "output.json"
            stdout_path = case_dir / "stdout.txt"
            stderr_path = case_dir / "stderr.txt"
            command = _build_case_command(
                source_root=Path(args.source_root).resolve(),
                route=route,
                case_dir=case_dir,
                output_json=output_json,
                ranks=int(args.ranks),
                stop_tol=float(args.stop_tol),
                maxit=int(args.maxit),
            )
            if args.force or not output_json.exists():
                exit_code, wall_time_s = _run_command(
                    cmd=command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            else:
                exit_code, wall_time_s = 0, float("nan")
            payload = _read_json(output_json)
            run_row = _normalize_run(
                payload=payload,
                run_id=run_slug,
                case_dir=case_dir,
                output_json=output_json,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
                wall_time_s=wall_time_s,
                exit_code=exit_code,
            )
            if run_kind == "measure":
                measured_rows.append(run_row)

        route_row = {
            "route": str(route["name"]),
            "display_label": str(route["display_label"]),
            "assembly_backend": str(route["assembly_backend"]),
            "solver_backend": str(route["solver_backend"]),
            "ranks": int(args.ranks),
            "measured_runs": int(args.measured_runs),
            "status": "completed" if all(bool(row["solver_success"]) for row in measured_rows) else "failed",
            "solver_success": bool(all(bool(row["solver_success"]) for row in measured_rows)),
            "median_wall_time_s": _median([float(row["wall_time_s"]) for row in measured_rows]),
            "median_solve_time_s": _median([float(row["solve_time_s"]) for row in measured_rows]),
            "median_nit": _median([float(row["nit"]) for row in measured_rows]),
            "median_linear_iterations_total": _median([float(row["linear_iterations_total"]) for row in measured_rows]),
            "median_final_metric": _median([float(row["final_metric"]) for row in measured_rows]),
            "final_metric_name": str(measured_rows[0]["final_metric_name"]) if measured_rows else "",
            "median_energy": _median([float(row["energy"]) for row in measured_rows]),
            "median_omega": _median([float(row["omega"]) for row in measured_rows]),
            "median_u_max": _median([float(row["u_max"]) for row in measured_rows]),
            "run_rows": measured_rows,
        }
        rows.append(route_row)

    payload = {
        "runner": RUNNER_NAME,
        "source_root": _repo_rel(Path(args.source_root).resolve()),
        "out_dir": _repo_rel(out_dir),
        "ranks": int(args.ranks),
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
        "stop_tol": float(args.stop_tol),
        "maxit": int(args.maxit),
        "row_keys": list(NORMALIZED_ROW_KEYS),
        "rows": rows,
    }
    _write_json(out_dir / SUMMARY_NAME, payload)
    print(out_dir / SUMMARY_NAME)


if __name__ == "__main__":
    main()
