#!/usr/bin/env python3
"""Run the maintained topology docs campaign and summarize its outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

THREAD_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}

SERIAL_REFERENCE_ARGS = [
    "--nx", "192",
    "--ny", "96",
    "--length", "2.0",
    "--height", "1.0",
    "--traction", "1.0",
    "--load_fraction", "0.2",
    "--fixed_pad_cells", "16",
    "--load_pad_cells", "16",
    "--volume_fraction_target", "0.4",
    "--theta_min", "0.001",
    "--solid_latent", "10.0",
    "--young", "1.0",
    "--poisson", "0.3",
    "--alpha_reg", "0.005",
    "--ell_pf", "0.08",
    "--mu_move", "0.01",
    "--beta_lambda", "12.0",
    "--volume_penalty", "10.0",
    "--p_start", "1.0",
    "--p_max", "4.0",
    "--p_increment", "0.5",
    "--continuation_interval", "20",
    "--outer_maxit", "180",
    "--outer_tol", "0.02",
    "--volume_tol", "0.001",
    "--mechanics_maxit", "200",
    "--design_maxit", "400",
    "--tolf", "1e-6",
    "--tolg", "1e-3",
    "--ksp_rtol", "1e-2",
    "--ksp_max_it", "80",
    "--save_outer_state_history",
    "--quiet",
]

DIRECT_PARALLEL_ARGS = [
    "--length", "2.0",
    "--height", "1.0",
    "--traction", "1.0",
    "--load_fraction", "0.2",
    "--fixed_pad_cells", "16",
    "--load_pad_cells", "16",
    "--volume_fraction_target", "0.4",
    "--theta_min", "1e-3",
    "--solid_latent", "10.0",
    "--young", "1.0",
    "--poisson", "0.3",
    "--alpha_reg", "0.005",
    "--ell_pf", "0.08",
    "--mu_move", "0.01",
    "--beta_lambda", "12.0",
    "--volume_penalty", "10.0",
    "--p_start", "1.0",
    "--p_max", "4.0",
    "--p_increment", "0.5",
    "--continuation_interval", "20",
    "--outer_maxit", "180",
    "--outer_tol", "0.02",
    "--volume_tol", "0.001",
    "--stall_theta_tol", "1e-6",
    "--stall_p_min", "4.0",
    "--design_maxit", "20",
    "--tolf", "1e-6",
    "--tolg", "1e-3",
    "--linesearch_tol", "0.1",
    "--linesearch_relative_to_bound",
    "--design_gd_line_search", "golden_adaptive",
    "--design_gd_adaptive_window_scale", "2.0",
    "--mechanics_ksp_type", "fgmres",
    "--mechanics_pc_type", "gamg",
    "--mechanics_ksp_rtol", "1e-4",
    "--mechanics_ksp_max_it", "100",
    "--quiet",
    "--save_outer_state_history",
]

MESH_TIMING_CASES = (
    {"nx": 192, "ny": 96, "pads": 8},
    {"nx": 384, "ny": 192, "pads": 16},
    {"nx": 768, "ny": 384, "pads": 32},
)

MESH_TIMING_ARGS_BASE = [
    "--length", "2.0",
    "--height", "1.0",
    "--traction", "1.0",
    "--load_fraction", "0.2",
    "--volume_fraction_target", "0.4",
    "--theta_min", "1e-6",
    "--solid_latent", "10.0",
    "--young", "1.0",
    "--poisson", "0.3",
    "--alpha_reg", "0.005",
    "--ell_pf", "0.08",
    "--mu_move", "0.01",
    "--beta_lambda", "12.0",
    "--volume_penalty", "10.0",
    "--p_start", "1.0",
    "--p_max", "10.0",
    "--p_increment", "0.2",
    "--continuation_interval", "1",
    "--outer_maxit", "2000",
    "--outer_tol", "0.02",
    "--volume_tol", "0.001",
    "--stall_theta_tol", "1e-6",
    "--stall_p_min", "4.0",
    "--design_maxit", "20",
    "--tolf", "1e-6",
    "--tolg", "1e-3",
    "--linesearch_tol", "0.1",
    "--linesearch_relative_to_bound",
    "--design_gd_line_search", "golden_adaptive",
    "--design_gd_adaptive_window_scale", "2.0",
    "--mechanics_ksp_type", "fgmres",
    "--mechanics_pc_type", "gamg",
    "--mechanics_ksp_rtol", "1e-4",
    "--mechanics_ksp_max_it", "100",
    "--quiet",
]


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_command(argv: list[str]) -> str:
    parts = []
    for part in argv:
        text = str(part)
        if text == str(PYTHON):
            text = "./.venv/bin/python"
        elif text.startswith(str(REPO_ROOT) + "/"):
            text = text[len(str(REPO_ROOT)) + 1 :]
        parts.append(text)
    return shlex.join(parts)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout_s: float = 21600.0) -> dict[str, Any]:
    env_map = os.environ.copy()
    env_map.update(env or {})
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env_map,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - t0
    return {
        "argv": argv,
        "command": _normalize_command(argv),
        "exit_code": int(proc.returncode),
        "elapsed_s": elapsed,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def _save_run_logs(run_info: dict[str, Any], log_prefix: Path) -> dict[str, Any]:
    stdout_path = log_prefix.with_suffix(".stdout.txt")
    stderr_path = log_prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(run_info.pop("stdout"), encoding="utf-8")
    stderr_path.write_text(run_info.pop("stderr"), encoding="utf-8")
    run_info["stdout_path"] = _repo_rel(stdout_path)
    run_info["stderr_path"] = _repo_rel(stderr_path)
    return run_info


def _final_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    final = dict(payload.get("final_metrics", {}))
    return {
        "result": str(payload.get("result", "unknown")),
        "wall_time_s": float(payload.get("time", 0.0)),
        "solve_time_s": float(payload.get("time", 0.0) - payload.get("setup_time", 0.0)),
        "setup_time_s": float(payload.get("setup_time", 0.0)),
        "outer_iterations": int(final.get("outer_iterations", 0)),
        "final_p_penal": float(final.get("final_p_penal", math.nan)),
        "final_compliance": float(final.get("final_compliance", math.nan)),
        "final_volume_fraction": float(final.get("final_volume_fraction", math.nan)),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _run_serial_report(base_dir: Path) -> dict[str, Any]:
    asset_dir = base_dir / "serial_reference"
    report_path = asset_dir / "report.md"
    argv = [
        str(PYTHON),
        "-u",
        "experiments/analysis/generate_report_assets.py",
        "--asset-dir",
        str(asset_dir),
        "--report-path",
        str(report_path),
    ]
    run_info = _save_run_logs(_run(argv, env=THREAD_ENV), asset_dir / "serial_reference")
    (asset_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
    if run_info["exit_code"] != 0:
        raise RuntimeError("serial_reference failed")
    result = _read_json(asset_dir / "report_run.json")
    summary = {"command": run_info["command"], **_final_metrics(result)}
    _write_json(asset_dir / "summary.json", summary)
    return summary


def _run_direct_comparison(base_dir: Path) -> dict[str, Any]:
    root = base_dir / "direct_comparison"
    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    def add_row(case_id: str, implementation: str, mpi_ranks: int, metrics: dict[str, Any], output_json: Path, command: str) -> None:
        rows.append(
            {
                "case_id": case_id,
                "implementation": implementation,
                "mpi_ranks": int(mpi_ranks),
                "result": metrics["result"],
                "wall_time_s": metrics["wall_time_s"],
                "final_compliance": metrics["final_compliance"],
                "final_volume_fraction": metrics["final_volume_fraction"],
                "outer_iterations": metrics["outer_iterations"],
                "json_path": _repo_rel(output_json),
                "command": command,
            }
        )

    # Serial reference repeated 3x on 192x96.
    case_id = "nx192_ny96_np1"
    for run_idx in range(1, 4):
        run_dir = raw_root / case_id / "jax_serial" / f"run{run_idx:02d}"
        output_json = run_dir / "output.json"
        state_npz = run_dir / "state.npz"
        if not output_json.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                str(PYTHON),
                "-u",
                "src/problems/topology/jax/solve_topopt_jax.py",
                *SERIAL_REFERENCE_ARGS,
                "--json_out",
                str(output_json),
                "--state_out",
                str(state_npz),
            ]
            run_info = _save_run_logs(_run(argv, env=THREAD_ENV), run_dir / "run")
            (run_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
            if run_info["exit_code"] != 0:
                raise RuntimeError(f"direct comparison serial run {run_idx} failed")
        payload = _read_json(output_json)
        add_row(case_id, "jax_serial", 1, _final_metrics(payload), output_json, (run_dir / "command.txt").read_text(encoding="utf-8").strip())

    # Parallel repeated 3x for np=1,2,4 on 192x96.
    for ranks in (1, 2, 4):
        case_id = f"nx192_ny96_np{ranks}"
        for run_idx in range(1, 4):
            run_dir = raw_root / case_id / "jax_parallel" / f"run{run_idx:02d}"
            output_json = run_dir / "output.json"
            state_npz = run_dir / "state.npz"
            if not output_json.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                argv = [
                    "mpiexec",
                    "-n",
                    str(ranks),
                    str(PYTHON),
                    "-u",
                    "src/problems/topology/jax/solve_topopt_parallel.py",
                    "--nx",
                    "192",
                    "--ny",
                    "96",
                    *DIRECT_PARALLEL_ARGS,
                    "--json_out",
                    str(output_json),
                    "--state_out",
                    str(state_npz),
                ]
                run_info = _save_run_logs(_run(argv, env=THREAD_ENV), run_dir / "run")
                (run_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
                if run_info["exit_code"] != 0:
                    raise RuntimeError(f"direct comparison parallel np={ranks} run {run_idx} failed")
            payload = _read_json(output_json)
            add_row(
                case_id,
                "jax_parallel",
                ranks,
                _final_metrics(payload),
                output_json,
                (run_dir / "command.txt").read_text(encoding="utf-8").strip(),
            )

    # Median summary in the same structure consumed by docs data.
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], row["implementation"], row["mpi_ranks"]), []).append(row)
    for (case_id, implementation, mpi_ranks), group in sorted(grouped.items()):
        statuses = {row["result"] for row in group}
        summary_rows.append(
            {
                "case_id": case_id,
                "implementation": implementation,
                "mpi_ranks": int(mpi_ranks),
                "median_wall_time_s": _median([float(row["wall_time_s"]) for row in group]),
                "median_final_compliance": _median([float(row["final_compliance"]) for row in group]),
                "median_final_volume_fraction": _median([float(row["final_volume_fraction"]) for row in group]),
                "status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
            }
        )

    _write_csv(
        root / "raw_runs.csv",
        rows,
        ["case_id", "implementation", "mpi_ranks", "result", "wall_time_s", "final_compliance", "final_volume_fraction", "outer_iterations", "json_path", "command"],
    )
    _write_csv(
        root / "direct_comparison.csv",
        summary_rows,
        ["case_id", "implementation", "mpi_ranks", "median_wall_time_s", "median_final_compliance", "median_final_volume_fraction", "status"],
    )
    payload = {"rows": rows, "median_rows": summary_rows}
    _write_json(root / "summary.json", payload)
    return payload


def _run_mesh_timing(base_dir: Path) -> dict[str, Any]:
    root = base_dir / "mesh_timing"
    rows: list[dict[str, Any]] = []
    for case in MESH_TIMING_CASES:
        nx = int(case["nx"])
        ny = int(case["ny"])
        pads = int(case["pads"])
        case_dir = root / f"nx{nx}_ny{ny}_np8"
        output_json = case_dir / "output.json"
        state_npz = case_dir / "state.npz"
        if not output_json.exists():
            case_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                "mpiexec",
                "-n",
                "8",
                str(PYTHON),
                "-u",
                "src/problems/topology/jax/solve_topopt_parallel.py",
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--fixed_pad_cells",
                str(pads),
                "--load_pad_cells",
                str(pads),
                *MESH_TIMING_ARGS_BASE,
                "--json_out",
                str(output_json),
                "--state_out",
                str(state_npz),
            ]
            run_info = _save_run_logs(_run(argv, env=THREAD_ENV), case_dir / "run")
            (case_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
            if run_info["exit_code"] != 0:
                raise RuntimeError(f"mesh timing {nx}x{ny} failed")
        payload = _read_json(output_json)
        mesh = payload["mesh"]
        metrics = _final_metrics(payload)
        rows.append(
            {
                "solver": "jax_parallel",
                "mesh_label": f"{mesh['nx']}x{mesh['ny']}",
                "nx": int(mesh["nx"]),
                "ny": int(mesh["ny"]),
                "nprocs": int(payload["nprocs"]),
                "problem_size": int(mesh["displacement_free_dofs"] + mesh["design_free_dofs"]),
                "wall_time_s": metrics["wall_time_s"],
                "solve_time_s": metrics["solve_time_s"],
                "final_compliance": metrics["final_compliance"],
                "final_volume_fraction": metrics["final_volume_fraction"],
                "outer_iterations": metrics["outer_iterations"],
                "result": metrics["result"],
                "json_path": _repo_rel(output_json),
            }
        )
    rows.sort(key=lambda row: int(row["problem_size"]))
    _write_csv(
        root / "mesh_timing_summary.csv",
        rows,
        ["solver", "mesh_label", "nx", "ny", "nprocs", "problem_size", "wall_time_s", "solve_time_s", "final_compliance", "final_volume_fraction", "outer_iterations", "result", "json_path"],
    )
    _write_json(root / "summary.json", {"rows": rows})
    return {"rows": rows}


def _run_parallel_scaling(base_dir: Path) -> dict[str, Any]:
    asset_dir = base_dir / "parallel_scaling"
    report_path = asset_dir / "report.md"
    argv = [
        str(PYTHON),
        "-u",
        "experiments/analysis/generate_parallel_scaling_stallstop_report.py",
        "--asset-dir",
        str(asset_dir),
        "--report-path",
        str(report_path),
    ]
    run_info = _save_run_logs(_run(argv, env=THREAD_ENV), asset_dir / "parallel_scaling")
    (asset_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
    if run_info["exit_code"] != 0:
        raise RuntimeError("parallel scaling failed")
    rows: list[dict[str, Any]] = []
    with (asset_dir / "scaling_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = {"rows": rows, "command": run_info["command"]}
    _write_json(asset_dir / "summary.json", payload)
    return payload


def _run_parallel_final(base_dir: Path) -> dict[str, Any]:
    asset_dir = base_dir / "parallel_final"
    run_json = asset_dir / "parallel_full_run.json"
    state_npz = asset_dir / "parallel_full_state.npz"
    frame_dir = asset_dir / "frames"
    if not run_json.exists():
        asset_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            "mpiexec",
            "-n",
            "32",
            str(PYTHON),
            "-u",
            "src/problems/topology/jax/solve_topopt_parallel.py",
            "--nx",
            "768",
            "--ny",
            "384",
            "--length",
            "2.0",
            "--height",
            "1.0",
            "--traction",
            "1.0",
            "--load_fraction",
            "0.2",
            "--fixed_pad_cells",
            "32",
            "--load_pad_cells",
            "32",
            "--volume_fraction_target",
            "0.4",
            "--theta_min",
            "1e-6",
            "--solid_latent",
            "10.0",
            "--young",
            "1.0",
            "--poisson",
            "0.3",
            "--alpha_reg",
            "0.005",
            "--ell_pf",
            "0.08",
            "--mu_move",
            "0.01",
            "--beta_lambda",
            "12.0",
            "--volume_penalty",
            "10.0",
            "--p_start",
            "1.0",
            "--p_max",
            "10.0",
            "--p_increment",
            "0.2",
            "--continuation_interval",
            "1",
            "--outer_maxit",
            "2000",
            "--outer_tol",
            "0.02",
            "--volume_tol",
            "0.001",
            "--stall_theta_tol",
            "1e-6",
            "--stall_p_min",
            "4.0",
            "--design_maxit",
            "20",
            "--tolf",
            "1e-6",
            "--tolg",
            "1e-3",
            "--linesearch_tol",
            "0.1",
            "--linesearch_relative_to_bound",
            "--design_gd_line_search",
            "golden_adaptive",
            "--design_gd_adaptive_window_scale",
            "2.0",
            "--mechanics_ksp_type",
            "fgmres",
            "--mechanics_pc_type",
            "gamg",
            "--mechanics_ksp_rtol",
            "1e-4",
            "--mechanics_ksp_max_it",
            "100",
            "--quiet",
            "--print_outer_iterations",
            "--save_outer_state_history",
            "--outer_snapshot_stride",
            "2",
            "--outer_snapshot_dir",
            str(frame_dir),
            "--json_out",
            str(run_json),
            "--state_out",
            str(state_npz),
        ]
        run_info = _save_run_logs(_run(argv, env=THREAD_ENV), asset_dir / "parallel_final_solver")
        (asset_dir / "command.txt").write_text(run_info["command"] + "\n", encoding="utf-8")
        if run_info["exit_code"] != 0:
            raise RuntimeError("parallel final solver failed")
    argv = [
        str(PYTHON),
        "-u",
        "experiments/analysis/generate_parallel_full_report.py",
        "--asset_dir",
        str(asset_dir),
        "--report_path",
        str(asset_dir / "report.md"),
    ]
    report_info = _save_run_logs(_run(argv, env=THREAD_ENV), asset_dir / "parallel_final_report")
    if report_info["exit_code"] != 0:
        raise RuntimeError("parallel full report failed")
    command_path = asset_dir / "command.txt"
    if not command_path.exists():
        report_text = (asset_dir / "report.md").read_text(encoding="utf-8")
        solver_command = "cached parallel final solver command unavailable"
        if "```bash\n" in report_text:
            solver_command = report_text.split("```bash\n", 1)[1].split("\n```", 1)[0].strip()
        command_path.write_text(solver_command + "\n", encoding="utf-8")
    payload = {
        "solver_command": command_path.read_text(encoding="utf-8").strip(),
        "report_command": report_info["command"],
        "metrics": _final_metrics(_read_json(run_json)),
    }
    _write_json(asset_dir / "summary.json", payload)
    return payload


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    serial = payload["serial_reference"]
    direct_rows = payload["direct_comparison"]["median_rows"]
    mesh_rows = payload["mesh_timing"]["rows"]
    scaling_rows = payload["parallel_scaling"]["rows"]
    final_metrics = payload["parallel_final"]["metrics"]

    lines = [
        "# Topology Docs Suite Summary",
        "",
        "| Section | Key result |",
        "|---|---|",
        (
            "| serial reference | "
            f"result={serial['result']}, outer={serial['outer_iterations']}, "
            f"C={serial['final_compliance']:.6f}, V={serial['final_volume_fraction']:.6f}, "
            f"wall={serial['wall_time_s']:.3f}s |"
        ),
        (
            "| parallel final | "
            f"result={final_metrics['result']}, outer={final_metrics['outer_iterations']}, "
            f"C={final_metrics['final_compliance']:.6f}, V={final_metrics['final_volume_fraction']:.6f}, "
            f"wall={final_metrics['wall_time_s']:.3f}s |"
        ),
        "",
        "## Direct Comparison Medians",
        "",
        "| case | implementation | ranks | wall [s] | compliance | volume | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in direct_rows:
        lines.append(
            "| {case_id} | {implementation} | {mpi_ranks} | {wall:.6f} | {comp:.6f} | {vol:.6f} | {status} |".format(
                case_id=row["case_id"],
                implementation=row["implementation"],
                mpi_ranks=row["mpi_ranks"],
                wall=float(row["median_wall_time_s"]),
                comp=float(row["median_final_compliance"]),
                vol=float(row["median_final_volume_fraction"]),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Mesh Timing",
            "",
            "| mesh | ranks | wall [s] | compliance | volume | outer | result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in mesh_rows:
        lines.append(
            "| {mesh} | {nprocs} | {wall:.6f} | {comp:.6f} | {vol:.6f} | {outer} | {result} |".format(
                mesh=row["mesh_label"],
                nprocs=row["nprocs"],
                wall=float(row["wall_time_s"]),
                comp=float(row["final_compliance"]),
                vol=float(row["final_volume_fraction"]),
                outer=row["outer_iterations"],
                result=row["result"],
            )
        )
    lines.extend(
        [
            "",
            "## Fine-Grid Strong Scaling",
            "",
            "| ranks | result | outer | p | volume | compliance | wall [s] | solve [s] |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scaling_rows:
        lines.append(
            "| {ranks} | {result} | {outer_iterations} | {final_p} | {final_volume} | {final_compliance} | {wall_time} | {solve_time} |".format(
                ranks=row["ranks"],
                result=row["result"],
                outer_iterations=row["outer_iterations"],
                final_p=row["final_p"],
                final_volume=row["final_volume"],
                final_compliance=row["final_compliance"],
                wall_time=row["wall_time"],
                solve_time=row["solve_time"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reproduction" / "local_topology_docs_suite" / "runs" / "topology",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "serial_reference": _run_serial_report(out_dir),
        "direct_comparison": _run_direct_comparison(out_dir),
        "mesh_timing": _run_mesh_timing(out_dir),
        "parallel_scaling": _run_parallel_scaling(out_dir),
        "parallel_final": _run_parallel_final(out_dir),
    }
    _write_json(out_dir / "summary.json", payload)
    _write_summary_markdown(out_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
