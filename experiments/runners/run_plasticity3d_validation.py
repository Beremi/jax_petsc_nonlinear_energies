#!/usr/bin/env python3
"""Build the Plasticity3D validation manifest and run the fixed-lambda source-reference campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
BACKEND_MIX_CASE = REPO_ROOT / "experiments" / "runners" / "run_plasticity3d_backend_mix_case.py"
SOURCE_FIXED = REPO_ROOT / "experiments" / "runners" / "source_fixed_lambda_3d_impl.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_LAYER1A_SOURCE_BRANCH = (
    REPO_ROOT
    / "tmp"
    / "source_compare"
    / "slope_stability_octave_ref"
    / "slope_stability"
    / "artifacts"
    / "compare_direct_branch_lambda1p6"
)
DEFAULT_LAYER1A_JAX_BRANCH = REPO_ROOT / "artifacts" / "raw_results" / "debug" / "p2_direct_branch_lambda1p6_merged"
DEFAULT_LAYER1B_REPORT = (
    DEFAULT_SOURCE_ROOT
    / "benchmarks"
    / "run_3D_hetero_seepage_SSR_comsol_capture"
    / "archive"
    / "report.md"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_validation"
DEFAULT_SCHEDULE = (1.0, 1.2, 1.4, 1.5, 1.55)
SUMMARY_NAME = "validation_manifest.json"
RUNNER_NAME = "plasticity3d_validation"

DEFAULT_ACCEPTED_LAYER1A = {
    "lambda_schedule": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    "work_relative_difference": 3.877021e-05,
    "displacement_relative_l2": 3.517247e-03,
    "deviatoric_strain_relative_l2": 8.72000570313859e-03,
}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _source_env(source_root: Path) -> dict[str, str]:
    env = _base_env()
    repo_path = str(REPO_ROOT.resolve())
    source_path = str((source_root / "src").resolve())
    current = env.get("PYTHONPATH", "")
    parts = [repo_path, source_path]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
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
            cwd=cwd,
            env=env,
            check=False,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
    return int(proc.returncode), float(perf_counter() - started)


def _maintained_command(
    *,
    source_root: Path,
    case_dir: Path,
    output_json: Path,
    state_npz: Path,
    lambda_target: float,
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
        str(BACKEND_MIX_CASE),
        "--assembly-backend",
        "local_constitutiveAD",
        "--solver-backend",
        "local",
        "--source-root",
        str(source_root),
        "--mesh-name",
        "hetero_ssr_L1",
        "--elem-degree",
        "2",
        "--constraint-variant",
        "glued_bottom",
        "--lambda-target",
        str(float(lambda_target)),
        "--stop-tol",
        str(float(stop_tol)),
        "--maxit",
        str(int(maxit)),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(output_json),
        "--state-out",
        str(state_npz),
    ]


def _source_command(
    *,
    source_root: Path,
    case_dir: Path,
    output_json: Path,
    lambda_target: float,
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
        str(SOURCE_FIXED),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(output_json),
        "--mesh-path",
        str(source_root / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"),
        "--mesh-boundary-type",
        "1",
        "--elem-type",
        "P2",
        "--node-ordering",
        "block_xyz",
        "--lambda-target",
        str(float(lambda_target)),
        "--solver-type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc-backend",
        "pmg_shell",
        "--stopping-criterion",
        "relative_correction",
        "--stopping-tol",
        str(float(stop_tol)),
        "--it-newt-max",
        str(int(maxit)),
        "--linear-tolerance",
        "1e-2",
        "--linear-max-iter",
        "100",
        "--threads",
        "1",
        "--elastic-initial-guess",
        "--no-write-debug-bundle",
        "--write-history-json",
        "--no-write-solution-vtu",
        "--no-write-plots",
        "--quiet",
    ]


def _build_layer2_row(
    *,
    lambda_target: float,
    maintained_dir: Path,
    source_dir: Path,
    maintained_output: Path,
    source_output: Path,
    maintained_state: Path,
    maintained_ranks: int,
    source_reference_ranks: int,
    stop_tol: float,
    maxit: int,
    source_root: Path,
) -> dict[str, object]:
    row = {
        "lambda_value": float(lambda_target),
        "maintained_ranks": int(maintained_ranks),
        "source_reference_ranks": int(source_reference_ranks),
        "stop_tol": float(stop_tol),
        "maxit": int(maxit),
        "maintained": {
            "case_dir": _repo_rel(maintained_dir),
            "output_json": _repo_rel(maintained_output),
            "state_npz": _repo_rel(maintained_state),
            "stdout_path": _repo_rel(maintained_dir / "stdout.txt"),
            "stderr_path": _repo_rel(maintained_dir / "stderr.txt"),
            "command": "",
            "exit_code": None,
            "wall_time_s": None,
        },
        "source_reference": {
            "case_dir": _repo_rel(source_dir),
            "output_json": _repo_rel(source_output),
            "petsc_run_npz": _repo_rel(source_dir / "data" / "petsc_run.npz"),
            "history_json": _repo_rel(source_dir / "exports" / "history.json"),
            "run_info_json": _repo_rel(source_dir / "data" / "run_info.json"),
            "stdout_path": _repo_rel(source_dir / "stdout.txt"),
            "stderr_path": _repo_rel(source_dir / "stderr.txt"),
            "command": "",
            "exit_code": None,
            "wall_time_s": None,
            "source_root": _repo_rel(source_root),
        },
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--layer1a-source-branch", type=Path, default=DEFAULT_LAYER1A_SOURCE_BRANCH)
    parser.add_argument("--layer1a-jax-branch", type=Path, default=DEFAULT_LAYER1A_JAX_BRANCH)
    parser.add_argument("--layer1b-report", type=Path, default=DEFAULT_LAYER1B_REPORT)
    parser.add_argument("--maintained-ranks", type=int, default=1)
    parser.add_argument("--source-reference-ranks", type=int, default=8)
    parser.add_argument("--stop-tol", type=float, default=2.0e-3)
    parser.add_argument("--maxit", type=int, default=80)
    parser.add_argument("--schedule", type=float, nargs="+", default=list(DEFAULT_SCHEDULE))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    source_root = Path(args.source_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "runner": RUNNER_NAME,
        "source_root": _repo_rel(source_root),
        "out_dir": _repo_rel(out_dir),
        "validation_contract": {
            "benchmark_family": "Plasticity3D",
            "canonical_case": "glued-bottom heterogeneous P2(L1)",
            "source_mesh_boundary_type": 1,
            "schedule": [float(v) for v in args.schedule],
            "maintained_ranks": int(args.maintained_ranks),
            "source_reference_ranks": int(args.source_reference_ranks),
            "stop_metric": "relative_correction",
            "stop_tol": float(args.stop_tol),
            "maxit": int(args.maxit),
            "acceptance_targets": {
                "highest_successful_lambda_relative_difference_max": 0.03,
                "umax_curve_relative_l2_max": 0.05,
                "endpoint_displacement_relative_l2_max": 0.10,
            },
        },
        "layer1a": {
            "kind": "exact_source_faithfulness",
            "source_branch_dir": _repo_rel(Path(args.layer1a_source_branch).resolve()),
            "jax_branch_dir": _repo_rel(Path(args.layer1a_jax_branch).resolve()),
            "accepted_baseline": dict(DEFAULT_ACCEPTED_LAYER1A),
        },
        "layer1b": {
            "kind": "published_source_family_triangulation",
            "report_md": _repo_rel(Path(args.layer1b_report).resolve()),
        },
        "layer2": {
            "kind": "fixed_lambda_source_operator_validation",
            "rows": [],
        },
    }

    layer2_rows: list[dict[str, object]] = []
    for lambda_target in [float(v) for v in args.schedule]:
        slug = f"lambda_{lambda_target:.3f}".replace(".", "p")
        maintained_dir = out_dir / "layer2" / slug / "maintained"
        source_dir = out_dir / "layer2" / slug / "source_reference"
        maintained_output = maintained_dir / "output.json"
        maintained_state = maintained_dir / "state.npz"
        source_output = source_dir / "output.json"
        row = _build_layer2_row(
            lambda_target=float(lambda_target),
            maintained_dir=maintained_dir,
            source_dir=source_dir,
            maintained_output=maintained_output,
            source_output=source_output,
            maintained_state=maintained_state,
            maintained_ranks=int(args.maintained_ranks),
            source_reference_ranks=int(args.source_reference_ranks),
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
            source_root=source_root,
        )

        maintained_cmd = _maintained_command(
            source_root=source_root,
            case_dir=maintained_dir,
            output_json=maintained_output,
            state_npz=maintained_state,
            lambda_target=float(lambda_target),
            ranks=int(args.maintained_ranks),
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
        )
        row["maintained"]["command"] = shlex.join(maintained_cmd)
        if args.force or not (maintained_output.exists() and maintained_state.exists()):
            exit_code, wall = _run_command(
                cmd=maintained_cmd,
                cwd=REPO_ROOT,
                env=_base_env(),
                stdout_path=maintained_dir / "stdout.txt",
                stderr_path=maintained_dir / "stderr.txt",
            )
        else:
            exit_code, wall = 0, float("nan")
        row["maintained"]["exit_code"] = int(exit_code)
        row["maintained"]["wall_time_s"] = float(wall)

        source_cmd = _source_command(
            source_root=source_root,
            case_dir=source_dir,
            output_json=source_output,
            lambda_target=float(lambda_target),
            ranks=int(args.source_reference_ranks),
            stop_tol=float(args.stop_tol),
            maxit=int(args.maxit),
        )
        row["source_reference"]["command"] = shlex.join(source_cmd)
        if args.force or not (
            source_output.exists()
            and (source_dir / "data" / "petsc_run.npz").exists()
            and (source_dir / "data" / "run_info.json").exists()
        ):
            exit_code, wall = _run_command(
                cmd=source_cmd,
                cwd=REPO_ROOT,
                env=_source_env(source_root),
                stdout_path=source_dir / "stdout.txt",
                stderr_path=source_dir / "stderr.txt",
            )
        else:
            exit_code, wall = 0, float("nan")
        row["source_reference"]["exit_code"] = int(exit_code)
        row["source_reference"]["wall_time_s"] = float(wall)
        layer2_rows.append(row)

    manifest["layer2"]["rows"] = layer2_rows
    _write_json(out_dir / SUMMARY_NAME, manifest)
    print(out_dir / SUMMARY_NAME)


if __name__ == "__main__":
    main()
