#!/usr/bin/env python3
"""Run Plasticity3D `P4(L1_2)`, lambda=1.55, with MUMPS coarse PMG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from src.core.benchmark.replication import command_text, now_iso, write_command_files

from experiments.runners import run_plasticity3d_backend_mix_compare as mix_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "source_compare" / "slope_stability_petsc4py"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "plasticity3d_p4_l1_2_lambda1p55_mumps_pmg_scaling"
)
RUNNER_NAME = "plasticity3d_p4_l1_2_lambda1p55_mumps_pmg_scaling"
SUMMARY_NAME = "comparison_summary.json"
ASSEMBLY_BACKEND = "local_constitutiveAD"
SOLVER_BACKEND = "local_pmg_mumps"
SOLVER_PROFILE = "same_mesh_p4_p2_p1_redundant_lu_mumps"
CONSTRAINT_VARIANT = "glued_bottom"

ROW_KEYS = tuple(
    list(mix_tools.NORMALIZED_ROW_KEYS)
    + [
        "solver_profile",
        "pmg_coarse_backend",
        "pmg_coarse_ksp_type",
        "pmg_coarse_pc_type",
        "pmg_coarse_redundant_number",
        "pmg_coarse_factor_solver_type",
        "process_wall_time_s",
    ]
)


def _case_id(ranks: int) -> str:
    return f"np{int(ranks)}:{ASSEMBLY_BACKEND}_assembly:{SOLVER_BACKEND}_solver"


def _case_dir(out_dir: Path, ranks: int) -> Path:
    return (
        out_dir
        / "runs"
        / f"np{int(ranks)}"
        / f"solver_{SOLVER_BACKEND}"
        / f"assembly_{ASSEMBLY_BACKEND}"
    )


def _build_env(
    *,
    source_root: Path,
    redundant_number: int,
    factor_solver: str,
    oversubscribe: bool,
) -> dict[str, str]:
    env = mix_tools._mixed_env(source_root)
    env["MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER"] = str(int(redundant_number))
    env["MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER"] = str(factor_solver)
    if bool(oversubscribe):
        env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    return env


def _build_case_command(
    *,
    source_root: Path,
    case_dir: Path,
    result_path: Path,
    ranks: int,
    maxit: int,
    grad_stop_tol: float,
    ksp_rtol: float,
    ksp_max_it: int,
    launcher: str,
    write_state: bool,
) -> list[str]:
    command = [
        str(launcher),
        "-n",
        str(int(ranks)),
        str(mix_tools.PYTHON),
        "-u",
        str(mix_tools.CASE_RUNNER),
        "--assembly-backend",
        ASSEMBLY_BACKEND,
        "--solver-backend",
        SOLVER_BACKEND,
        "--source-root",
        str(source_root),
        "--out-dir",
        str(case_dir),
        "--output-json",
        str(result_path),
        "--mesh-name",
        "hetero_ssr_L1_2",
        "--constraint-variant",
        CONSTRAINT_VARIANT,
        "--lambda-target",
        "1.55",
        "--ksp-rtol",
        str(float(ksp_rtol)),
        "--ksp-max-it",
        str(int(ksp_max_it)),
        "--convergence-mode",
        "gradient_only",
        "--grad-stop-tol",
        str(float(grad_stop_tol)),
        "--stop-tol",
        "0.0",
        "--maxit",
        str(int(maxit)),
        "--line-search",
        "armijo",
        "--armijo-max-ls",
        "40",
    ]
    if bool(write_state):
        command.extend(["--state-out", str(case_dir / "state.npz")])
    return command


def _profile_from_payload(payload: dict[str, object]) -> dict[str, object]:
    profile = dict(payload.get("pmg_linear_profile", {}))
    return {
        "pmg_coarse_backend": str(profile.get("coarse_backend", "redundant_lu")),
        "pmg_coarse_ksp_type": str(profile.get("coarse_ksp_type", "preonly")),
        "pmg_coarse_pc_type": str(profile.get("coarse_pc_type", "redundant")),
        "pmg_coarse_redundant_number": int(profile.get("coarse_redundant_number", 1)),
        "pmg_coarse_factor_solver_type": str(
            profile.get("coarse_factor_solver_type", "mumps")
        ),
    }


def _augment_row(row: dict[str, object], result_path: Path) -> dict[str, object]:
    payload = mix_tools._read_json(result_path)
    augmented = dict(row)
    augmented["solver_profile"] = SOLVER_PROFILE
    augmented.update(_profile_from_payload(payload))
    return augmented


def _failed_row(
    *,
    ranks: int,
    exit_code: int,
    message: str,
    case_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
    command: list[str],
    redundant_number: int,
    factor_solver: str,
) -> dict[str, object]:
    row = mix_tools._failed_row(
        case_id=_case_id(ranks),
        assembly_backend=ASSEMBLY_BACKEND,
        solver_backend=SOLVER_BACKEND,
        ranks=int(ranks),
        exit_code=int(exit_code),
        message=str(message),
        case_dir=case_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
        command=command,
    )
    row["solver_profile"] = SOLVER_PROFILE
    row["pmg_coarse_backend"] = "redundant_lu"
    row["pmg_coarse_ksp_type"] = "preonly"
    row["pmg_coarse_pc_type"] = "redundant"
    row["pmg_coarse_redundant_number"] = int(redundant_number)
    row["pmg_coarse_factor_solver_type"] = str(factor_solver)
    row["process_wall_time_s"] = float("nan")
    return row


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary(
    *,
    summary_path: Path,
    rows_by_case: dict[str, dict[str, object]],
    ranks: list[int],
    source_root: Path,
    maxit: int,
    grad_stop_tol: float,
    ksp_rtol: float,
    ksp_max_it: int,
    redundant_number: int,
    factor_solver: str,
    launcher: str,
) -> None:
    ordered_rows = sorted(
        rows_by_case.values(), key=lambda row: int(row.get("ranks", 10**9))
    )
    payload = {
        "runner": RUNNER_NAME,
        "timestamp_utc": now_iso(),
        "source_root": mix_tools._repo_rel(source_root),
        "out_dir": mix_tools._repo_rel(summary_path.parent),
        "ranks": [int(v) for v in ranks],
        "mesh_name": "hetero_ssr_L1_2",
        "elem_degree": 4,
        "constraint_variant": CONSTRAINT_VARIANT,
        "lambda_target": 1.55,
        "assembly_backend": ASSEMBLY_BACKEND,
        "solver_backend": SOLVER_BACKEND,
        "solver_profile": SOLVER_PROFILE,
        "pmg_strategy": "same_mesh_p4_p2_p1",
        "pmg_coarse_backend": "redundant_lu",
        "pmg_coarse_ksp_type": "preonly",
        "pmg_coarse_pc_type": "redundant",
        "pmg_coarse_redundant_number": int(redundant_number),
        "pmg_coarse_factor_solver_type": str(factor_solver),
        "grad_stop_tol": float(grad_stop_tol),
        "maxit": int(maxit),
        "ksp_rtol": float(ksp_rtol),
        "ksp_max_it": int(ksp_max_it),
        "launcher": str(launcher),
        "row_keys": list(ROW_KEYS),
        "rows": ordered_rows,
    }
    _write_json(summary_path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--launcher", type=str, default="mpiexec")
    parser.add_argument("--maxit", type=int, default=5)
    parser.add_argument("--grad-stop-tol", type=float, default=1.0e-2)
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-1)
    parser.add_argument("--ksp-max-it", type=int, default=100)
    parser.add_argument("--redundant-number", type=int, default=1)
    parser.add_argument("--factor-solver", type=str, default="mumps")
    parser.add_argument("--write-state", action="store_true")
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_NAME
    ranks_list = [int(v) for v in args.ranks]
    env = _build_env(
        source_root=source_root,
        redundant_number=int(args.redundant_number),
        factor_solver=str(args.factor_solver),
        oversubscribe=bool(args.oversubscribe),
    )

    rows_by_case: dict[str, dict[str, object]] = {}
    if bool(args.resume) and summary_path.exists():
        payload = mix_tools._read_json(summary_path)
        for row in payload.get("rows", []):
            if isinstance(row, dict):
                rows_by_case[str(row.get("case_id", ""))] = dict(row)

    dry_plan: list[dict[str, object]] = []
    for ranks in ranks_list:
        case_id = _case_id(ranks)
        case_dir = _case_dir(out_dir, ranks)
        stdout_path = case_dir / "stdout.txt"
        stderr_path = case_dir / "stderr.txt"
        result_path = case_dir / "output.json"
        command = _build_case_command(
            source_root=source_root,
            case_dir=case_dir,
            result_path=result_path,
            ranks=int(ranks),
            maxit=int(args.maxit),
            grad_stop_tol=float(args.grad_stop_tol),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            launcher=str(args.launcher),
            write_state=bool(args.write_state),
        )
        write_command_files(case_dir, command=command, cwd=REPO_ROOT, env=env)

        if bool(args.dry_run):
            dry_plan.append(
                {
                    "case_id": case_id,
                    "ranks": int(ranks),
                    "case_dir": mix_tools._repo_rel(case_dir),
                    "command": command_text(command),
                    "env": {
                        "MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER": env[
                            "MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER"
                        ],
                        "MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER": env[
                            "MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER"
                        ],
                    },
                }
            )
            continue

        existing = rows_by_case.get(case_id)
        if (
            bool(args.resume)
            and existing is not None
            and str(existing.get("status", "")).startswith("completed")
            and result_path.exists()
        ):
            continue

        exit_code, process_wall_time_s = mix_tools._run_command(
            cmd=command,
            cwd=REPO_ROOT,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_path=result_path,
        )

        if exit_code != 0 or not result_path.exists():
            message = (
                mix_tools._tail_text(stderr_path)
                or mix_tools._tail_text(stdout_path)
                or "subprocess failed"
            )
            row = _failed_row(
                ranks=int(ranks),
                exit_code=int(exit_code),
                message=message,
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
                redundant_number=int(args.redundant_number),
                factor_solver=str(args.factor_solver),
            )
        else:
            row = mix_tools._normalize_payload(
                case_id=case_id,
                assembly_backend=ASSEMBLY_BACKEND,
                solver_backend=SOLVER_BACKEND,
                ranks=int(ranks),
                exit_code=int(exit_code),
                case_dir=case_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                result_path=result_path,
                command=command,
            )
            row = _augment_row(row, result_path)
            row["process_wall_time_s"] = float(process_wall_time_s)

        rows_by_case[case_id] = row
        _write_summary(
            summary_path=summary_path,
            rows_by_case=rows_by_case,
            ranks=ranks_list,
            source_root=source_root,
            maxit=int(args.maxit),
            grad_stop_tol=float(args.grad_stop_tol),
            ksp_rtol=float(args.ksp_rtol),
            ksp_max_it=int(args.ksp_max_it),
            redundant_number=int(args.redundant_number),
            factor_solver=str(args.factor_solver),
            launcher=str(args.launcher),
        )

    if bool(args.dry_run):
        print(json.dumps({"runner": RUNNER_NAME, "plan": dry_plan}, indent=2))
        return

    _write_summary(
        summary_path=summary_path,
        rows_by_case=rows_by_case,
        ranks=ranks_list,
        source_root=source_root,
        maxit=int(args.maxit),
        grad_stop_tol=float(args.grad_stop_tol),
        ksp_rtol=float(args.ksp_rtol),
        ksp_max_it=int(args.ksp_max_it),
        redundant_number=int(args.redundant_number),
        factor_solver=str(args.factor_solver),
        launcher=str(args.launcher),
    )
    print(shlex.quote(str(summary_path)))


if __name__ == "__main__":
    main()
