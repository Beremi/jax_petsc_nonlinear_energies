#!/usr/bin/env python3
"""Run and report the current fine-grid parallel scaling study."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLVER = REPO_ROOT / "src" / "problems" / "topology" / "jax" / "solve_topopt_parallel.py"
DEFAULT_ASSET_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "topology_reports"
    / "scaling"
)
RANKS = [1, 2, 4, 8, 16, 32]
USE_CACHE = os.environ.get("TOPOPT_SCALING_USE_CACHE", "0") == "1"

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

CASE_ARGS = [
    "--nx", "768",
    "--ny", "384",
    "--length", "2.0",
    "--height", "1.0",
    "--traction", "1.0",
    "--load_fraction", "0.2",
    "--fixed_pad_cells", "32",
    "--load_pad_cells", "32",
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
    "--mechanics_ksp_rtol", "1e-4",
    "--mechanics_ksp_max_it", "100",
    "--mechanics_ksp_type", "fgmres",
    "--mechanics_pc_type", "gamg",
    "--linesearch_relative_to_bound",
    "--design_gd_line_search", "golden_adaptive",
    "--design_gd_adaptive_window_scale", "2.0",
    "--quiet",
]


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _read_boost_state() -> str:
    candidates = [
        Path("/sys/devices/system/cpu/cpufreq/boost"),
        Path("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = path.read_text().strip()
        if path.name == "boost":
            return "enabled" if value == "1" else "disabled"
        if path.name == "no_turbo":
            return "disabled" if value == "1" else "enabled"
    return "unknown"


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _asset_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _run_case(ranks: int, json_out: Path) -> dict:
    if USE_CACHE and json_out.exists():
        return json.loads(json_out.read_text())

    cmd: list[str] = []
    if ranks > 1:
        cmd.extend(["mpiexec", "-n", str(ranks)])
    cmd.extend([str(PYTHON), str(SOLVER), *CASE_ARGS, "--json_out", str(json_out)])

    env = os.environ.copy()
    env.update(THREAD_ENV)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return json.loads(json_out.read_text())


def _summarize(result: dict) -> dict[str, float | int | str | bool]:
    history = result["history"]
    mech_assemble = np.array([row.get("mechanics_assemble_time", 0.0) for row in history], dtype=np.float64)
    mech_build_v = np.array([row.get("mechanics_build_v_local_time", 0.0) for row in history], dtype=np.float64)
    mech_elem = np.array([row.get("mechanics_elem_hessian_time", 0.0) for row in history], dtype=np.float64)
    mech_scatter = np.array([row.get("mechanics_scatter_time", 0.0) for row in history], dtype=np.float64)
    mech_coo = np.array([row.get("mechanics_coo_assembly_time", 0.0) for row in history], dtype=np.float64)
    mech_solve = np.array([row.get("mechanics_solve_time", 0.0) for row in history], dtype=np.float64)
    design_grad = np.array([row.get("design_grad_time", 0.0) for row in history], dtype=np.float64)
    design_ls = np.array([row.get("design_ls_time", 0.0) for row in history], dtype=np.float64)
    design_update = np.array([row.get("design_update_time", 0.0) for row in history], dtype=np.float64)
    design_iter = np.array([row.get("design_iter_time", 0.0) for row in history], dtype=np.float64)

    final = result["final_metrics"]
    return {
        "ranks": int(result["nprocs"]),
        "result": str(result["result"]),
        "outer_iterations": int(final["outer_iterations"]),
        "outer_stall_converged": bool(final.get("outer_stall_converged", False)),
        "final_p": float(final["final_p_penal"]),
        "final_compliance": float(final["final_compliance"]),
        "final_volume": float(final["final_volume_fraction"]),
        "wall_time": float(result["time"]),
        "setup_time": float(result["setup_time"]),
        "solve_time": float(result["time"] - result["setup_time"]),
        "sum_mech_assemble": float(np.sum(mech_assemble)),
        "sum_mech_build_v": float(np.sum(mech_build_v)),
        "sum_mech_elem": float(np.sum(mech_elem)),
        "sum_mech_scatter": float(np.sum(mech_scatter)),
        "sum_mech_coo": float(np.sum(mech_coo)),
        "sum_mech_solve": float(np.sum(mech_solve)),
        "sum_design_grad": float(np.sum(design_grad)),
        "sum_design_ls": float(np.sum(design_ls)),
        "sum_design_update": float(np.sum(design_update)),
        "sum_design_iter": float(np.sum(design_iter)),
        "sum_mech_ksp_its": int(sum(int(row.get("mechanics_ksp_its", 0)) for row in history)),
        "sum_design_gd_its": int(sum(int(row.get("design_iters", 0)) for row in history)),
        "sum_design_ls_evals": int(sum(int(row.get("design_ls_evals", 0)) for row in history)),
        "mean_mech_ksp_its": float(np.mean([row.get("mechanics_ksp_its", 0) for row in history])) if history else math.nan,
        "mean_design_gd_its": float(np.mean([row.get("design_iters", 0) for row in history])) if history else math.nan,
        "mean_design_ls_evals": float(np.mean([row.get("design_ls_evals", 0) for row in history])) if history else math.nan,
        "u_dofs": int(result["mesh"]["displacement_free_dofs"]),
        "z_dofs": int(result["mesh"]["design_free_dofs"]),
        "total_dofs": int(result["mesh"]["displacement_free_dofs"] + result["mesh"]["design_free_dofs"]),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_wall_scaling(rows: list[dict], path: Path) -> None:
    ranks = np.array([row["ranks"] for row in rows], dtype=np.float64)
    wall = np.array([row["wall_time"] for row in rows], dtype=np.float64)
    setup = np.array([row["setup_time"] for row in rows], dtype=np.float64)
    solve = np.array([row["solve_time"] for row in rows], dtype=np.float64)
    ideal = wall[0] * ranks[0] / ranks

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(ranks, wall, marker="o", label="wall time")
    ax.loglog(ranks, solve, marker="s", label="solve time")
    ax.loglog(ranks, setup, marker="^", label="setup time")
    ax.loglog(ranks, ideal, linestyle="--", color="black", label="ideal 1/N")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time [s]")
    ax.set_title("Strong scaling: total time")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_scaling(rows: list[dict], path: Path) -> None:
    ranks = np.array([row["ranks"] for row in rows], dtype=np.float64)
    phases = {
        "mech assemble": np.array([row["sum_mech_assemble"] for row in rows], dtype=np.float64),
        "mech solve": np.array([row["sum_mech_solve"] for row in rows], dtype=np.float64),
        "design grad": np.array([row["sum_design_grad"] for row in rows], dtype=np.float64),
        "design LS": np.array([row["sum_design_ls"] for row in rows], dtype=np.float64),
        "design update": np.array([row["sum_design_update"] for row in rows], dtype=np.float64),
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for label, values in phases.items():
        ax.loglog(ranks, values, marker="o", label=label)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("accumulated phase time [s]")
    ax.set_title("Strong scaling: accumulated phase costs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_quality(rows: list[dict], path: Path) -> None:
    ranks = np.array([row["ranks"] for row in rows], dtype=np.float64)
    outers = np.array([row["outer_iterations"] for row in rows], dtype=np.float64)
    pvals = np.array([row["final_p"] for row in rows], dtype=np.float64)
    vols = np.array([row["final_volume"] for row in rows], dtype=np.float64)
    comp = np.array([row["final_compliance"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(ranks, outers, marker="o", label="outer its")
    axes[0].plot(ranks, pvals, marker="s", label="final p")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("MPI ranks")
    axes[0].set_title("Termination point")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(ranks, vols, marker="o", label="final volume")
    axes[1].plot(ranks, comp, marker="s", label="final compliance")
    axes[1].axhline(0.4, linestyle="--", color="black", linewidth=1.0, label="target volume")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("MPI ranks")
    axes[1].set_title("Final state quality")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_efficiency(rows: list[dict], path: Path) -> None:
    ranks = np.array([row["ranks"] for row in rows], dtype=np.float64)
    wall = np.array([row["wall_time"] for row in rows], dtype=np.float64)
    solve = np.array([row["solve_time"] for row in rows], dtype=np.float64)
    speedup_wall = wall[0] / wall
    speedup_solve = solve[0] / solve
    eff_wall = speedup_wall / ranks
    eff_solve = speedup_solve / ranks

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(ranks, eff_wall, marker="o", base=2, label="wall efficiency")
    ax.semilogx(ranks, eff_solve, marker="s", base=2, label="solve efficiency")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("parallel efficiency")
    ax.set_title("Strong scaling efficiency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(rows: list[dict], report_path: Path, asset_dir: Path, *, boost_state: str) -> None:
    baseline = rows[0]
    summary_rows = [
        [
            row["ranks"],
            row["result"],
            row["outer_iterations"],
            _fmt(row["final_p"], 2),
            _fmt(row["final_volume"], 6),
            _fmt(row["final_compliance"], 6),
            _fmt(row["wall_time"], 3),
            _fmt(row["solve_time"], 3),
            row["sum_mech_ksp_its"],
            row["sum_design_gd_its"],
            row["sum_design_ls_evals"],
            _fmt(baseline["wall_time"] / row["wall_time"], 3),
        ]
        for row in rows
    ]
    phase_rows = [
        [
            row["ranks"],
            _fmt(row["sum_mech_assemble"], 3),
            _fmt(row["sum_mech_solve"], 3),
            _fmt(row["sum_design_grad"], 3),
            _fmt(row["sum_design_ls"], 3),
            _fmt(row["sum_design_update"], 3),
        ]
        for row in rows
    ]

    text = "\n".join(
        [
            "# Parallel Scaling Report",
            "",
            "This report covers the current fine-grid strong-scaling sweep for the parallel topology benchmark after fixing the post-write MPI hang.",
            "",
            "## Hang Investigation",
            "",
            "- Root cause: the final snapshot bookkeeping used `snapshot_records` only on rank 0, while non-root ranks kept an empty local list.",
            "- Effect: at the very end of a run, non-root ranks incorrectly entered one extra `_maybe_save_snapshot(...)` call, which performs a distributed `theta` gather; rank 0 skipped that call, so the non-root ranks hung in the final collective.",
            "- Fix: track `last_snapshot_outer` on all ranks, update it inside `_maybe_save_snapshot(...)` after the collective gather, and use that shared value for the final snapshot decision.",
            "- Verification: a traced 32-rank reproduction now exits cleanly with all 32 ranks reaching `atexit`.",
            "",
            "## Benchmark Configuration",
            "",
            "- mesh: `768 x 384`",
            "- support/load pads: `32` cells each to preserve physical patch size relative to the coarse `384 x 192` case",
            "- ranks: `1, 2, 4, 8, 16, 32`",
            "- mechanics: `fgmres + gamg`, `rtol = 1e-4`, `max_it = 100`",
            "- design: `golden_adaptive`, `tolg = 1e-3`, `design_maxit = 20`",
            "- continuation: `p += 0.2` every outer step, max-it gate over last 10 outers",
            "- graceful stall stop: `dtheta < 1e-6`, `dtheta_state < 1e-6`, `p >= 4.0`",
            "- threading: `1` CPU thread per MPI rank for JAX/BLAS/OpenMP",
            f"- CPU frequency boost during sweep: `{boost_state}`",
            "",
            "## Summary Table",
            "",
            _table(
                [
                    "ranks",
                    "result",
                    "outer",
                    "final p",
                    "final V",
                    "final C",
                    "wall [s]",
                    "solve [s]",
                    "KSP its",
                    "GD its",
                    "LS evals",
                    "speedup",
                ],
                summary_rows,
            ),
            "",
            "## Phase Table",
            "",
            _table(
                ["ranks", "mech asm [s]", "mech solve [s]", "design grad [s]", "design LS [s]", "design update [s]"],
                phase_rows,
            ),
            "",
            "## Plots",
            "",
            f"![Wall scaling]({_asset_rel(asset_dir / 'wall_scaling.png')})",
            "",
            f"![Phase scaling]({_asset_rel(asset_dir / 'phase_scaling.png')})",
            "",
            f"![Efficiency]({_asset_rel(asset_dir / 'efficiency.png')})",
            "",
            f"![Quality vs ranks]({_asset_rel(asset_dir / 'quality_vs_ranks.png')})",
            "",
            "## Raw Data",
            "",
            f"- CSV summary: `{_asset_rel(asset_dir / 'scaling_summary.csv')}`",
            *[
                f"- Rank {row['ranks']} JSON: `{_asset_rel(asset_dir / f'run_r{row['ranks']:02d}.json')}`"
                for row in rows
            ],
            "",
        ]
    )
    report_path.write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    asset_dir = args.asset_dir.resolve()
    report_path = (args.report_path.resolve() if args.report_path is not None else asset_dir / "report.md")
    asset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ranks in RANKS:
        json_out = asset_dir / f"run_r{ranks:02d}.json"
        result = _run_case(ranks, json_out)
        rows.append(_summarize(result))

    _write_csv(rows, asset_dir / "scaling_summary.csv")
    _plot_wall_scaling(rows, asset_dir / "wall_scaling.png")
    _plot_phase_scaling(rows, asset_dir / "phase_scaling.png")
    _plot_quality(rows, asset_dir / "quality_vs_ranks.png")
    _plot_efficiency(rows, asset_dir / "efficiency.png")
    _write_report(rows, report_path, asset_dir, boost_state=_read_boost_state())
    print(f"Wrote {_asset_rel(report_path)}")


if __name__ == "__main__":
    main()
