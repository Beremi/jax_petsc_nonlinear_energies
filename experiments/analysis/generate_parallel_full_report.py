#!/usr/bin/env python3
"""Generate a parallel topology benchmark report from saved run artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_GIF_FRAMES = 120
DEFAULT_ASSET_DIR = (
    REPO_ROOT
    / "artifacts"
    / "raw_results"
    / "topology_reports"
    / "parallel_final"
)


def _asset_rel(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _report_context(result: dict) -> dict:
    params = dict(result.get("parameters", {}))
    mesh = dict(result.get("mesh", {}))
    solver = dict(result.get("solver_options", {}))
    ctx = {}
    ctx.update(mesh)
    ctx.update(params)
    ctx.update(solver)
    return ctx


def _cfg(result: dict, key: str, default=None):
    solver = result.get("solver_options", {})
    params = result.get("parameters", {})
    if key in solver:
        return solver[key]
    if key in params:
        return params[key]
    return default


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(cell) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell_density(theta_grid: np.ndarray) -> np.ndarray:
    return 0.25 * (
        theta_grid[:-1, :-1]
        + theta_grid[1:, :-1]
        + theta_grid[:-1, 1:]
        + theta_grid[1:, 1:]
    )


def _write_history_csv(history: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "outer_iter",
        "p_penal",
        "p_step",
        "lambda_correction",
        "lambda_penalty",
        "lambda_reference",
        "lambda_effective",
        "compliance",
        "volume_fraction_before",
        "volume_residual_before",
        "volume_fraction",
        "volume_residual",
        "theta_state_change",
        "design_change",
        "compliance_change",
        "mechanics_ksp_its",
        "design_iters",
        "design_ls_evals",
        "design_grad_norm_last",
        "mechanics_message",
        "design_message",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _make_state_figure(theta_grid: np.ndarray, u_grid: np.ndarray, params: dict, state_png: Path) -> None:
    cell_density = _cell_density(theta_grid)
    umag = np.linalg.norm(u_grid, axis=2)
    extent = (0.0, float(params["length"]), 0.0, float(params["height"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    im0 = axes[0].imshow(
        cell_density.T,
        extent=extent,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\theta_c$")
    axes[0].set_title("Final density field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(
        umag.T,
        extent=extent,
        cmap="magma",
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\|u\|_2$")
    axes[1].set_title("Final displacement magnitude")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(state_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _oriented_frame_gray(frame_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(frame_path).convert("L"), dtype=np.uint8)
    # Older snapshots were written with swapped axes; transpose once to recover
    # the physical x-horizontal, y-vertical orientation.
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _rewrite_snapshot_frames(snapshot_dir: Path) -> None:
    for frame_path in sorted(snapshot_dir.glob("*.png")):
        arr = _oriented_frame_gray(frame_path)
        Image.fromarray(arr, mode="L").save(frame_path)


def _make_convergence_figure(result: dict, conv_png: Path) -> None:
    history = [row for row in result["history"] if "compliance" in row]
    it = np.array([row["outer_iter"] for row in history], dtype=np.int32)
    compliance = np.array([row["compliance"] for row in history], dtype=np.float64)
    volume = np.array([row["volume_fraction"] for row in history], dtype=np.float64)
    vol_res = np.abs(np.array([row["volume_residual"] for row in history], dtype=np.float64))
    theta_state_change = np.array([row["theta_state_change"] for row in history], dtype=np.float64)
    design_change = np.array([row["design_change"] for row in history], dtype=np.float64)
    compliance_change = np.array([row["compliance_change"] for row in history], dtype=np.float64)
    mech_ksp = np.array([row["mechanics_ksp_its"] for row in history], dtype=np.int32)
    design_iters = np.array([row["design_iters"] for row in history], dtype=np.int32)
    p_penal = np.array([row["p_penal"] for row in history], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(it, compliance, marker="o", color="#005f73")
    axes[0, 0].set_title("Compliance")
    axes[0, 0].set_xlabel("Outer iteration")
    axes[0, 0].set_ylabel(r"$f^T u$")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(it, volume, marker="o", color="#9b2226", label="achieved volume fraction")
    axes[0, 1].axhline(
        result["parameters"]["volume_fraction_target"],
        color="#6d6875",
        linestyle="--",
        linewidth=1.5,
        label="target",
    )
    axes[0, 1].set_title("Volume control")
    axes[0, 1].set_xlabel("Outer iteration")
    axes[0, 1].set_ylabel(r"$V(\theta)/|\Omega|$")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    valid_comp = np.isfinite(compliance_change)
    axes[1, 0].semilogy(it, np.maximum(vol_res, 1e-12), marker="o", label=r"$|V-V^*|$")
    axes[1, 0].semilogy(it, np.maximum(theta_state_change, 1e-12), marker="s", label=r"$\Delta \theta_{\mathrm{state}}$")
    axes[1, 0].semilogy(it, np.maximum(design_change, 1e-12), marker="d", label=r"$\Delta \theta$")
    axes[1, 0].semilogy(
        it[valid_comp],
        np.maximum(compliance_change[valid_comp], 1e-12),
        marker="^",
        label=r"$\Delta C$",
    )
    axes[1, 0].set_title("Outer convergence indicators")
    axes[1, 0].set_xlabel("Outer iteration")
    axes[1, 0].set_ylabel("indicator")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(it, mech_ksp, marker="o", color="#264653", label="mechanics KSP its")
    axes[1, 1].plot(it, design_iters, marker="s", color="#e76f51", label="design GD iters")
    ax2 = axes[1, 1].twinx()
    ax2.step(it, p_penal, where="mid", color="#8d99ae", linestyle="--", label="SIMP p")
    axes[1, 1].set_title("Inner work and staircase continuation")
    axes[1, 1].set_xlabel("Outer iteration")
    axes[1, 1].set_ylabel("iterations")
    ax2.set_ylabel("penalization p")
    axes[1, 1].grid(alpha=0.25)
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(conv_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_density_step_figure(result: dict, density_step_png: Path) -> None:
    history = [row for row in result["history"] if "compliance" in row]
    it = np.array([row["outer_iter"] for row in history], dtype=np.int32)
    design_change = np.array([row["design_change"] for row in history], dtype=np.float64)
    theta_state_change = np.array([row["theta_state_change"] for row in history], dtype=np.float64)
    p_penal = np.array([row["p_penal"] for row in history], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.semilogy(it, np.maximum(design_change, 1e-16), marker="o", ms=3.5, color="#0a9396", label=r"$\Delta \theta$")
    ax.semilogy(
        it,
        np.maximum(theta_state_change, 1e-16),
        marker="s",
        ms=3.0,
        color="#bb3e03",
        alpha=0.85,
        label=r"$\Delta \theta_{\mathrm{state}}$",
    )
    ax.set_title("Outer density step size")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("step size")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.step(it, p_penal, where="mid", color="#6c757d", linestyle="--", label="SIMP p")
    ax2.set_ylabel("penalization p")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(density_step_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_density_gif(
    result: dict,
    snapshot_outer: np.ndarray,
    snapshot_p: np.ndarray,
    snapshot_volume: np.ndarray,
    snapshot_dir: Path,
    density_gif: Path,
) -> None:
    history_lookup = {int(row["outer_iter"]): row for row in result["history"]}
    frame_files = sorted(snapshot_dir.glob("theta_*.png"))
    if not frame_files:
        return
    if MAX_GIF_FRAMES > 0 and len(frame_files) > MAX_GIF_FRAMES:
        idx = np.linspace(0, len(frame_files) - 1, MAX_GIF_FRAMES, dtype=np.int32)
        frame_files = [frame_files[i] for i in idx.tolist()]

    manifest = {int(outer): (float(p), float(v)) for outer, p, v in zip(snapshot_outer, snapshot_p, snapshot_volume)}
    cmap = plt.get_cmap("cividis")
    frames: list[Image.Image] = []
    for frame_path in frame_files:
        outer_iter = int(frame_path.stem.split("_outer_")[-1])
        gray = _oriented_frame_gray(frame_path)
        rgb = np.asarray(cmap(gray.astype(np.float32) / 255.0)[..., :3] * 255.0, dtype=np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
        canvas = Image.new("RGB", (img.width, img.height + 42), (255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        p_value, v_value = manifest.get(outer_iter, (float("nan"), float("nan")))
        hist = history_lookup.get(outer_iter, {})
        compliance = hist.get("compliance", float("nan"))
        label = (
            f"outer {outer_iter}   "
            f"p={p_value:.2f}   "
            f"V={v_value:.4f}   "
            f"C={float(compliance):.4f}"
        )
        draw.text((12, img.height + 12), label, fill=0)
        frames.append(canvas)

    frames[0].save(
        density_gif,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
        optimize=False,
    )


def _solver_command(result: dict, snapshot_dir: Path, run_json: Path, state_npz: Path) -> str:
    ctx = _report_context(result)
    design_gd_line_search = ctx.get("design_gd_line_search", "golden_adaptive")
    design_gd_adaptive_window_scale = ctx.get("design_gd_adaptive_window_scale", None)
    return "\n".join(
        [
            f"mpiexec -n {result['nprocs']} ./.venv/bin/python src/problems/topology/jax/solve_topopt_parallel.py \\",
            f"    --nx {ctx['nx']} --ny {ctx['ny']} --length {ctx['length']} --height {ctx['height']} \\",
            f"    --traction {ctx['traction']} --load_fraction {ctx['load_fraction']} \\",
            f"    --fixed_pad_cells {ctx['fixed_pad_cells']} --load_pad_cells {ctx['load_pad_cells']} \\",
            f"    --volume_fraction_target {ctx['volume_fraction_target']} --theta_min {ctx['theta_min']} \\",
            f"    --solid_latent {ctx['solid_latent']} --young {ctx['young']} --poisson {ctx['poisson']} \\",
            f"    --alpha_reg {ctx['alpha_reg']} --ell_pf {ctx['ell_pf']} --mu_move {ctx['mu_move']} \\",
            f"    --beta_lambda {ctx['beta_lambda']} --volume_penalty {ctx['volume_penalty']} \\",
            f"    --p_start {ctx['p_start']} --p_max {ctx['p_max']} --p_increment {ctx['p_increment']} \\",
            f"    --continuation_interval {ctx['continuation_interval']} --outer_maxit {ctx['outer_maxit']} \\",
            f"    --outer_tol {ctx['outer_tol']} --volume_tol {ctx['volume_tol']} \\",
            f"    --design_maxit {ctx['design_maxit']} --tolf {ctx['tolf']} --tolg {ctx['tolg']} \\",
            f"    --linesearch_tol {ctx['linesearch_tol']} --mechanics_ksp_rtol {ctx['mechanics_ksp_rtol']} \\",
            (
                "    --linesearch_relative_to_bound \\"
                if ctx.get("linesearch_relative_to_bound", False)
                else ""
            ),
            f"    --design_gd_line_search {design_gd_line_search} \\",
            (
                f"    --design_gd_adaptive_window_scale {design_gd_adaptive_window_scale} \\"
                if design_gd_adaptive_window_scale is not None
                else ""
            ),
            f"    --mechanics_ksp_max_it {ctx['mechanics_ksp_max_it']} --quiet --print_outer_iterations \\",
            f"    --save_outer_state_history --outer_snapshot_stride 2 \\",
            f"    --outer_snapshot_dir {_asset_rel(snapshot_dir)} \\",
            f"    --json_out {_asset_rel(run_json)} --state_out {_asset_rel(state_npz)}",
        ]
    ).replace("\\\n\n", "\\\n")


def _make_report(
    result: dict,
    theta_grid: np.ndarray,
    u_grid: np.ndarray,
    snapshot_outer: np.ndarray,
    snapshot_p: np.ndarray,
    snapshot_volume: np.ndarray,
    *,
    asset_dir: Path,
    run_json: Path,
    state_npz: Path,
    csv_path: Path,
    state_png: Path,
    conv_png: Path,
    density_step_png: Path,
    density_gif: Path,
    ls_outer_png: Path,
    ls_stage_png: Path,
    ls_gamma_png: Path,
    snapshot_dir: Path,
) -> str:
    history = [row for row in result["history"] if "compliance" in row]
    params = result["parameters"]
    solver_options = result["solver_options"]
    final = result["final_metrics"]
    cell_density = _cell_density(theta_grid)
    gray_ratio = float(np.mean((cell_density > 0.1) & (cell_density < 0.9)))
    total_mech_ksp = int(sum(row["mechanics_ksp_its"] for row in history))
    total_design_iters = int(sum(row["design_iters"] for row in history))
    total_design_ls = int(sum(row["design_ls_evals"] for row in history))
    total_mech_solve = float(sum(row.get("mechanics_solve_time", 0.0) for row in history))
    total_mech_scatter = float(sum(row.get("mechanics_scatter_time", 0.0) for row in history))
    total_design_grad = float(sum(row.get("design_grad_time", 0.0) for row in history))
    total_design_ls_time = float(sum(row.get("design_ls_time", 0.0) for row in history))
    last = history[-1]
    stall_row = None
    zero_streak = 0
    for row in history:
        if int(row.get("design_iters", -1)) == 0:
            zero_streak += 1
        else:
            zero_streak = 0
        if zero_streak >= 3:
            stall_row = row
            break

    config_rows = [
        ["MPI ranks", result["nprocs"]],
        ["Mesh", f"{result['mesh']['nx']} x {result['mesh']['ny']}"],
        ["Elements", result["mesh"]["elements"]],
        ["Free displacement DOFs", result["mesh"]["displacement_free_dofs"]],
        ["Free design DOFs", result["mesh"]["design_free_dofs"]],
        ["Target volume fraction", _fmt(params["volume_fraction_target"], 4)],
        ["SIMP schedule", f"p = p + {params['p_increment']} every {params['continuation_interval']} outer iterations"],
        ["Final p target", _fmt(params["p_max"], 2)],
        ["Mechanics solver", f"{solver_options['mechanics_ksp_type']} + {solver_options['mechanics_pc_type']}"],
        ["Near-nullspace", solver_options["mechanics_use_near_nullspace"]],
        ["Design LS policy", solver_options.get("design_gd_line_search", "golden_adaptive")],
        ["Design LS scale", _fmt(solver_options.get("design_gd_adaptive_window_scale", float("nan")), 3)],
        ["Design LS tol mode", "relative to bound" if solver_options.get("linesearch_relative_to_bound", False) else "absolute"],
    ]
    if params.get("stall_theta_tol", 0.0) > 0.0:
        config_rows.append(["Graceful stall tol", _fmt(params["stall_theta_tol"], 2)])
        config_rows.append(["Graceful stall p min", _fmt(params["stall_p_min"], 2)])

    summary_rows = [
        ["Result", result["result"]],
        ["Outer iterations", final["outer_iterations"]],
        ["Final p", _fmt(final["final_p_penal"], 4)],
        ["Wall time [s]", _fmt(result["time"], 3)],
        ["Setup time [s]", _fmt(result["setup_time"], 3)],
        ["Solve time [s]", _fmt(result["time"] - result["setup_time"], 3)],
        ["Final compliance", _fmt(final["final_compliance"], 6)],
        ["Final volume fraction", _fmt(final["final_volume_fraction"], 6)],
        ["Final volume error", _fmt(last["volume_residual"], 6)],
        ["Final design change", _fmt(last["design_change"], 6)],
        ["Final compliance change", _fmt(last["compliance_change"], 6)],
        ["Gray fraction (cell 0.1-0.9)", _fmt(gray_ratio, 4)],
    ]

    timing_rows = [
        ["Total mechanics KSP time [s]", _fmt(total_mech_solve, 3)],
        ["Total mechanics scatter time [s]", _fmt(total_mech_scatter, 3)],
        ["Total design grad time [s]", _fmt(total_design_grad, 3)],
        ["Total design line-search time [s]", _fmt(total_design_ls_time, 3)],
        ["Total mechanics KSP iterations", total_mech_ksp],
        ["Total design GD iterations", total_design_iters],
        ["Total design line-search evals", total_design_ls],
    ]

    history_rows = []
    for row in history:
        history_rows.append(
            [
                row["outer_iter"],
                _fmt(row["p_penal"], 2),
                row["mechanics_ksp_its"],
                row["design_iters"],
                row["design_ls_evals"],
                _fmt(row["compliance"], 6),
                _fmt(row["volume_fraction"], 6),
                _fmt(row["volume_residual"], 6),
                _fmt(row["design_change"], 6),
                _fmt(row["compliance_change"], 6),
            ]
        )

    history_table = _markdown_table(
        ["k", "p", "mech KSP", "GD", "LS evals", "compliance", "volume", "vol error", "dtheta", "dC"],
        history_rows,
    )
    design_maxit = _cfg(result, "design_maxit", "?")
    linesearch_tol = _cfg(result, "linesearch_tol", "?")
    has_density_gif = density_gif.exists()
    has_ls_diagnostics = ls_outer_png.exists() and ls_stage_png.exists()
    has_ls_gamma = ls_gamma_png.exists()
    mechanics_line = (
        f"- mechanics solved by PETSc `{solver_options['mechanics_ksp_type']} + {solver_options['mechanics_pc_type']}`"
    )
    design_line = (
        f"- design updated by distributed gradient descent with adaptive golden-section line search "
        f"(`design_maxit = {design_maxit}`, `linesearch_tol = {linesearch_tol}`)"
    )
    if int(design_maxit) == 1:
        design_line += "\n- one accepted GD step per outer iteration"
    sections = [
        "# Parallel JAX Topology Benchmark",
        "",
        f"This report documents the current `{result['nprocs']}`-rank parallel topology run on the distributed mesh `{result['mesh']['nx']} x {result['mesh']['ny']}`.",
        "",
        "The implementation follows the current stable parallel path in the repository:",
        "",
        mechanics_line,
        "- rigid-body near-nullspace supplied to GAMG",
        design_line,
        f"- fixed staircase SIMP continuation up to `p = {params['p_max']}`",
        "",
        "## Configuration",
        "",
        _markdown_table(["Knob", "Value"], config_rows),
        "",
        "## Final State",
        "",
        f"![Final state]({_asset_rel(state_png)})",
        "",
        "## Convergence History",
        "",
        f"![Convergence history]({_asset_rel(conv_png)})",
        "",
        "## Density Step Size",
        "",
        f"![Outer density step size]({_asset_rel(density_step_png)})",
        "",
        "## Run Summary",
        "",
        _markdown_table(["Metric", "Value"], summary_rows),
        "",
        "## Status Notes",
        "",
    ]
    if stall_row is not None:
        sections.extend(
            [
                (
                    f"- The design iteration first stalled at outer iteration `{stall_row['outer_iter']}`: "
                    f"`design_iters = 0`, `design_message = {stall_row.get('design_message', 'n/a')}`."
                ),
                (
                    f"- After that point the outer loop kept advancing on an unchanged design, so the tail of the "
                    f"history is a frozen-state plateau rather than further optimisation progress."
                ),
                "",
            ]
        )
    else:
        sections.extend(
            [
                "- No zero-step design stall was detected in the saved outer history.",
                "",
            ]
        )
    if final.get("outer_stall_converged", False):
        sections.extend(
            [
                (
                    f"- The run stopped gracefully once both `dtheta` and `dtheta_state` fell below "
                    f"`{params.get('stall_theta_tol', 0.0):.1e}` at `p >= {params.get('stall_p_min', float('nan')):.2f}`."
                ),
                "",
            ]
        )
    if has_density_gif:
        sections[sections.index("## Run Summary"):sections.index("## Run Summary")] = [
            "## Density Evolution",
            "",
            f"![Density evolution]({_asset_rel(density_gif)})",
            "",
        ]
    sections.extend(
        [
        "## Parallel Work Summary",
        "",
        _markdown_table(["Metric", "Value"], timing_rows),
        "",
        ]
    )
    if has_ls_diagnostics:
        sections.extend(
            [
                "## LS Diagnostics",
                "",
                f"![LS diagnostics by outer iteration]({_asset_rel(ls_outer_png)})",
                "",
                f"![LS diagnostics by p stage]({_asset_rel(ls_stage_png)})",
                "",
            ]
        )
        if has_ls_gamma:
            sections.extend(
                [
                    f"![Gamma distance along -grad]({_asset_rel(ls_gamma_png)})",
                    "",
                ]
            )
    sections.extend(
        [
        "## Outer Iteration Table",
        "",
        history_table,
        "",
        "## Artifacts",
        "",
        f"- JSON result: `{_asset_rel(run_json)}`",
        f"- Final state: `{_asset_rel(state_npz)}`",
        f"- Outer-history CSV: `{_asset_rel(csv_path)}`",
        f"- Final-state figure: `{_asset_rel(state_png)}`",
        f"- Convergence figure: `{_asset_rel(conv_png)}`",
        f"- Density-step figure: `{_asset_rel(density_step_png)}`",
        "",
        "## Reproduction",
        "",
        "```bash",
        _solver_command(result, snapshot_dir, run_json, state_npz),
        "```",
        ]
    )
    if has_density_gif:
        sections.insert(-5, f"- Density-evolution GIF: `{_asset_rel(density_gif)}`")
    else:
        sections.insert(-5, "- Density-evolution GIF: not available for this run (no snapshot frames were saved)")
    if has_ls_diagnostics:
        sections.insert(-5, f"- LS diagnostics by outer iteration: `{_asset_rel(ls_outer_png)}`")
        sections.insert(-5, f"- LS diagnostics by p stage: `{_asset_rel(ls_stage_png)}`")
    if has_ls_gamma:
        sections.insert(-5, f"- LS gamma-distance plot: `{_asset_rel(ls_gamma_png)}`")
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset_dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    asset_dir = args.asset_dir.resolve()
    run_json = asset_dir / "parallel_full_run.json"
    state_npz = asset_dir / "parallel_full_state.npz"
    csv_path = asset_dir / "parallel_full_outer_history.csv"
    state_png = asset_dir / "final_state.png"
    conv_png = asset_dir / "convergence_history.png"
    density_step_png = asset_dir / "density_step_history.png"
    density_gif = asset_dir / "density_evolution.gif"
    ls_outer_png = asset_dir / "ls_diagnostics_outer.png"
    ls_stage_png = asset_dir / "ls_diagnostics_by_p.png"
    ls_gamma_png = asset_dir / "ls_gamma_distance.png"
    snapshot_dir = asset_dir / "frames"
    report_path = (args.report_path.resolve() if args.report_path is not None else asset_dir / "report.md")

    asset_dir.mkdir(parents=True, exist_ok=True)
    if not run_json.exists():
        raise FileNotFoundError(run_json)
    if not state_npz.exists():
        raise FileNotFoundError(state_npz)

    result = json.loads(run_json.read_text())
    state = np.load(state_npz)
    theta_grid = np.asarray(state["theta_grid"], dtype=np.float64)
    u_grid = np.asarray(state["u_grid"], dtype=np.float64)
    snapshot_outer = np.asarray(state["snapshot_outer"], dtype=np.int32)
    snapshot_p = np.asarray(state["snapshot_p"], dtype=np.float64)
    snapshot_volume = np.asarray(state["snapshot_volume"], dtype=np.float64)

    _rewrite_snapshot_frames(snapshot_dir)
    _write_history_csv(result["history"], csv_path)
    _make_state_figure(theta_grid, u_grid, result["parameters"], state_png)
    _make_convergence_figure(result, conv_png)
    _make_density_step_figure(result, density_step_png)
    _make_density_gif(result, snapshot_outer, snapshot_p, snapshot_volume, snapshot_dir, density_gif)
    report_path.write_text(
        _make_report(
            result,
            theta_grid,
            u_grid,
            snapshot_outer,
            snapshot_p,
            snapshot_volume,
            asset_dir=asset_dir,
            run_json=run_json,
            state_npz=state_npz,
            csv_path=csv_path,
            state_png=state_png,
            conv_png=conv_png,
            density_step_png=density_step_png,
            density_gif=density_gif,
            ls_outer_png=ls_outer_png,
            ls_stage_png=ls_stage_png,
            ls_gamma_png=ls_gamma_png,
            snapshot_dir=snapshot_dir,
        )
    )
    print(f"Wrote {_asset_rel(report_path)}")


if __name__ == "__main__":
    main()
