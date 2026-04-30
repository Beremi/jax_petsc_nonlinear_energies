#!/usr/bin/env python3
"""Generate the compact fine-grid JAX topology benchmark report."""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import date
from pathlib import Path
from textwrap import dedent

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from src.problems.topology.jax.mesh import CantileverTopologyMesh
from src.problems.topology.jax.solve_topopt_jax import run_topology_optimisation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = REPO_ROOT / "artifacts" / "raw_results" / "topology_reports" / "serial_reference"
DEFAULT_REPORT_PATH = DEFAULT_ASSET_DIR / "report.md"
ASSET_DIR = DEFAULT_ASSET_DIR
REPORT_PATH = DEFAULT_REPORT_PATH
JSON_PATH = ASSET_DIR / "report_run.json"
NPZ_PATH = ASSET_DIR / "report_state.npz"
CSV_PATH = ASSET_DIR / "report_outer_history.csv"
MESH_PNG = ASSET_DIR / "mesh_preview.png"
STATE_PNG = ASSET_DIR / "final_state.png"
CONV_PNG = ASSET_DIR / "convergence_history.png"
DENSITY_GIF = ASSET_DIR / "density_evolution.gif"
REPORT_TITLE = "JAX Topology Optimisation Benchmark"
REPORT_GENERATOR_CMD = "./.venv/bin/python experiments/analysis/generate_report_assets.py"
GIF_MAX_FRAMES = 80

CASE_PARAMS = {
    "nx": 192,
    "ny": 96,
    "length": 2.0,
    "height": 1.0,
    "traction": 1.0,
    "load_fraction": 0.2,
    "fixed_pad_cells": 16,
    "load_pad_cells": 16,
    "volume_fraction_target": 0.4,
    "theta_min": 1e-3,
    "solid_latent": 10.0,
    "young": 1.0,
    "poisson": 0.3,
    "alpha_reg": 5e-3,
    "ell_pf": 0.08,
    "mu_move": 0.01,
    "lambda_init": 0.0,
    "beta_lambda": 12.0,
    "volume_penalty": 10.0,
    "p_start": 1.0,
    "p_max": 4.0,
    "p_increment": 0.5,
    "continuation_interval": 20,
    "outer_maxit": 180,
    "outer_tol": 2e-2,
    "volume_tol": 1e-3,
    "mechanics_maxit": 200,
    "design_maxit": 400,
    "tolf": 1e-6,
    "tolg": 1e-3,
    "linesearch_tol": 1e-2,
    "ksp_rtol": 1e-2,
    "ksp_max_it": 80,
    "mechanics_solver_type": "amg",
    "design_nonlinear_method": "newton_trust",
    "verbose": False,
}


ENERGY_SNIPPET = """```python
def mechanics_energy(
    u_free, u_0, freedofs, elems, elem_B, elem_area, material_scale, constitutive, force
):
    u_full = expand_free_dofs(u_free, u_0, freedofs)
    u_elem = u_full[elems]
    strain = jnp.einsum("eij,ej->ei", elem_B, u_elem)
    elastic_density = 0.5 * jnp.einsum("ei,ij,ej->e", strain, constitutive, strain)
    return jnp.sum(elem_area * material_scale * elastic_density) - jnp.dot(force, u_full)


def design_energy(
    z_free, z_0, freedofs, elems, elem_grad_phi, elem_area, e_frozen,
    z_old_full, lambda_volume, alpha_reg, ell_pf, mu_move, theta_min, p_penal
):
    z_full = expand_free_dofs(z_free, z_0, freedofs)
    theta_full = theta_from_latent(z_full, theta_min)
    theta_elem = theta_full[elems]
    theta_centroid = jnp.mean(theta_elem, axis=1)
    grad_theta = jnp.einsum("eia,ei->ea", elem_grad_phi, theta_elem)
    z_delta_centroid = jnp.mean(z_full[elems] - z_old_full[elems], axis=1)

    double_well = theta_centroid**2 * (1.0 - theta_centroid) ** 2
    reg_density = 0.5 * ell_pf * jnp.sum(grad_theta * grad_theta, axis=1) + double_well / ell_pf
    proximal_density = 0.5 * mu_move * z_delta_centroid**2
    design_density = e_frozen * theta_centroid ** (-p_penal) + lambda_volume * theta_centroid
    return jnp.sum(elem_area * (design_density + alpha_reg * reg_density + proximal_density))
```"""


AUTODIFF_SNIPPET = """```python
mechanics_drv = EnergyDerivator(
    mechanics_energy,
    mechanics_params,
    mesh.adjacency_u,
    jnp.asarray(u_free, dtype=jnp.float64),
)
mechanics_F, mechanics_dF, mechanics_ddF = mechanics_drv.get_derivatives()
mechanics_hess_solver = HessSolverGenerator(
    ddf=mechanics_ddF,
    solver_type="amg",
    elastic_kernel=mesh.elastic_kernel,
    tol=ksp_rtol,
    maxiter=ksp_max_it,
)

design_drv = EnergyDerivator(
    design_energy,
    design_params,
    mesh.adjacency_z,
    jnp.asarray(z_free, dtype=jnp.float64),
)
design_F, design_dF, design_ddF = design_drv.get_derivatives()
design_hess_solver = HessSolverGenerator(
    ddf=design_ddF,
    solver_type="direct",
    tol=ksp_rtol,
    maxiter=ksp_max_it,
)
```"""


STAIRCASE_SNIPPET = """```python
def staircase_p_step(p_penal, *, p_max, p_increment, continuation_interval, outer_it):
    if p_penal >= p_max or outer_it % continuation_interval != 0:
        return 0.0
    return min(p_increment, p_max - p_penal)
```"""


OUTER_PSEUDOCODE = """```text
build mesh, free-DOF masks, element operators
initialize z_0 from the target volume fraction and set u_0 = 0
define Π_h(u_free; z) and G_h(z_free; u, z_old, lambda, p)
use JAX to generate:
    grad_u Π_h, Hess_u Π_h on the displacement graph
    grad_z G_h, Hess_z G_h on the design graph
build sparse linear solvers for both Hessians

for outer iteration k = 1, 2, ...:
    theta_k <- theta_from_latent(z_k)
    material_scale <- theta(theta_k)^p_k

    solve mechanics subproblem in u_free:
        Newton steps use grad_u Π_h and Hess_u Π_h
        each Newton step solves a sparse linear system in the displacement unknowns

    freeze element strain-energy density e_k from u_{k+1}
    build lambda_effective from the sensitivity quantile, lambda_k, and volume penalty

    solve design subproblem in z_free:
        Newton steps use grad_z G_h and Hess_z G_h
        each Newton step solves a sparse linear system in the design unknowns

    update lambda_k from the achieved volume error
    record compliance, volume, design change, and Newton counts
    if p_k is already at p_max and all outer tolerances are satisfied:
        stop
    otherwise update p_k with the fixed staircase rule
```"""


NEWTON_PSEUDOCODE = """```text
given x_n:
    evaluate F(x_n)
    evaluate g_n = grad F(x_n)        <- JAX autodiff
    evaluate H_n = Hess F(x_n)        <- JAX autodiff on fixed sparse graph
    solve H_n * delta = -g_n          <- sparse linear solver
    line-search / trust-region accept or reject the step
    x_{n+1} = x_n + alpha * delta
repeat until function and gradient tolerances are met
```"""


def _fmt(value: float, digits: int = 6) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf"
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _asset_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _strip_report_indent(report: str) -> str:
    lines = []
    in_fence = False
    for line in report.splitlines():
        if not in_fence and line.startswith("        "):
            line = line[8:]
        lines.append(line)
        if line.startswith("```"):
            in_fence = not in_fence
    return "\n".join(lines)


def _element_density(theta_nodal: np.ndarray, elems: np.ndarray) -> np.ndarray:
    return np.asarray(theta_nodal[elems].mean(axis=1), dtype=np.float64)


def _make_mesh_figure(mesh: CantileverTopologyMesh) -> None:
    tri = mtri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], mesh.scalar_elems)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.triplot(tri, color="#bcbcbc", linewidth=0.5, zorder=1)

    left_nodes = np.isclose(mesh.coords[:, 0], 0.0)
    fixed_nodes = mesh.fixed_design_mask
    ax.scatter(
        mesh.coords[left_nodes, 0],
        mesh.coords[left_nodes, 1],
        s=18,
        c="#0b6e4f",
        label="clamped displacement edge",
        zorder=3,
    )
    ax.scatter(
        mesh.coords[fixed_nodes, 0],
        mesh.coords[fixed_nodes, 1],
        s=8,
        c="#d1495b",
        label="fixed-solid design nodes",
        zorder=2,
        alpha=0.9,
    )

    load_min, load_max = mesh._load_patch_bounds()
    ax.plot(
        [mesh.length, mesh.length],
        [load_min, load_max],
        color="#0077b6",
        linewidth=4,
        solid_capstyle="round",
        label="downward traction patch",
        zorder=4,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * mesh.length, 1.05 * mesh.length)
    ax.set_ylim(-0.05 * mesh.height, 1.05 * mesh.height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Structured cantilever mesh: {mesh.n_nodes} nodes, {mesh.scalar_elems.shape[0]} triangles"
    )
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(MESH_PNG, dpi=200)
    plt.close(fig)


def _make_state_figure(mesh: CantileverTopologyMesh, state: dict) -> None:
    tri = mtri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], mesh.scalar_elems)
    theta = np.asarray(state["theta"], dtype=np.float64)
    theta_elem = _element_density(theta, mesh.scalar_elems)
    disp = np.asarray(state["u"], dtype=np.float64).reshape(-1, 2)
    umag = np.linalg.norm(disp, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    pc0 = axes[0].tripcolor(
        tri,
        facecolors=theta_elem,
        shading="flat",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(pc0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\theta_e$")
    axes[0].set_title("Final density field")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    pc1 = axes[1].tripcolor(tri, umag, shading="gouraud", cmap="magma")
    fig.colorbar(pc1, ax=axes[1], fraction=0.046, pad=0.04, label=r"$\|u\|_2$")
    axes[1].set_title("Final displacement magnitude")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(STATE_PNG, dpi=200)
    plt.close(fig)


def _make_density_gif(mesh: CantileverTopologyMesh, state: dict, result: dict) -> None:
    tri = mtri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], mesh.scalar_elems)
    frame_lookup = {int(row["outer_iter"]): row for row in result["history"]}
    frames: list[Image.Image] = []
    snapshots = state.get("theta_history", [])
    if GIF_MAX_FRAMES > 0 and len(snapshots) > GIF_MAX_FRAMES:
        idx = np.linspace(0, len(snapshots) - 1, GIF_MAX_FRAMES, dtype=np.int32)
        snapshots = [snapshots[i] for i in idx.tolist()]

    for snapshot in snapshots:
        theta_elem = _element_density(np.asarray(snapshot["theta"], dtype=np.float64), mesh.scalar_elems)
        outer_iter = int(snapshot["outer_iter"])
        summary = frame_lookup.get(outer_iter, {})

        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        pc = ax.tripcolor(
            tri,
            facecolors=theta_elem,
            shading="flat",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.04, label=r"$\theta_e$")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, mesh.length)
        ax.set_ylim(0.0, mesh.height)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Density evolution: outer iteration {outer_iter}")
        if outer_iter == 0:
            subtitle = f"initial design, V = {snapshot['volume_fraction']:.4f}"
        else:
            subtitle = (
                f"p = {snapshot['p_penal']:.2f}, "
                f"V = {summary.get('volume_fraction', snapshot['volume_fraction']):.4f}, "
                f"C = {summary.get('compliance', float('nan')):.4f}"
            )
        ax.text(
            0.02,
            0.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        with Image.open(buf) as img:
            frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

    if not frames:
        return

    frames[0].save(
        DENSITY_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        optimize=False,
    )


def _make_convergence_figure(result: dict) -> None:
    history = [row for row in result["history"] if "compliance" in row]
    it = np.array([row["outer_iter"] for row in history], dtype=np.int32)
    compliance = np.array([row["compliance"] for row in history], dtype=np.float64)
    volume = np.array([row["volume_fraction"] for row in history], dtype=np.float64)
    vol_res = np.abs(np.array([row["volume_residual"] for row in history], dtype=np.float64))
    theta_state_change = np.array([row["theta_state_change"] for row in history], dtype=np.float64)
    compliance_change = np.array([row["compliance_change"] for row in history], dtype=np.float64)
    mech_iters = np.array([row["mechanics_iters"] for row in history], dtype=np.int32)
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

    axes[1, 1].plot(it, mech_iters, marker="o", color="#264653", label="mechanics Newton iters")
    axes[1, 1].plot(it, design_iters, marker="s", color="#e76f51", label="design Newton iters")
    ax2 = axes[1, 1].twinx()
    ax2.step(it, p_penal, where="mid", color="#8d99ae", linestyle="--", label="SIMP p")
    axes[1, 1].set_title("Inner work and staircase continuation")
    axes[1, 1].set_xlabel("Outer iteration")
    axes[1, 1].set_ylabel("Newton iterations")
    ax2.set_ylabel("penalization p")
    axes[1, 1].grid(alpha=0.25)
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(CONV_PNG, dpi=200)
    plt.close(fig)


def _write_history_csv(history: list[dict]) -> None:
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
        "mechanics_iters",
        "design_iters",
        "mechanics_energy",
        "design_energy",
        "mechanics_message",
        "design_message",
    ]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _solver_command(params: dict) -> str:
    return "\n".join(
        [
            "./.venv/bin/python src/problems/topology/jax/solve_topopt_jax.py \\",
            f"    --nx {params['nx']} --ny {params['ny']} --length {params['length']} --height {params['height']} \\",
            f"    --traction {params['traction']} --load_fraction {params['load_fraction']} \\",
            f"    --fixed_pad_cells {params['fixed_pad_cells']} --load_pad_cells {params['load_pad_cells']} \\",
            f"    --volume_fraction_target {params['volume_fraction_target']} --theta_min {params['theta_min']} \\",
            f"    --solid_latent {params['solid_latent']} --young {params['young']} --poisson {params['poisson']} \\",
            f"    --alpha_reg {params['alpha_reg']} --ell_pf {params['ell_pf']} --mu_move {params['mu_move']} \\",
            f"    --beta_lambda {params['beta_lambda']} --volume_penalty {params['volume_penalty']} \\",
            f"    --p_start {params['p_start']} --p_max {params['p_max']} --p_increment {params['p_increment']} \\",
            f"    --continuation_interval {params['continuation_interval']} --outer_maxit {params['outer_maxit']} \\",
            f"    --outer_tol {params['outer_tol']} --volume_tol {params['volume_tol']} \\",
            f"    --mechanics_maxit {params['mechanics_maxit']} --design_maxit {params['design_maxit']} \\",
            f"    --tolf {params['tolf']} --tolg {params['tolg']} \\",
            (
                f"    --ksp_rtol {params['ksp_rtol']} --ksp_max_it {params['ksp_max_it']} "
                f"--save_outer_state_history --quiet \\"
            ),
            f"    --json_out {_asset_rel(JSON_PATH)} --state_out {_asset_rel(NPZ_PATH)}",
        ]
    )


def _make_report(result: dict, state: dict, mesh: CantileverTopologyMesh) -> str:
    history = [row for row in result["history"] if "compliance" in row]
    params = result["parameters"]
    solver_options = result["solver_options"]
    final = result["final_metrics"]
    theta = np.asarray(state["theta"], dtype=np.float64)
    gray_ratio = float(np.mean((theta > 0.1) & (theta < 0.9)))
    gray_ratio_05 = float(np.mean((theta > 0.05) & (theta < 0.95)))
    total_mech_iters = int(sum(row["mechanics_iters"] for row in history))
    total_design_iters = int(sum(row["design_iters"] for row in history))
    last = history[-1]
    solve_time = result["time"] - result["setup_time"]

    config_rows = [
        ["Mesh", f"{result['mesh']['nx']} x {result['mesh']['ny']}"],
        ["Elements", result["mesh"]["elements"]],
        ["Free displacement DOFs", result["mesh"]["displacement_free_dofs"]],
        ["Free design DOFs", result["mesh"]["design_free_dofs"]],
        ["Target volume fraction", _fmt(params["volume_fraction_target"], 4)],
        ["Staircase schedule", f"p = p + {params['p_increment']} every {params['continuation_interval']} outer iterations"],
        ["Final p target", _fmt(params["p_max"], 2)],
        ["Volume control", f"beta_lambda = {params['beta_lambda']}, volume_penalty = {params['volume_penalty']}"],
        ["Regularisation", f"alpha = {params['alpha_reg']}, ell = {params['ell_pf']}, mu_move = {params['mu_move']}"],
    ]

    derivative_rows = [
        [r"$\Pi_h(u; z_k)$", "`u_free`", "gradient and sparse Hessian of the mechanics subproblem"],
        [r"$G_h(z; u_{k+1})$", "`z_free`", "gradient and sparse Hessian of the design subproblem"],
    ]

    summary_rows = [
        ["Result", result["result"]],
        ["Outer iterations", final["outer_iterations"]],
        ["Final p", _fmt(final["final_p_penal"], 4)],
        ["Wall time [s]", _fmt(result["time"], 3)],
        ["JAX setup [s]", _fmt(result["setup_time"], 3)],
        ["Solve time [s]", _fmt(solve_time, 3)],
        ["Final compliance", _fmt(final["final_compliance"], 6)],
        ["Final volume fraction", _fmt(final["final_volume_fraction"], 6)],
        ["Final volume error", _fmt(last["volume_residual"], 6)],
        ["Final state change", _fmt(last["theta_state_change"], 6)],
        ["Final design change", _fmt(last["design_change"], 6)],
        ["Final compliance change", _fmt(last["compliance_change"], 6)],
        ["Total mechanics Newton iterations", total_mech_iters],
        ["Total design Newton iterations", total_design_iters],
    ]

    quality_rows = [
        ["Gray fraction on 0.1 < theta < 0.9", _fmt(gray_ratio, 4)],
        ["Gray fraction on 0.05 < theta < 0.95", _fmt(gray_ratio_05, 4)],
        ["theta_min", _fmt(final["final_theta_min"], 6)],
        ["theta_max", _fmt(final["final_theta_max"], 6)],
    ]

    timing_rows = []
    for stage_name, timings in result["jax_setup_timing"].items():
        for key, value in timings.items():
            timing_rows.append([f"{stage_name}: {key}", _fmt(value, 6)])

    history_rows = []
    for row in history:
        history_rows.append(
            [
                row["outer_iter"],
                _fmt(row["p_penal"], 3),
                _fmt(row["p_step"], 3),
                _fmt(row["lambda_effective"], 5),
                row["mechanics_iters"],
                row["design_iters"],
                _fmt(row["compliance"], 6),
                _fmt(row["volume_fraction"], 6),
                _fmt(row["volume_residual"], 6),
                _fmt(row["theta_state_change"], 6),
                _fmt(row["compliance_change"], 6),
            ]
        )

    solver_cmd = _solver_command({**params, **CASE_PARAMS})
    report = dedent(
        f"""\
        # {REPORT_TITLE}

        Date: {date.today()}

        This report fixes the JAX topology benchmark to a single clean reference
        configuration: a fine `192 x 96` cantilever mesh, a staggered
        displacement/design solve, and a fixed staircase SIMP continuation.
        The intent is no longer to compare continuation heuristics; it is to
        document one compact working implementation that demonstrates how the
        repository can define energies in JAX, autodifferentiate them, assemble
        sparse Hessians on fixed graphs, and solve a nontrivial benchmark.

        ## Benchmark Definition

        The domain is the cantilever rectangle

        $$
        \\Omega = [0, L] \\times [0, H],
        $$

        with the left edge clamped and a downward traction patch on the right
        edge. The design variable is a latent nodal field $z$, mapped to a
        physical density by

        $$
        \\theta(z) = \\theta_{{\\min}} + (1 - \\theta_{{\\min}})\\,\\sigma(z),
        \\qquad
        \\sigma(z) = \\frac{{1}}{{1 + e^{{-z}}}}.
        $$

        The mechanics energy for fixed design is

        $$
        \\Pi_h(u; z_k)
        =
        \\sum_e A_e\\,\\frac{{1}}{{2}}\\,\\theta_e(z_k)^p\\,\\varepsilon_e(u)^T C\\,\\varepsilon_e(u)
        - f^T u,
        $$

        and the frozen design energy is

        $$
        G_h(z; u_{{k+1}})
        =
        \\sum_e A_e
        \\left[
        e_{{k,e}}\\,\\theta_e(z)^{{-p}}
        + \\lambda_k\\,\\theta_e(z)
        + \\alpha\\left(\\frac{{\\ell}}{{2}}|\\nabla \\theta_e(z)|^2 + \\frac{{W(\\theta_e(z))}}{{\\ell}}\\right)
        + \\frac{{\\mu}}{{2}}(\\bar z_e - \\bar z^{{old}}_e)^2
        \\right],
        $$

        with

        $$
        W(\\theta) = \\theta^2(1-\\theta)^2.
        $$

        The SIMP exponent follows the fixed staircase schedule

        $$
        p_{{k+1}} =
        \\min\\bigl(p_{{\\max}},\\, p_k + \\Delta p\\bigr)
        \\quad \\text{{every }} m \\text{{ outer iterations}},
        $$

        using `\\Delta p = {params['p_increment']}` and `m = {params['continuation_interval']}`.

        ## Reference Configuration

        {_markdown_table(["Knob", "Value"], config_rows)}

        ## Minimal JAX Problem Definition

        The problem-specific input to JAX is just the energy definition. The
        following two functions are the parts that are actually autodifferentiated:

        {ENERGY_SNIPPET}

        ## Where JAX Is Used

        The solver asks JAX for derivatives with respect to the free unknowns of
        each subproblem only:

        {_markdown_table(["Energy", "Differentiated with respect to", "What is generated"], derivative_rows)}

        The corresponding calls are:

        {AUTODIFF_SNIPPET}

        `EnergyDerivator` provides the energy, gradient, and sparse Hessian-value
        callbacks. `HessSolverGenerator` then builds the linear solve stage used
        inside Newton: AMG for mechanics and a direct sparse solve for the design
        subproblem.

        The continuation itself is intentionally fixed and minimal:

        {STAIRCASE_SNIPPET}

        ## Solver Structure

        ### Outer staggered loop

        {OUTER_PSEUDOCODE}

        ### Inner Newton solve

        {NEWTON_PSEUDOCODE}

        ## Geometry And Final State

        ![Mesh preview]({_asset_rel(MESH_PNG)})

        ![Final state]({_asset_rel(STATE_PNG)})

        The density plot is elementwise constant: each triangle is coloured by
        its average density $\\theta_e$.

        ## Convergence History

        ![Convergence history]({_asset_rel(CONV_PNG)})

        ## Density Evolution

        ![Density evolution]({_asset_rel(DENSITY_GIF)})

        ## Run Summary

        {_markdown_table(["Metric", "Value"], summary_rows)}

        ## Density Quality Indicators

        {_markdown_table(["Indicator", "Value"], quality_rows)}

        ## JAX Setup Timings

        {_markdown_table(["Stage", "Time [s]"], timing_rows)}

        ## Outer Iteration Table

        {_markdown_table(
            [
                "k",
                "p",
                "dp",
                "lambda_eff",
                "mech iters",
                "design iters",
                "compliance",
                "volume",
                "vol error",
                "state change",
                "comp change",
            ],
            history_rows,
        )}

        ## Artifacts

        - JSON result: `{_asset_rel(JSON_PATH)}`
        - Final state: `{_asset_rel(NPZ_PATH)}`
        - Outer-history CSV: `{_asset_rel(CSV_PATH)}`
        - Mesh figure: `{_asset_rel(MESH_PNG)}`
        - Final-state figure: `{_asset_rel(STATE_PNG)}`
        - Convergence figure: `{_asset_rel(CONV_PNG)}`
        - Density-evolution GIF: `{_asset_rel(DENSITY_GIF)}`

        ## Reproduction

        Regenerate the benchmark report and assets with:

        ```bash
        {REPORT_GENERATOR_CMD}
        ```

        Run the solver directly with the same fine-grid staircase setup:

        ```bash
        {solver_cmd}
        ```
        """
    )
    report = _strip_report_indent(report)
    return report.strip() + "\n"


def generate_benchmark_assets(
    *,
    case_params: dict | None = None,
    asset_dir: Path | None = None,
    report_path: Path | None = None,
) -> tuple[dict, dict, CantileverTopologyMesh]:
    global JSON_PATH, NPZ_PATH, CSV_PATH, MESH_PNG, STATE_PNG, CONV_PNG, DENSITY_GIF, ASSET_DIR

    case_params = dict(CASE_PARAMS if case_params is None else case_params)
    asset_dir = (ASSET_DIR if asset_dir is None else Path(asset_dir)).resolve()
    report_path = (REPORT_PATH if report_path is None else Path(report_path)).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    old_paths = (JSON_PATH, NPZ_PATH, CSV_PATH, MESH_PNG, STATE_PNG, CONV_PNG, DENSITY_GIF, ASSET_DIR)
    JSON_PATH = asset_dir / "report_run.json"
    NPZ_PATH = asset_dir / "report_state.npz"
    CSV_PATH = asset_dir / "report_outer_history.csv"
    MESH_PNG = asset_dir / "mesh_preview.png"
    STATE_PNG = asset_dir / "final_state.png"
    CONV_PNG = asset_dir / "convergence_history.png"
    DENSITY_GIF = asset_dir / "density_evolution.gif"
    ASSET_DIR = asset_dir

    try:
        result, state = run_topology_optimisation(
            **case_params,
            save_outer_state_history=True,
        )
        with JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        np.savez(
            NPZ_PATH,
            coords=state["coords"],
            triangles=state["triangles"],
            theta=state["theta"],
            u=state["u"],
            z=state["z"],
            theta_history=np.stack([snap["theta"] for snap in state["theta_history"]], axis=0),
            theta_history_outer=np.array([snap["outer_iter"] for snap in state["theta_history"]], dtype=np.int32),
            theta_history_p=np.array([snap["p_penal"] for snap in state["theta_history"]], dtype=np.float64),
        )
        _write_history_csv(result["history"])
        mesh = CantileverTopologyMesh(
            nx=case_params["nx"],
            ny=case_params["ny"],
            length=case_params["length"],
            height=case_params["height"],
            traction=case_params["traction"],
            load_fraction=case_params["load_fraction"],
            fixed_pad_cells=case_params["fixed_pad_cells"],
            load_pad_cells=case_params["load_pad_cells"],
        )
        _make_mesh_figure(mesh)
        _make_state_figure(mesh, state)
        _make_convergence_figure(result)
        _make_density_gif(mesh, state, result)
        report_path.write_text(_make_report(result, state, mesh), encoding="utf-8")
        return result, state, mesh
    finally:
        JSON_PATH, NPZ_PATH, CSV_PATH, MESH_PNG, STATE_PNG, CONV_PNG, DENSITY_GIF, ASSET_DIR = old_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    _, _, _ = generate_benchmark_assets(asset_dir=args.asset_dir, report_path=args.report_path)
    report_path = Path(args.report_path).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    print(f"Wrote {_asset_rel(report_path)}")
    print(f"Artifacts in {_asset_rel(asset_dir)}")


if __name__ == "__main__":
    main()
