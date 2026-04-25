#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.analysis.hyperelastic_companion_common import centerline_profile


DEFAULT_SUMMARY = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "raw_results"
    / "jax_fem_hyperelastic_baseline"
    / "comparison_summary.json"
)


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=220, bbox_inches="tight")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _energy_history_figure(summary: dict[str, object], out_base: Path) -> None:
    step_rows = [dict(row) for row in summary["step_rows"]]
    steps = [int(row["step"]) for row in step_rows]
    repo = [float(row["repo_energy"]) for row in step_rows]
    jax_fem = [float(row["jax_fem_energy"]) for row in step_rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.plot(steps, repo, marker="o", linewidth=2.0, label="Repo serial direct")
    ax.plot(steps, jax_fem, marker="s", linewidth=2.0, label="JAX-FEM serial")
    ax.set_xlabel("Load step")
    ax.set_ylabel("Stored energy")
    ax.set_title("Hyperelastic companion energy path")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    _save(fig, out_base)
    plt.close(fig)


def _centerline_figure(summary: dict[str, object], out_base: Path) -> None:
    impls = [dict(row) for row in summary["implementations"]]
    repo_state = np.load(impls[0]["state_npz"])
    jax_state = np.load(impls[1]["state_npz"])
    coords_ref = np.asarray(repo_state["coords_ref"], dtype=np.float64)
    repo_profile = centerline_profile(coords_ref, np.asarray(repo_state["displacement"], dtype=np.float64))
    jax_profile = centerline_profile(coords_ref, np.asarray(jax_state["displacement"], dtype=np.float64))
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.plot(repo_profile["x"], repo_profile["ux"], marker="o", linewidth=2.0, label="Repo serial direct")
    ax.plot(jax_profile["x"], jax_profile["ux"], marker="s", linewidth=2.0, label="JAX-FEM serial")
    ax.set_xlabel("Reference x")
    ax.set_ylabel("Centerline $u_x$")
    ax.set_title("Centerline axial displacement")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    _save(fig, out_base)
    plt.close(fig)


def _deformed_overlay_figure(summary: dict[str, object], out_base: Path) -> None:
    impls = [dict(row) for row in summary["implementations"]]
    repo_state = np.load(impls[0]["state_npz"])
    jax_state = np.load(impls[1]["state_npz"])
    coords_ref = np.asarray(repo_state["coords_ref"], dtype=np.float64)
    repo_final = np.asarray(repo_state["coords_final"], dtype=np.float64)
    jax_final = np.asarray(jax_state["coords_final"], dtype=np.float64)

    face_tol = 1.0e-12
    x_min = float(np.min(coords_ref[:, 0]))
    x_max = float(np.max(coords_ref[:, 0]))
    mask = np.isclose(coords_ref[:, 0], x_min, atol=face_tol) | np.isclose(coords_ref[:, 0], x_max, atol=face_tol)
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.scatter(repo_final[mask, 0], repo_final[mask, 1], s=10.0, alpha=0.8, label="Repo serial direct")
    ax.scatter(jax_final[mask, 0], jax_final[mask, 1], s=10.0, alpha=0.8, marker="x", label="JAX-FEM serial")
    ax.set_xlabel("Deformed x")
    ax.set_ylabel("Deformed y")
    ax.set_title("End-face overlay in the deformed configuration")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    _save(fig, out_base)
    plt.close(fig)


def _write_report(summary: dict[str, object], out_path: Path) -> None:
    metrics = dict(summary["final_metrics"])
    fairness = dict(summary["fairness_gate"])
    timing = dict(summary["timing_medians_s"])
    lines = [
        "# JAX-FEM Hyperelastic Companion Baseline",
        "",
        "## Fairness Gate",
        "",
        f"- passed: `{bool(fairness['passed'])}`",
        f"- policy: {fairness['policy']}",
        "",
        "## Final Metrics",
        "",
        f"- energy relative difference: `{float(metrics['energy_rel_diff']):.3e}`",
        f"- displacement-field relative L2: `{float(metrics['field_relative_l2']):.3e}`",
        f"- centerline relative L2: `{float(metrics['centerline_relative_l2']):.3e}`",
        f"- u_max curve relative L2: `{float(metrics['umax_curve_relative_l2']):.3e}`",
        "",
        "## Timing Medians",
        "",
        "| implementation | median wall time [s] |",
        "| --- | ---: |",
    ]
    for name, value in timing.items():
        lines.append(f"| `{name}` | `{float(value):.3f}` |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures and a report for the JAX-FEM hyperelastic baseline.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = _load_json(args.summary)
    out_dir = args.out_dir if args.out_dir is not None else args.summary.parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    _energy_history_figure(summary, out_dir / "energy_history")
    _centerline_figure(summary, out_dir / "centerline_profile")
    _deformed_overlay_figure(summary, out_dir / "deformed_overlay")
    _write_report(summary, out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
