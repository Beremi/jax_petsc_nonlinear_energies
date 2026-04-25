#!/usr/bin/env python3
"""Generate plots and a short report for the Plasticity3D derivative ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_derivative_ablation" / "comparison_summary.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_derivative_ablation" / "assets"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=260)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    out_dir = Path(args.out_dir).resolve()
    rows = [dict(row) for row in summary["rows"]]
    labels = [str(row["display_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), dpi=180)
    axes[0].bar(x, [float(row["median_wall_time_s"]) for row in rows], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("median wall time [s]")
    axes[0].set_title("Derivative-route wall time")

    axes[1].bar(x, [float(row["median_linear_iterations_total"]) for row in rows], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("median total linear iterations")
    axes[1].set_title("Derivative-route linear work")
    fig.tight_layout()
    _save_figure(fig, out_dir / "derivative_ablation_bars")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    for row, color in zip(rows, ("#4c78a8", "#f58518", "#54a24b"), strict=True):
        if not row["run_rows"]:
            continue
        first = dict(row["run_rows"][0])
        iterations = np.arange(1, int(first["nit"]) + 1, dtype=np.int64)
        values = []
        for run_row in row["run_rows"]:
            result_path = REPO_ROOT / str(run_row["output_json"])
            payload = _read_json(result_path)
            history = list(payload.get("history", []))
            values.append(
                np.asarray(
                    [float(item.get("step_rel", item.get("metric", np.nan))) for item in history],
                    dtype=np.float64,
                )
            )
        min_len = min(len(v) for v in values)
        curve = np.mean(np.vstack([v[:min_len] for v in values]), axis=0)
        ax.semilogy(np.arange(1, min_len + 1), curve, marker="o", color=color, label=str(row["display_label"]))
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("relative correction")
    ax.set_title("Derivative-route convergence")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_dir / "derivative_ablation_convergence")

    report_lines = [
        "# Plasticity3D derivative-route ablation",
        "",
        "| route | median wall [s] | median solve [s] | median nit | median linear iters | median energy | median omega | median u_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            "| {label} | {wall:.6f} | {solve:.6f} | {nit:.3f} | {linear:.3f} | {energy:.6f} | {omega:.6f} | {u_max:.6f} |".format(
                label=str(row["display_label"]),
                wall=float(row["median_wall_time_s"]),
                solve=float(row["median_solve_time_s"]),
                nit=float(row["median_nit"]),
                linear=float(row["median_linear_iterations_total"]),
                energy=float(row["median_energy"]),
                omega=float(row["median_omega"]),
                u_max=float(row["median_u_max"]),
            )
        )
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
