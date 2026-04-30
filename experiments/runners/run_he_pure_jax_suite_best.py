#!/usr/bin/env python3
"""Run the pure-JAX HE benchmark suite with the frozen serial trust settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.benchmark.results import (
    summarize_pure_jax_load_step_case as _summarize_payload,
    sum_step_history as _sum_step_history,
    sum_step_linear as _sum_step_linear_iters,
    sum_step_linear_time as _sum_step_linear_time,
)
from src.problems.hyperelasticity.jax.solve_HE_jax_newton import run_level


DEFAULT_OUT_DIR = Path("artifacts/raw_results/he_pure_jax_stcg_best")
CASES = [
    (24, 1),
    (24, 2),
    (24, 3),
    (96, 1),
    (96, 2),
    (96, 3),
]
def _write_case_markdown(out_path: Path, payload: dict) -> None:
    steps = payload["steps"]
    lines = [
        f"# {payload['solver']} level {payload['level']} steps {payload['total_steps']}",
        "",
        "## Run Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total DOFs | {payload['total_dofs']} |",
        f"| free DOFs | {payload['free_dofs']} |",
        f"| setup time [s] | {payload['setup_time']:.6f} |",
        f"| total time [s] | {payload['time']:.6f} |",
        f"| total Newton iters | {payload['total_newton_iters']} |",
        f"| total linear iters | {payload['total_linear_iters']} |",
        f"| result | {payload['result']} |",
        "",
        "## Per-Step Summary",
        "",
        "| Step | Newton iters | Linear iters | Energy | Time [s] | Assembly [s] | PC init [s] | KSP solve [s] | Line search [s] | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for step in steps:
        lines.append(
            "| {step} | {nit} | {lit} | {energy:.9f} | {time:.6f} | {assembly:.6f} | {pc:.6f} | {solve:.6f} | {ls:.6f} | {status} |".format(
                step=step["step"],
                nit=step["iters"],
                lit=_sum_step_linear_iters(step),
                energy=float(step["energy"]),
                time=float(step["time"]),
                assembly=_sum_step_linear_time(step, "assemble_time"),
                pc=_sum_step_linear_time(step, "pc_setup_time"),
                solve=_sum_step_linear_time(step, "solve_time"),
                ls=_sum_step_history(step, "t_ls"),
                status=step["message"],
            )
        )

    for step in steps:
        lines.extend(
            [
                "",
                f"## Step {step['step']} Newton Detail",
                "",
                "| Newton | Energy | dE | Grad norm | Alpha | KSP it | TR rho | TR radius | Direction |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rec in step.get("history", []):
            direction = "grad fallback" if rec.get("used_gradient_fallback", False) else "newton"
            lines.append(
                "| {it} | {energy:.9f} | {dE:.9e} | {grad:.9e} | {alpha:.9f} | {ksp} | {rho:.9e} | {radius:.9f} | {direction} |".format(
                    it=int(rec.get("it", 0)),
                    energy=float(rec.get("energy", 0.0)),
                    dE=float(rec.get("dE", 0.0)),
                    grad=float(rec.get("grad_norm", 0.0)),
                    alpha=float(rec.get("alpha", 0.0)),
                    ksp=int(rec.get("ksp_its", 0)),
                    rho=float(rec.get("trust_ratio", float("nan"))),
                    radius=float(rec.get("trust_radius", float("nan"))),
                    direction=direction,
                )
            )

    out_path.write_text("\n".join(lines) + "\n")


def _write_summary_markdown(out_path: Path, rows: list[dict]) -> None:
    lines = [
        "# Pure JAX HE STCG Summary",
        "",
        "Raw case data live next to this file as `pure_jax_steps*_l*.json` and `*.md`.",
        "",
        "| Solver | Level | Total steps | Total DOFs | Free DOFs | Total time [s] | Newton | Linear | Max step [s] | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {solver} | {level} | {total_steps} | {dofs} | {free_dofs} | {time:.6f} | {newton} | {linear} | {max_step:.6f} | {result} |".format(
                solver=row["solver"],
                level=row["level"],
                total_steps=row["total_steps"],
                dofs=row["total_dofs"],
                free_dofs=row["free_dofs"],
                time=row["time"],
                newton=row["total_newton_iters"],
                linear=row["total_linear_iters"],
                max_step=row["max_step_time"],
                result=row["result"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of mesh levels to run. Defaults to the frozen full suite.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of total-step counts to run. Defaults to the frozen full suite.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    allowed_levels = set(args.levels) if args.levels is not None else None
    allowed_total_steps = set(args.total_steps) if args.total_steps is not None else None
    cases = [
        (total_steps, level)
        for total_steps, level in CASES
        if (allowed_levels is None or level in allowed_levels)
        and (allowed_total_steps is None or total_steps in allowed_total_steps)
    ]
    if not cases:
        raise SystemExit("No pure-JAX HE cases selected.")

    for total_steps, level in cases:
        payload = run_level(
            level=level,
            steps=total_steps,
            total_steps=total_steps,
            linesearch_tol=1e-1,
            ksp_rtol=1e-1,
            ksp_max_it=30,
            tolf=1e-4,
            tolg=1e-3,
            tolg_rel=1e-3,
            tolx_rel=1e-3,
            tolx_abs=1e-10,
            require_all_convergence=True,
            use_trust_region=True,
            trust_radius_init=0.5,
            trust_radius_min=1e-8,
            trust_radius_max=1e6,
            trust_shrink=0.5,
            trust_expand=1.5,
            trust_eta_shrink=0.05,
            trust_eta_expand=0.75,
            trust_max_reject=6,
            trust_subproblem_line_search=True,
            verbose=False,
        )

        stem = f"pure_jax_steps{total_steps}_l{level}"
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(json.dumps(payload, indent=2))
        _write_case_markdown(md_path, payload)

        rows.append(_summarize_payload(payload))

    summary = {"rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_summary_markdown(out_dir / "summary.md", rows)


if __name__ == "__main__":
    main()
