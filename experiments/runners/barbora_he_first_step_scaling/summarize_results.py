#!/usr/bin/env python3
"""Summarize Barbora HyperElasticity first-step scaling outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(output_json: Path) -> dict:
    case_dir = output_json.parent
    metadata_path = case_dir / "case_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    payload = _load_json(output_json)
    result = payload.get("result", {})
    steps = list(result.get("steps", []))
    step = steps[0] if steps else {}

    return {
        "case_id": metadata.get("case_id", case_dir.parent.name),
        "job_id": metadata.get("job_id", case_dir.name.removeprefix("job_")),
        "backend": metadata.get("backend", payload.get("case", {}).get("backend", "")),
        "level": metadata.get("he_level", result.get("mesh_level", "")),
        "nodes": metadata.get("nodes", ""),
        "ranks_per_socket": metadata.get("ranks_per_socket", ""),
        "ranks_per_node": metadata.get("ranks_per_node", ""),
        "total_ranks": metadata.get("total_ranks", result.get("metadata", {}).get("nprocs", "")),
        "cpus_per_task": metadata.get("cpus_per_task", ""),
        "total_steps": metadata.get("total_steps", payload.get("case", {}).get("total_steps", "")),
        "completed_steps": len(steps),
        "total_dofs": result.get("total_dofs", ""),
        "free_dofs": result.get("free_dofs", ""),
        "setup_time_s": result.get("setup_time", ""),
        "first_step_time_s": step.get("time", ""),
        "total_time_s": result.get("total_time", ""),
        "newton_iters": step.get("nit", ""),
        "linear_iters": step.get("linear_iters", ""),
        "energy": step.get("energy", ""),
        "message": step.get("message", ""),
        "output_json": str(output_json),
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    headers = [
        "backend",
        "level",
        "nodes",
        "ranks_per_socket",
        "total_ranks",
        "total_dofs",
        "first_step_time_s",
        "newton_iters",
        "linear_iters",
        "energy",
        "message",
    ]
    lines = [
        "# HyperElasticity First-Step Scaling Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_root", help="Campaign output root")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    output_paths = sorted(out_root.glob("cases/*/job_*/output.json"))
    rows = [_row(path) for path in output_paths]
    rows.sort(
        key=lambda row: (
            str(row["backend"]),
            int(row["level"] or 0),
            int(row["nodes"] or 0),
            int(row["ranks_per_socket"] or 0),
        )
    )

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "results_summary.csv"
    md_path = summary_dir / "results_summary.md"

    fieldnames = [
        "case_id",
        "job_id",
        "backend",
        "level",
        "nodes",
        "ranks_per_socket",
        "ranks_per_node",
        "total_ranks",
        "cpus_per_task",
        "total_steps",
        "completed_steps",
        "total_dofs",
        "free_dofs",
        "setup_time_s",
        "first_step_time_s",
        "total_time_s",
        "newton_iters",
        "linear_iters",
        "energy",
        "message",
        "output_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(md_path, rows)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
