from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_derivative_ablation_asset_generator_writes_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "comparison_summary.json"
    out_dir = tmp_path / "assets"
    route_dirs = []
    rows = []
    for idx, (name, label) in enumerate(
        (
            ("element_ad", "Element AD"),
            ("constitutive_ad", "Constitutive AD"),
            ("colored_sfd", "Colored SFD"),
        ),
        start=1,
    ):
        case_dir = tmp_path / name / "measure_01"
        case_dir.mkdir(parents=True, exist_ok=True)
        output_json = case_dir / "output.json"
        output_json.write_text(
            json.dumps(
                {
                    "history": [
                        {"step_rel": 1.0},
                        {"step_rel": 1.0e-1 / idx},
                        {"step_rel": 1.0e-2 / idx},
                    ]
                }
            ),
            encoding="utf-8",
        )
        route_dirs.append(case_dir)
        rows.append(
            {
                "route": name,
                "display_label": label,
                "assembly_backend": "local",
                "solver_backend": "local",
                "ranks": 8,
                "measured_runs": 1,
                "status": "completed",
                "solver_success": True,
                "median_wall_time_s": float(10 * idx),
                "median_solve_time_s": float(8 * idx),
                "median_nit": float(6 + idx),
                "median_linear_iterations_total": float(100 * idx),
                "median_final_metric": float(1.0e-3 / idx),
                "final_metric_name": "relative_correction",
                "median_energy": float(-1.0 * idx),
                "median_omega": float(2.0 * idx),
                "median_u_max": float(0.1 * idx),
                "run_rows": [
                    {
                        "output_json": str(output_json),
                        "nit": 3,
                    }
                ],
            }
        )
    summary_path.write_text(
        json.dumps(
            {
                "runner": "plasticity3d_derivative_ablation",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py",
            "--summary-json",
            str(summary_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "derivative_ablation_bars.png").exists()
    assert (out_dir / "derivative_ablation_convergence.png").exists()
