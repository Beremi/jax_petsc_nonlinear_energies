from __future__ import annotations

import json
from pathlib import Path

from experiments.runners import run_plasticity3d_derivative_ablation as runner


def _fake_output(route_name: str) -> dict[str, object]:
    scale = {
        "local": 1.0,
        "local_constitutiveAD": 0.8,
        "local_sfd": 1.3,
    }[route_name]
    return {
        "status": "completed",
        "solver_success": True,
        "solve_time": 100.0 * scale,
        "nit": 8,
        "linear_iterations_total": 376,
        "final_metric": 1.0e-3 * scale,
        "final_metric_name": "relative_correction",
        "energy": -3010860.0,
        "omega": 6027722.0,
        "u_max": 0.7318479,
    }


def test_derivative_ablation_runner_emits_three_expected_routes(tmp_path: Path, monkeypatch) -> None:
    def fake_run_command(*, cmd: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
        assembly_backend = cmd[cmd.index("--assembly-backend") + 1]
        output_json = Path(cmd[cmd.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(_fake_output(assembly_backend)), encoding="utf-8")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")
        return 0, float({"local": 288.0, "local_constitutiveAD": 222.0, "local_sfd": 372.0}[assembly_backend])

    monkeypatch.setattr(runner, "_run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_plasticity3d_derivative_ablation.py",
            "--out-dir",
            str(tmp_path),
            "--source-root",
            str(tmp_path / "source-root"),
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ],
    )

    runner.main()

    summary = json.loads((tmp_path / "comparison_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    assert [row["route"] for row in rows] == ["element_ad", "constitutive_ad", "colored_sfd"]
    assert [row["display_label"] for row in rows] == ["Element AD", "Constitutive AD", "Colored SFD"]
    for row in rows:
        assert row["status"] == "completed"
        assert row["solver_success"] is True
        assert set(
            [
                "median_wall_time_s",
                "median_solve_time_s",
                "median_nit",
                "median_linear_iterations_total",
                "median_final_metric",
                "median_energy",
                "median_omega",
                "median_u_max",
            ]
        ).issubset(row.keys())
