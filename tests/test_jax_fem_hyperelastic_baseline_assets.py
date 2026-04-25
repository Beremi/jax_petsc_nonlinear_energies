from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.analysis.generate_jax_fem_hyperelastic_baseline_assets import main


def _write_state(path: Path, scale: float) -> None:
    coords_ref = np.asarray(
        [
            [0.0, -0.005, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.005, 0.0],
            [0.2, -0.005, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.005, 0.0],
            [0.4, -0.005, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.005, 0.0],
        ],
        dtype=np.float64,
    )
    displacement = np.zeros_like(coords_ref)
    displacement[:, 0] = scale * coords_ref[:, 0]
    coords_final = coords_ref + displacement
    tetrahedra = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=np.int32)
    np.savez(
        path,
        coords_ref=coords_ref,
        coords_final=coords_final,
        displacement=displacement,
        tetrahedra=tetrahedra,
    )


def test_jax_fem_baseline_assets_generator(tmp_path, monkeypatch):
    repo_state = tmp_path / "repo_state.npz"
    jax_state = tmp_path / "jax_state.npz"
    _write_state(repo_state, 0.02)
    _write_state(jax_state, 0.019)
    summary = {
        "implementations": [
            {"name": "repo_serial_direct", "state_npz": str(repo_state)},
            {"name": "jax_fem_umfpack_serial", "state_npz": str(jax_state)},
        ],
        "step_rows": [
            {
                "step": 1,
                "repo_energy": 1.0,
                "jax_fem_energy": 1.01,
            },
            {
                "step": 2,
                "repo_energy": 1.5,
                "jax_fem_energy": 1.49,
            },
        ],
        "final_metrics": {
            "energy_rel_diff": 1.0e-2,
            "field_relative_l2": 2.0e-2,
            "centerline_relative_l2": 1.5e-2,
            "umax_curve_relative_l2": 1.0e-2,
        },
        "fairness_gate": {
            "passed": True,
            "policy": "test policy",
        },
        "timing_medians_s": {
            "repo_serial_direct": 0.4,
            "jax_fem_umfpack_serial": 0.6,
        },
    }
    summary_path = tmp_path / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_dir = tmp_path / "assets"

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_jax_fem_hyperelastic_baseline_assets.py",
            "--summary",
            str(summary_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    main()

    assert (out_dir / "energy_history.pdf").exists()
    assert (out_dir / "centerline_profile.pdf").exists()
    assert (out_dir / "deformed_overlay.pdf").exists()
    assert (out_dir / "REPORT.md").exists()
