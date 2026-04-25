from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from experiments.runners import run_jax_fem_hyperelastic_baseline as baseline_runner


@pytest.mark.skipif(
    os.environ.get("RUN_EXTERNAL_BASELINE_SMOKE") != "1",
    reason="Opt-in external baseline smoke test.",
)
def test_jax_fem_worker_single_step_smoke(tmp_path: Path) -> None:
    env_python = baseline_runner.DEFAULT_ENV_PYTHON
    if not env_python.exists():
        pytest.skip(f"Missing external JAX-FEM env: {env_python}")

    env = os.environ.copy()
    extra_paths = [str(baseline_runner.REPO_ROOT)]
    if baseline_runner.DEFAULT_MAIN_SITE.exists():
        extra_paths.append(str(baseline_runner.DEFAULT_MAIN_SITE))
    if env.get("PYTHONPATH"):
        extra_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    out_json = tmp_path / "jax_fem_smoke.json"
    state_out = tmp_path / "jax_fem_smoke_state.npz"
    subprocess.run(
        [
            str(env_python),
            str(baseline_runner.WORKER_SCRIPT),
            "--level",
            "1",
            "--schedule",
            "0.0025",
            "--out",
            str(out_json),
            "--state-out",
            str(state_out),
        ],
        cwd=baseline_runner.REPO_ROOT,
        env=env,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["implementation"] == "jax_fem_umfpack_serial"
    assert len(payload["rows"]) == 1
    assert state_out.exists()
