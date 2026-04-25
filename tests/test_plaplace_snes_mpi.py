from __future__ import annotations

import json
import importlib
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependency_available(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message depends on local stack
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def test_plaplace_snes_level5_np2_completes(tmp_path: Path):
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is required for the parallel FEniCS smoke test")
    available, reason = _optional_dependency_available("dolfinx")
    if not available:
        pytest.skip(f"DOLFINx stack is unavailable: {reason}")

    out_path = tmp_path / "plaplace_snes_np2.json"
    cmd = [
        mpiexec,
        "-n",
        "2",
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(REPO_ROOT / "src" / "problems" / "plaplace" / "fenics" / "solve_pLaplace_snes_newton.py"),
        "--levels",
        "5",
        "--json",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["nprocs"] == 2
    assert payload["results"][0]["mesh_level"] == 5
