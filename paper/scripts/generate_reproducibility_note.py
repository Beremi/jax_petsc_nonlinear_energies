#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from common import BUILD_ROOT, REPO_ROOT, ensure_paper_dirs, read_json, write_text


DEFAULT_OUTPUT = BUILD_ROOT / "reproducibility_note.md"
P3D_VALIDATION_MANIFEST = REPO_ROOT / "artifacts/raw_results/plasticity3d_validation/validation_manifest.json"
P3D_ABLATION_SUMMARY = REPO_ROOT / "artifacts/raw_results/plasticity3d_derivative_ablation/comparison_summary.json"
JAX_FEM_BASELINE_MANIFEST = REPO_ROOT / "artifacts/raw_results/jax_fem_hyperelastic_baseline/run_manifest.json"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _python_version(python_bin: Path) -> str:
    return subprocess.check_output([str(python_bin), "--version"], cwd=REPO_ROOT, text=True).strip()


def _repo_relative_display(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact reproducibility note for the paper artifact bundle.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    args = parser.parse_args()

    ensure_paper_dirs()
    validation_manifest = read_json(P3D_VALIDATION_MANIFEST)
    ablation_summary = read_json(P3D_ABLATION_SUMMARY)
    baseline_manifest = read_json(JAX_FEM_BASELINE_MANIFEST)

    lines = [
        "# Reproducibility Note",
        "",
        f"- git commit: `{_git_head()}`",
        f"- host: `{platform.node()}`",
        f"- platform: `{platform.platform()}`",
        f"- repo python: `{_python_version(args.repo_python)}`",
        "",
        "## Artifact Commands",
        "",
        f"- Plasticity3D validation manifest: `{Path(validation_manifest['runner']).name}` -> `{P3D_VALIDATION_MANIFEST.relative_to(REPO_ROOT)}`",
        "- Regenerate validation assets: `./.venv/bin/python experiments/analysis/generate_plasticity3d_validation_assets.py`",
        "- Regenerate derivative ablation assets: `./.venv/bin/python experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py`",
        f"- JAX-FEM baseline runner: `{_repo_relative_display(baseline_manifest['jax_fem_python'])}` + `experiments/runners/run_jax_fem_hyperelastic_baseline.py`",
        "- Regenerate JAX-FEM baseline assets: `./.venv/bin/python experiments/analysis/generate_jax_fem_hyperelastic_baseline_assets.py`",
        "",
        "## Locked Cases",
        "",
        f"- Plasticity3D validation schedule: `{validation_manifest['validation_contract']['schedule']}`",
        f"- Plasticity3D derivative routes: `{[row['route'] for row in ablation_summary['rows']]}`",
        f"- JAX-FEM baseline schedule: `{baseline_manifest['schedule']}`",
        "",
    ]
    write_text(args.out, "\n".join(lines))
    print(args.out)


if __name__ == "__main__":
    main()
