#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import FIGURES_ROOT, TABLES_ROOT, ensure_paper_dirs


REQUIRED_FIGURES = [
    "framework_overview.pdf",
    "derivative_paths.pdf",
    "globalization_schematic.pdf",
    "coloring_schematic.pdf",
    "autodiff_modes.pdf",
    "plaplace_state.pdf",
    "ginzburg_landau_state.pdf",
    "hyperelasticity_state.pdf",
    "plasticity2d_state.pdf",
    "plasticity3d_state.pdf",
    "plasticity3d_strain.pdf",
    "topology_density.pdf",
    "plasticity3d_recommended_scaling.pdf",
    "plasticity3d_recommended_components.pdf",
    "plasticity3d_local_vs_source.pdf",
    "plasticity3d_sourcefixed_compare.pdf",
    "plasticity3d_validation_layer1a_boundary.pdf",
    "plasticity3d_validation_umax_curve.pdf",
    "plasticity3d_derivative_ablation_bars.pdf",
    "jax_fem_hyperelastic_baseline_energy_history.pdf",
    "jax_fem_hyperelastic_baseline_centerline.pdf",
]

REQUIRED_TABLES = [
    "implementation_capability_matrix.tex",
    "benchmark_specification_matrix.tex",
    "sota_framework_comparison.tex",
    "plasticity3d_recommended_scaling.tex",
    "plasticity3d_local_vs_source.tex",
    "plasticity3d_sourcefixed_alternative.tex",
    "topology_summary.tex",
    "reference_availability.tex",
    "plasticity3d_validation_summary.tex",
    "plasticity3d_derivative_ablation.tex",
    "jax_fem_hyperelastic_baseline.tex",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the paper asset generation produced the expected files.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_ROOT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_ROOT)
    args = parser.parse_args()
    ensure_paper_dirs()
    missing: list[str] = []
    for name in REQUIRED_FIGURES:
        path = args.figures_dir / name
        if not path.exists():
            missing.append(str(path))
    for name in REQUIRED_TABLES:
        path = args.tables_dir / name
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise SystemExit("Missing paper assets:\n" + "\n".join(missing))
    print("Paper assets validated.")


if __name__ == "__main__":
    main()
