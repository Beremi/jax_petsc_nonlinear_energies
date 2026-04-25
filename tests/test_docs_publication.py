from __future__ import annotations

from pathlib import Path

from experiments.analysis.docs_assets import common


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _doc_names(subdir: str) -> list[str]:
    return sorted(path.name for path in (DOCS_ROOT / subdir).glob("*.md"))
ASSETS_ROOT = DOCS_ROOT / "assets"
BUILD_ROOT = REPO_ROOT / "experiments" / "analysis" / "docs_assets"
BANNED_SNIPPETS = (
    "/home/",
    "/workdir/",
    "/usr/bin/mpiexec",
    "artifacts/figures/img",
    "docs/benchmarks/",
    "docs/overview/",
    "replications/",
    "experiment_scripts/",
)


def test_current_docs_structure_exists() -> None:
    expected = [
        DOCS_ROOT / "README.md",
        DOCS_ROOT / "setup" / "quickstart.md",
        DOCS_ROOT / "setup" / "local_build.md",
        DOCS_ROOT / "problems" / "pLaplace.md",
        DOCS_ROOT / "problems" / "pLaplace_u3_thesis_replications.md",
        DOCS_ROOT / "problems" / "pLaplace_up_arctan.md",
        DOCS_ROOT / "problems" / "GinzburgLandau.md",
        DOCS_ROOT / "problems" / "HyperElasticity.md",
        DOCS_ROOT / "problems" / "Plasticity.md",
        DOCS_ROOT / "problems" / "Topology.md",
        DOCS_ROOT / "results" / "pLaplace.md",
        DOCS_ROOT / "results" / "GinzburgLandau.md",
        DOCS_ROOT / "results" / "HyperElasticity.md",
        DOCS_ROOT / "results" / "Plasticity.md",
        DOCS_ROOT / "results" / "Topology.md",
    ]
    for path in expected:
        assert path.exists(), path


def test_retired_overview_markdown_surface_is_gone() -> None:
    assert not (REPO_ROOT / "overview").exists()


def test_problem_pages_contain_required_sections() -> None:
    for name in ("pLaplace", "GinzburgLandau", "HyperElasticity", "Plasticity", "Topology"):
        text = (DOCS_ROOT / "problems" / f"{name}.md").read_text(encoding="utf-8")
        assert "## Mathematical Formulation" in text
        assert "## Commands Used" in text
        assert "![" in text
        assert ("## Maintained Implementations" in text) or ("## Implementation Status" in text)
        assert ("Energy Table Across Levels" in text) or ("Resolution / Objective Table" in text)
    assert ".gif" in (DOCS_ROOT / "problems" / "Topology.md").read_text(encoding="utf-8")


def test_plaplace_u3_thesis_replication_page_contains_required_sections() -> None:
    text = (DOCS_ROOT / "problems" / "pLaplace_u3_thesis_replications.md").read_text(encoding="utf-8")
    assert "Michaela Bailová" in text
    assert "## Thesis Problem Statement And Functionals" in text
    assert "## Thesis Geometries, Discretisation, And Seeds" in text
    assert "## Implementation Map" in text
    assert "../../src/problems/plaplace_u3/thesis/scripts/solve_case.py" in text
    assert "## RMPA Square Principal-Branch Replication" in text
    assert "## Square-With-Hole OA2 Study (Figure 5.13)" in text
    assert "## What Matches, What Needs Context, And What Does Not Match" in text
    assert "**Problem spec**" in text
    assert "**Column legend**" in text
    assert "**Discrepancy notes**" in text
    assert "low impact" in text
    assert "timing note:" in text
    assert "timing status" in text
    assert "runtime context" in text
    assert "thesis t[s]" in text
    assert "repo t[s]" in text
    assert "1 proc, serial python" in text
    assert "## Stage C Timing Summary" in text
    assert "## Convergence Diagnostics" in text
    assert "Table 5.12 is the thesis wall-time comparison" in text
    assert "timing complete" in text
    assert "timing unavailable" in text
    assert "non-completed" in text
    assert "solver status" in text
    assert "maxit=1000" in text
    assert "thesis Table 5.12 timings are surfaced alongside the current local timings" in text
    assert "thesis Table 5.13 timings are shown beside fresh local serial-python reruns" in text
    assert "## Convergence Diagnostics" in text
    assert "root-cause category" in text
    assert "action taken" in text
    assert "MPA accepted-step peak cycling / slow stop decay" in text
    assert "exact-direction Step 6 halving failure" in text
    assert "cheap-direction Step 6 halving failure" in text
    assert "timing comparison remains blocked" not in text
    assert "../assets/plaplace_u3_thesis/" in text
    assert '<span style="color:#1d4ed8;"><em>' in text
    assert '<span style="color:#b91c1c;"><strong>' in text
    assert "artifacts/raw_results/plaplace_u3_thesis_full/summary.json" in text


def test_plaplace_up_arctan_problem_page_contains_required_sections() -> None:
    text = (DOCS_ROOT / "problems" / "pLaplace_up_arctan.md").read_text(encoding="utf-8")
    assert "Source note used for this implementation" in text
    assert "## Mathematical Specification" in text
    assert "### p = 2 Validation Problem" in text
    assert "### p = 3 Main Problem" in text
    assert "## Solvability And Proof Notes" in text
    assert "This page therefore claims existence/solvability" in text
    assert "## Discretization And Algorithm Notes" in text
    assert ("## Raw Versus Certified Diagnostics" in text) or ("## Cross-Method Comparison" in text)
    assert "## p = 2 Validation Study" in text
    assert "## p = 3 Eigenvalue Stage" in text
    assert "## p = 3 Main Study" in text
    assert "## JAX + PETSc Backend" in text
    assert "## PETSc Timing And Scaling" in text
    assert "## Alternative Certified Branch: Shifted-Line RMPA + Newton" in text
    assert "## Seed And Endpoint Geometry Comparison" in text
    assert "nonlinear its" in text
    assert "MPA iters" in text
    assert "Newton iters" in text
    assert "linear its" in text
    assert "Chebyshev" in text
    assert "Jacobi" in text
    assert "## Cross-Method Comparison" in text
    assert "## Commands Used" in text
    assert "../assets/plaplace_up_arctan/" in text
    assert "No separate `docs/results/pLaplace_up_arctan.md` page is maintained" in text
    assert "artifacts/raw_results/plaplace_up_arctan_petsc/summary.json" in text


def test_plaplace_u3_thesis_report_contains_readable_blocks() -> None:
    text = (REPO_ROOT / "artifacts" / "reports" / "plaplace_u3_thesis" / "README.md").read_text(encoding="utf-8")
    assert "## Thesis Problem And Functionals" in text
    assert "## Assignment Snapshot" in text
    assert "**Problem spec**" in text
    assert "**Column legend**" in text
    assert "**Discrepancy notes**" in text
    assert "low impact" in text
    assert "timing note:" in text
    assert "timing status" in text
    assert "runtime context" in text
    assert "thesis t[s]" in text
    assert "repo t[s]" in text
    assert "1 proc, serial python" in text
    assert "## Stage C Timing Summary" in text
    assert "## Convergence Diagnostics" in text
    assert "Square cross-method timing table for MPA, RMPA, and OA1." in text
    assert "timing complete" in text
    assert "timing unavailable" in text
    assert "non-completed" in text
    assert "solver status" in text
    assert "maxit=1000" in text
    assert "thesis Table 5.12 timings are surfaced alongside the current local timings" in text
    assert "thesis Table 5.13 timings are shown beside fresh local serial-python reruns" in text
    assert "## Convergence Diagnostics" in text
    assert "root-cause category" in text
    assert "action taken" in text
    assert "MPA accepted-step peak cycling / slow stop decay" in text
    assert "exact-direction Step 6 halving failure" in text
    assert "cheap-direction Step 6 halving failure" in text
    assert "timing comparison remains blocked" not in text


def test_plaplace_up_arctan_report_contains_required_sections() -> None:
    text = (REPO_ROOT / "artifacts" / "reports" / "plaplace_up_arctan" / "README.md").read_text(encoding="utf-8")
    assert "# pLaplace_up_arctan Report" in text
    assert ("## Raw Versus Certified Diagnostics" in text) or ("## Summary" in text)
    assert "## p = 2 Validation Study" in text
    assert "## p = 3 Eigenvalue Stage" in text
    assert "## p = 3 Main Study" in text
    assert "## JAX + PETSc Backend" in text
    assert "## PETSc Timing And Scaling" in text
    assert "## Alternative Certified Branch: Shifted-Line RMPA + Newton" in text
    assert "## Seed And Endpoint Geometry Comparison" in text
    assert ("## Cross-Method Diagnostics" in text) or ("## Cross-Method Comparison" in text)
    assert "## Reproduction" in text
    assert "artifacts/raw_results/plaplace_up_arctan_full/summary.json" in text
    assert "artifacts/raw_results/plaplace_up_arctan_petsc/summary.json" in text


def test_results_pages_contain_required_sections() -> None:
    for name in ("pLaplace", "GinzburgLandau", "HyperElasticity", "Plasticity", "Topology"):
        text = (DOCS_ROOT / "results" / f"{name}.md").read_text(encoding="utf-8")
        assert "## Current Maintained Comparison" in text
        assert "## Reproduction Commands" in text
        assert "![" in text


def test_current_assets_exist_under_docs_assets() -> None:
    expected_pdf = [
        "plaplace/plaplace_sample_state.pdf",
        "plaplace/plaplace_energy_levels.pdf",
        "plaplace/plaplace_strong_scaling.pdf",
        "plaplace/plaplace_mesh_timing.pdf",
        "ginzburg_landau/ginzburg_landau_sample_state.pdf",
        "ginzburg_landau/ginzburg_landau_energy_levels.pdf",
        "ginzburg_landau/ginzburg_landau_strong_scaling.pdf",
        "ginzburg_landau/ginzburg_landau_mesh_timing.pdf",
        "hyperelasticity/hyperelasticity_sample_state.pdf",
        "hyperelasticity/hyperelasticity_energy_levels.pdf",
        "hyperelasticity/hyperelasticity_strong_scaling.pdf",
        "hyperelasticity/hyperelasticity_mesh_timing.pdf",
        "plasticity/mc_plasticity_p4_l5_displacement.pdf",
        "plasticity/mc_plasticity_p4_l5_deviatoric_strain_robust.pdf",
        "topology/topology_final_density.pdf",
        "topology/topology_objective_history.pdf",
        "topology/topology_strong_scaling.pdf",
        "topology/topology_mesh_timing.pdf",
        "plaplace_up_arctan/p2_solution_panel.pdf",
        "plaplace_up_arctan/p2_convergence_history.pdf",
        "plaplace_up_arctan/p3_solution_panel.pdf",
        "plaplace_up_arctan/p3_convergence_history.pdf",
        "plaplace_up_arctan/p3_eigenfunction.pdf",
        "plaplace_up_arctan/lambda_convergence.pdf",
        "plaplace_up_arctan/iteration_counts.pdf",
        "plaplace_up_arctan/reference_error_refinement.pdf",
        "plaplace_up_arctan/petsc_mesh_timing.pdf",
        "plaplace_up_arctan/petsc_strong_scaling.pdf",
    ]
    expected_png = [rel.replace(".pdf", ".png") for rel in expected_pdf]
    expected_png_only = [
        "plasticity/plasticity_p4_l7_scaling_overall_loglog.png",
        "plasticity/plasticity_p4_l7_scaling_per_linear_iteration_loglog.png",
        "plasticity/plasticity_p4_l7_setup_subparts_loglog.png",
        "plasticity/plasticity_p4_l7_callback_breakdown_loglog.png",
        "plasticity/plasticity_p4_l7_linear_breakdown_loglog.png",
        "plasticity/plasticity_p4_l7_pmg_internal_loglog.png",
        "plaplace_u3_thesis/plaplace_u3_sample_state.png",
        "plaplace_u3_thesis/square_multibranch_panel.png",
        "plaplace_u3_thesis/square_hole_panel.png",
        "plaplace_up_arctan/p2_solution_panel.png",
        "plaplace_up_arctan/p2_convergence_history.png",
        "plaplace_up_arctan/p3_solution_panel.png",
        "plaplace_up_arctan/p3_convergence_history.png",
        "plaplace_up_arctan/p3_eigenfunction.png",
        "plaplace_up_arctan/lambda_convergence.png",
        "plaplace_up_arctan/iteration_counts.png",
        "plaplace_up_arctan/reference_error_refinement.png",
        "plaplace_up_arctan/petsc_mesh_timing.png",
        "plaplace_up_arctan/petsc_strong_scaling.png",
    ]
    extra_pdf = [
        "plaplace_u3_thesis/plaplace_u3_sample_state.pdf",
    ]
    for rel in expected_pdf:
        assert (ASSETS_ROOT / rel).exists(), rel
    for rel in expected_png:
        assert (ASSETS_ROOT / rel).exists(), rel
    for rel in expected_png_only:
        assert (ASSETS_ROOT / rel).exists(), rel
    for rel in extra_pdf:
        assert (ASSETS_ROOT / rel).exists(), rel
    assert (ASSETS_ROOT / "topology" / "topology_parallel_final_evolution.gif").exists()


def test_docs_use_only_current_repo_relative_paths() -> None:
    md_paths = [DOCS_ROOT / "README.md", *sorted(DOCS_ROOT.glob("**/*.md")), REPO_ROOT / "README.md"]
    violations: list[str] = []
    for path in md_paths:
        text = path.read_text(encoding="utf-8")
        for snippet in BANNED_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not violations, "Current docs still contain banned stale paths:\n" + "\n".join(violations)


def test_current_docs_have_one_problem_and_one_results_page_per_family() -> None:
    problems = _doc_names("problems")
    results = _doc_names("results")
    assert problems == [
        "GinzburgLandau.md",
        "HyperElasticity.md",
        "Plasticity.md",
        "Plasticity3D.md",
        "Topology.md",
        "pLaplace.md",
        "pLaplace_u3_thesis_replications.md",
        "pLaplace_up_arctan.md",
    ]
    assert results == [
        "GinzburgLandau.md",
        "HyperElasticity.md",
        "Plasticity.md",
        "Plasticity3D.md",
        "Topology.md",
        "pLaplace.md",
    ]
    assert not (DOCS_ROOT / "results" / "pLaplace_up_arctan.md").exists()


def test_publication_style_constants_remain_locked() -> None:
    assert common.FIGURE_WIDTH_CM == 11.0
    assert common.OVERVIEW_FONT_PT == 12.0


def test_internal_figure_build_scripts_still_exist() -> None:
    expected = [
        BUILD_ROOT / "build_plaplace_figures.py",
        BUILD_ROOT / "build_ginzburg_landau_figures.py",
        BUILD_ROOT / "build_hyperelasticity_figures.py",
        BUILD_ROOT / "build_topology_figures.py",
        BUILD_ROOT / "build_all.py",
    ]
    for path in expected:
        assert path.exists(), path
