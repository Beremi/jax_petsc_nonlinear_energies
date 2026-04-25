#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import REPO_ROOT, TABLES_ROOT, ensure_paper_dirs, read_csv_rows, read_json, write_json, write_text


LOCAL_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/comparison_summary.json"
)
MIXED_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/comparison_summary.json"
)
SOURCEFIXED_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg/comparison_summary.json"
)
P3D_DEGREE_ENERGY_STUDY_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/plasticity3d_lambda1p55_degree_mesh_energy_study/comparison_summary.json"
)
P3D_VALIDATION_SUMMARY = REPO_ROOT / "artifacts/raw_results/plasticity3d_validation/comparison_summary.json"
P3D_DERIVATIVE_ABLATION_SUMMARY = (
    REPO_ROOT / "artifacts/raw_results/plasticity3d_derivative_ablation/comparison_summary.json"
)
JAX_FEM_BASELINE_SUMMARY = REPO_ROOT / "artifacts/raw_results/jax_fem_hyperelastic_baseline/comparison_summary.json"

PLAPLACE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/parity_showcase.csv"
GL_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/parity_showcase.csv"
HE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/parity_showcase.csv"

PLAPLACE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/strong_scaling.csv"
GL_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/strong_scaling.csv"
HE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/strong_scaling.csv"
TOPO_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/strong_scaling.csv"
TOPO_RESOLUTION = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/resolution_objectives.csv"

P2D_SHOWCASE = REPO_ROOT / "artifacts/raw_results/docs_showcase/mc_plasticity_p4_l5/output.json"
P2D_L6_SUMMARY = REPO_ROOT / "artifacts/raw_results/slope_stability_l6_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json"
P2D_L7_SUMMARY = REPO_ROOT / "artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json"
SOURCE_CONT_NP8 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix/data/run_info.json"
)
SOURCE_CONT_NP32 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix/data/run_info.json"
)
SOURCE_CONT_NP8_PROGRESS = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix/data/progress_latest.json"
)
SOURCE_CONT_NP32_PROGRESS = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix/data/progress_latest.json"
)

LOCAL_IMPL = "local_constitutiveAD_local_pmg_armijo"
SOURCE_IMPL = "source_local_pmg_armijo"
LOCAL_SOURCEFIXED_IMPL = "local_constitutiveAD_local_pmg_sourcefixed_armijo"
SOURCE_SOURCEFIXED_IMPL = "source_local_pmg_sourcefixed_armijo"


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_sci(value: float) -> str:
    return f"{float(value):.3e}"


def xcol(weight: float, align: str = "RaggedRight") -> str:
    return rf">{{\hsize={float(weight):.2f}\hsize\linewidth=\hsize\{align}\arraybackslash}}X"


def latex_table(spec: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def latex_tabularx(spec: str, header: list[str], rows: list[list[str]], *, width: str = r"\textwidth") -> str:
    lines = [
        rf"\begin{{tabularx}}{{{width}}}{{{spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines) + "\n"


def load_rows(path: Path) -> list[dict[str, object]]:
    data = read_json(path)
    rows = [dict(row) for row in data["rows"]]
    rows.sort(key=lambda row: (int(row.get("ranks", 10**6)), str(row.get("implementation", ""))))
    return rows


def find_rows(rows: list[dict[str, object]], impl: str) -> list[dict[str, object]]:
    selected = [row for row in rows if str(row.get("implementation", "")) == impl]
    selected.sort(key=lambda row: int(row["ranks"]))
    return selected


def write_table(name: str, spec: str, header: list[str], rows: list[list[str]]) -> None:
    write_text(TABLES_ROOT / name, latex_table(spec, header, rows))


def write_tablex(name: str, spec: str, header: list[str], rows: list[list[str]], *, width: str = r"\textwidth") -> None:
    write_text(TABLES_ROOT / name, latex_tabularx(spec, header, rows, width=width))


def select_csv_rows(path: Path, implementations: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [row for row in rows if row.get("implementation") in implementations]


def select_topology_rows(labels: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(TOPO_RESOLUTION)
    return [row for row in rows if row.get("label") in labels]


def plasticity2d_resolution_rows() -> list[dict[str, object]]:
    showcase = read_json(P2D_SHOWCASE)
    l5_result = showcase["result"]["steps"][0]
    rows: list[dict[str, object]] = [
        {
            "label": r"\code{P4(L5)}",
            "free_dofs": int(showcase["mesh"]["free_dofs"]),
            "energy": float(l5_result["energy"]),
            "total_time_s": float(showcase["timings"]["total_time"]),
            "status": "endpoint converged",
            "note": "curated showcase",
        }
    ]
    for path, ranks, label in (
        (P2D_L6_SUMMARY, 8, r"\code{P4(L6)}"),
        (P2D_L7_SUMMARY, 16, r"\code{P4(L7)}"),
    ):
        summary_rows = read_json(path)
        selected = next(row for row in summary_rows if int(row["ranks"]) == ranks)
        rows.append(
            {
                "label": label,
                "free_dofs": int(selected["free_dofs"]),
                "energy": float(selected["energy"]),
                "total_time_s": float(selected["total_time_sec"]),
                "status": str(selected["status"]),
                "note": f"fixed-work at {ranks} ranks",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready LaTeX tables.")
    parser.add_argument("--out-dir", type=Path, default=TABLES_ROOT)
    args = parser.parse_args()
    ensure_paper_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    local_rows = load_rows(LOCAL_P3D_SUMMARY)
    mixed_rows = load_rows(MIXED_P3D_SUMMARY)
    sourcefixed_rows = load_rows(SOURCEFIXED_P3D_SUMMARY)
    degree_energy_rows = load_rows(P3D_DEGREE_ENERGY_STUDY_SUMMARY)

    pl_rows = read_csv_rows(PLAPLACE_SCALING)
    gl_rows = read_csv_rows(GL_SCALING)
    he_rows = read_csv_rows(HE_SCALING)
    topo_rows = read_csv_rows(TOPO_SCALING)

    source8 = read_json(SOURCE_CONT_NP8)
    source32 = read_json(SOURCE_CONT_NP32)
    source8_progress = read_json(SOURCE_CONT_NP8_PROGRESS)
    source32_progress = read_json(SOURCE_CONT_NP32_PROGRESS)
    p3d_validation = read_json(P3D_VALIDATION_SUMMARY)
    p3d_ablation = read_json(P3D_DERIVATIVE_ABLATION_SUMMARY)
    jax_fem_baseline = read_json(JAX_FEM_BASELINE_SUMMARY)

    local_scaling_rows = find_rows(local_rows, LOCAL_IMPL)
    mixed_local_rows = find_rows(mixed_rows, LOCAL_IMPL)
    mixed_source_rows = find_rows(mixed_rows, SOURCE_IMPL)
    sourcefixed_local_rows = find_rows(sourcefixed_rows, LOCAL_SOURCEFIXED_IMPL)
    sourcefixed_source_rows = find_rows(sourcefixed_rows, SOURCE_SOURCEFIXED_IMPL)

    pl_showcase = select_csv_rows(PLAPLACE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    gl_showcase = select_csv_rows(GL_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    he_showcase = select_csv_rows(HE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_serial"))
    p2d_rows = plasticity2d_resolution_rows()
    topo_benchmark_rows = select_topology_rows(("serial_reference", "parallel_final"))

    write_tablex(
        "implementation_capability_matrix.tex",
        "@{}" + xcol(1.00) + xcol(0.72, "Centering") + xcol(0.74, "Centering") + xcol(0.82, "Centering") + xcol(1.72) + "@{}",
        ["Family", "FEniCS", "pure JAX", "JAX+PETSc", "Higher-order / advanced AD"],
        [
            ["$p$-Laplace", "yes", "yes", "yes", "element AD and local colored SFD"],
            ["Ginzburg--Landau", "yes", "no", "yes", "element AD and local colored SFD"],
            ["Hyperelasticity", "yes", "yes", "yes", "trust-region element AD"],
            ["Plasticity2D", "no", "no", "yes", "repository scalarized potential and same-mesh PMG"],
            ["Plasticity3D", "no", "no", "yes", "constitutive AD, element AD, and same-mesh PMG"],
            ["Topology optimization", "no", "yes", "yes", "distributed design updates and PETSc mechanics"],
        ],
    )

    write_tablex(
        "benchmark_specification_matrix.tex",
        "@{}" + xcol(0.88) + xcol(0.72) + xcol(0.88) + xcol(1.08) + xcol(1.60) + "@{}",
        ["Family", "Grid / mesh", "Stop rule", "Compared paths", "Main difficulty"],
        [
            ["$p$-Laplace", "L9", "Newton + line search", "FEniCS, pure JAX, JAX+PETSc", "nonlinear elliptic solve with exact sparse Hessians"],
            ["Ginzburg--Landau", "L9", "Newton + line search", "FEniCS, JAX+PETSc", "indefinite local curvature from the double well"],
            ["Hyperelasticity", "L4, 24 steps", "trust-region path", "FEniCS, pure JAX, JAX+PETSc", "nonconvex large-deformation mechanics"],
            ["Plasticity2D", "$P4(L5$--$L7)$", "endpoint or fixed work", "JAX+PETSc only", "same-mesh PMG and nonlinear tail behavior"],
            ["Plasticity3D", "$P1/P2/P4$", "$\\|g\\|<10^{-2}$", "local, source, PMG variants", "heterogeneous 3D Mohr--Coulomb with constitutive AD"],
            ["Topology", "$768\\times384$", "stall-stop continuation", "pure JAX, JAX+PETSc", "distributed design-mechanics coupling"],
        ],
    )

    write_tablex(
        "reference_availability.tex",
        "@{}"
        + xcol(1.08)
        + "@{\\hspace{0.8em}}"
        + xcol(0.58, "Centering")
        + "@{\\hspace{0.7em}}"
        + xcol(0.62, "Centering")
        + "@{\\hspace{1.0em}}"
        + xcol(1.72)
        + "@{}",
        ["Family", "FEniCS", "pure JAX", "Notes"],
        [
            ["$p$-Laplace", "yes", "yes", "All three stacks exist on maintained showcase cases."],
            ["Ginzburg--Landau", "yes", "no", "FEniCS and JAX+PETSc form the maintained comparison."],
            ["Hyperelasticity", "yes", "yes", "pure JAX is a serial formulation reference only."],
            ["Plasticity2D", "no", "no", "The maintained story is JAX+PETSc only."],
            ["Plasticity3D", "no", "no", "Source assembly exists as supporting comparison, not as a maintained reference path."],
            ["Topology", "no", "yes", "Parallel fine-grid path is JAX+PETSc; pure JAX remains the serial design reference."],
        ],
    )

    write_tablex(
        "sota_framework_comparison.tex",
        "@{}" + xcol(0.92) + xcol(0.86) + xcol(0.92) + xcol(0.94) + xcol(0.84) + xcol(1.02) + "@{}",
        ["Family", "Modeling", "Differentiation", "2nd-order route", "Parallel", "Closest overlap"],
        [
            [
                "\\shortstack[l]{FEniCS\\\\DOLFINx\\\\\\citep{logg2012fenicsbook,baratta2025dolfinx}}",
                "High-level variational forms",
                "Problem-specific; AD is not the central claim",
                "Manual or application-specific",
                "Distributed FEM assembly and solve",
                "Elliptic and finite-strain mechanics",
            ],
            [
                "\\shortstack[l]{dolfin-adjoint\\\\pyadjoint\\\\cashocs\\\\\\citep{farrell2013dolfinadjoint,mitusch2019pyadjoint,blauth2023cashocsv2}}",
                "High-level PDE plus optimization loop",
                "Adjoint-based first-order sensitivities",
                "Not the advertised main path",
                "MPI via the host FEM stack",
                "PDE control, shape, and topology",
            ],
            [
                "\\shortstack[l]{JAX-FEM\\\\AutoPDEx\\\\\\citep{xue2023jaxfem,bode2025autopdex}}",
                "JAX-native PDE / FE programs",
                "Program-level forward and reverse AD",
                "Higher-order JAX derivatives",
                "JAX execution; PETSc optional in AutoPDEx",
                "Nonlinear mechanics and inverse design",
            ],
            [
                "\\shortstack[l]{Firedrake--JAX\\\\FEniCSx ext. ops\\\\\\citep{yashchuk2023bringing,latyshev2025externaloperators}}",
                "Host FEM stack plus AD bridge",
                "Tangent, adjoint, or local constitutive AD",
                "Local-operator derivatives, not full sparse Hessians",
                "Host framework parallel back end",
                "Parameterized PDEs and constitutive models",
            ],
            [
                "\\shortstack[l]{JAX-CPFEM\\\\\\citep{hu2025jaxcpfem}}",
                "JAX-native crystal-plasticity FEM",
                "Differentiable constitutive simulator",
                "AD constitutive derivatives",
                "GPU-oriented execution",
                "Crystal plasticity",
            ],
            [
                "\\shortstack[l]{FEniTop\\\\\\citep{jia2024fenitop}}",
                "FEniCSx topology code",
                "Sensitivity-based design updates",
                "Not a second-order solver paper",
                "Parallel FEniCSx workflow",
                "2D and 3D topology optimization",
            ],
            [
                "\\shortstack[l]{This\\\\repo}",
                "JAX local energies plus PETSc sparse solvers",
                "Element AD, constitutive AD, and colored SFD",
                "Local Hessians/tangents or sparse recovery",
                "PETSc MPI vectors, matrices, SNES, and multigrid",
                "$p$-Laplace, GL, hyperelasticity, plasticity, topology",
            ],
        ],
    )

    write_tablex(
        "plaplace_benchmark_summary.tex",
        "@{}" + xcol(1.10) + "r r r r@{}",
        ["Path", "Energy", "Newton", "Linear", "Wall [s]"],
        [
            [
                row["implementation"].replace("_", r"\_"),
                fmt_float(float(row["final_energy"]), 9),
                row["newton_iters"],
                row["linear_iters"],
                fmt_float(float(row["wall_time_s"]), 4),
            ]
            for row in pl_showcase
        ],
    )

    write_tablex(
        "ginzburg_landau_benchmark_summary.tex",
        "@{}" + xcol(1.10) + "r r r r@{}",
        ["Path", "Energy", "Newton", "Linear", "Wall [s]"],
        [
            [
                row["implementation"].replace("_", r"\_"),
                fmt_float(float(row["final_energy"]), 10),
                row["newton_iters"],
                row["linear_iters"],
                fmt_float(float(row["wall_time_s"]), 4),
            ]
            for row in gl_showcase
        ],
    )

    write_tablex(
        "hyperelasticity_benchmark_summary.tex",
        "@{}" + xcol(1.10) + "r r r r@{}",
        ["Path", "Energy", "Steps", "Linear", "Wall [s]"],
        [
            [
                row["implementation"].replace("_", r"\_"),
                fmt_float(float(row["final_energy"]), 6),
                row["completed_steps"],
                row["total_linear_iters"],
                fmt_float(float(row["wall_time_s"]), 3),
            ]
            for row in he_showcase
        ],
    )

    write_tablex(
        "plasticity2d_benchmark_summary.tex",
        "@{}" + xcol(0.88) + "r r r " + xcol(1.32) + "@{}",
        ["Case", "Free DOFs", "Energy", "Wall [s]", "Note"],
        [
            [
                str(row["label"]),
                str(int(row["free_dofs"])),
                fmt_float(float(row["energy"]), 9),
                fmt_float(float(row["total_time_s"]), 3),
                str(row["note"]),
            ]
            for row in p2d_rows
        ],
    )

    write_tablex(
        "plasticity3d_benchmark_summary.tex",
        "@{}" + xcol(0.52) + xcol(0.62) + "r r r r@{}",
        ["Degree", "Mesh", "Free DOFs", "Energy", "$\\|g\\|_{\\mathrm{final}}$", "Wall [s]"],
        [
            [
                str(row["degree_line"]),
                str(row["mesh_alias"]).replace("_", r"\_"),
                str(int(row["free_dofs"])),
                fmt_float(float(row["energy"]), 6),
                fmt_sci(float(row["final_grad_norm"])),
                fmt_float(float(row["total_time_s"]), 3),
            ]
            for row in sorted(
                degree_energy_rows,
                key=lambda row: (int(str(row["degree_line"]).replace("P", "")), int(row["free_dofs"])),
            )
        ],
    )

    write_tablex(
        "topology_benchmark_summary.tex",
        "@{}" + xcol(0.92) + "r r r r r@{}",
        ["Case", "Ranks", "Outer", "Compliance", "Volume", "Wall [s]"],
        [
            [
                row["mesh"].replace("x", r"$\times$"),
                row["ranks"],
                row["outer_iterations"],
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
                fmt_float(float(row["wall_time_s"]), 3),
            ]
            for row in topo_benchmark_rows
        ],
    )

    write_table(
        "plasticity3d_recommended_scaling.tex",
        "rrrrrrr",
        ["Ranks", "Wall [s]", "Solve [s]", "Speedup", "Eff.", "Newton", "Linear"],
        [
            [
                str(int(row["ranks"])),
                fmt_float(float(row["wall_time_s"])),
                fmt_float(float(row["solve_time_s"])),
                fmt_float(float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])),
                fmt_float((float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])) / float(row["ranks"])),
                str(int(row["nit"])),
                str(int(row["linear_iterations_total"])),
            ]
            for row in local_scaling_rows
        ],
    )

    write_tablex(
        "plasticity3d_local_vs_source.tex",
        "@{}r r r r r r@{}",
        ["Ranks", "Local wall [s]", "Source wall [s]", "Local solve [s]", "Source solve [s]", "Ratio"],
        [
            [
                str(int(lrow["ranks"])),
                fmt_float(float(lrow["wall_time_s"])),
                fmt_float(float(srow["wall_time_s"])),
                fmt_float(float(lrow["solve_time_s"])),
                fmt_float(float(srow["solve_time_s"])),
                fmt_float(float(lrow["wall_time_s"]) / float(srow["wall_time_s"])),
            ]
            for lrow, srow in zip(mixed_local_rows, mixed_source_rows, strict=True)
        ],
    )

    write_tablex(
        "plasticity3d_sourcefixed_alternative.tex",
        "@{}r " + xcol(1.10) + xcol(1.10) + "@{}",
        ["Ranks", "Local + sourcefixed", "Source + sourcefixed"],
        [
            [
                str(rank),
                next(
                    (
                        f"{fmt_float(float(row['wall_time_s']), 1)} s; {int(row['nit'])} N; {int(row['linear_iterations_total'])} L; {fmt_sci(float(row['final_metric']))}"
                        for row in sourcefixed_local_rows
                        if int(row["ranks"]) == rank and str(row["status"]) == "completed"
                    ),
                    "--",
                ),
                next(
                    (
                        f"{fmt_float(float(row['wall_time_s']), 1)} s; {int(row['nit'])} N; {int(row['linear_iterations_total'])} L; {fmt_sci(float(row['final_metric']))}"
                        for row in sourcefixed_source_rows
                        if int(row["ranks"]) == rank and str(row["status"]) == "completed"
                    ),
                    "--",
                ),
            ]
            for rank in [4, 8, 16, 32]
        ],
    )

    write_tablex(
        "plasticity3d_degree_energy_study.tex",
        "@{}" + xcol(0.72) + xcol(0.72) + "r r r " + xcol(0.74) + "@{}",
        ["Degree", "Mesh", "Free DOFs", "Energy", "Total [s]", "Status"],
        [
            [
                str(row["degree_line"]),
                str(row["mesh_alias"]).replace("_", r"\_"),
                str(int(row["free_dofs"])),
                fmt_float(float(row["energy"]), 6),
                fmt_float(float(row["total_time_s"])),
                "reused" if bool(row.get("reused", False)) else str(row["status"]),
            ]
            for row in degree_energy_rows
        ],
    )

    topo_best = [row for row in topo_rows if row["result"] == "completed"]
    topo_best.sort(key=lambda row: int(row["ranks"]))
    write_table(
        "topology_summary.tex",
        "rrrrrrr",
        ["Ranks", "Wall [s]", "Solve [s]", "Outer", "$p$", "Compliance", "Volume"],
        [
            [
                row["ranks"],
                fmt_float(float(row["wall_time_s"])),
                fmt_float(float(row["solve_time_s"])),
                row["outer_iterations"],
                fmt_float(float(row["final_p_penal"]), 2),
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
            ]
            for row in topo_best
        ],
    )

    write_tablex(
        "source_continuation_compare.tex",
        "@{}" + xcol(1.45) + "r r r r@{}",
        ["Case", "Runtime [s]", "Init linear", "Continuation linear", "Final $\\lambda$"],
        [
            [
                "source PMG-shell fixed, 8 ranks",
                fmt_float(float(source8["run_info"]["runtime_seconds"])),
                str(int(source8["timings"]["linear"]["init_linear_iterations"])),
                str(int(source8["timings"]["linear"]["attempt_linear_iterations_total"])),
                fmt_float(float(source8_progress["lambda_last"]), 6),
            ],
            [
                "source PMG-shell fixed, 32 ranks",
                fmt_float(float(source32["run_info"]["runtime_seconds"])),
                str(int(source32["timings"]["linear"]["init_linear_iterations"])),
                str(int(source32["timings"]["linear"]["attempt_linear_iterations_total"])),
                fmt_float(float(source32_progress["lambda_last"]), 6),
            ],
        ],
    )

    layer1a_metrics = p3d_validation["layer1a"]["final_metrics"]
    layer2_metrics = p3d_validation["layer2"]
    write_tablex(
        "plasticity3d_validation_summary.tex",
        "@{}" + xcol(1.10) + xcol(1.45) + "r@{}",
        ["Layer", "Metric", "Value"],
        [
            ["1A", "work relative difference", fmt_sci(float(layer1a_metrics["work_relative_difference"]))],
            ["1A", "displacement relative L2", fmt_sci(float(layer1a_metrics["displacement_relative_l2"]))],
            ["1A", "deviatoric-strain relative L2", fmt_sci(float(layer1a_metrics["deviatoric_strain_relative_l2"]))],
            [
                "2",
                "highest-successful-$\\lambda$ relative difference",
                fmt_sci(float(layer2_metrics["critical_lambda_schedule_proxy"]["relative_difference"])),
            ],
            ["2", "$u_{\\max}(\\lambda)$ relative L2", fmt_sci(float(layer2_metrics["umax_curve_relative_l2"]))],
            [
                "2",
                "endpoint displacement relative L2",
                fmt_sci(float(layer2_metrics["endpoint_displacement_relative_l2"])),
            ],
            [
                "2",
                "boundary-profile relative L2",
                fmt_sci(float(layer2_metrics["boundary_profile_relative_l2"])),
            ],
            [
                "2",
                "acceptance",
                "pass" if bool(layer2_metrics["acceptance"]["overall_pass"]) else "fail",
            ],
        ],
    )

    ablation_rows = [dict(row) for row in p3d_ablation["rows"]]
    write_tablex(
        "plasticity3d_derivative_ablation.tex",
        "@{}" + xcol(0.88) + "r r r r r r r@{}",
        ["Route", "Wall [s]", "Solve [s]", "Newton", "Linear", "Energy", "$\\omega$", "$u_{\\max}$"],
        [
            [
                str(row["display_label"]),
                fmt_float(float(row["median_wall_time_s"]), 3),
                fmt_float(float(row["median_solve_time_s"]), 3),
                fmt_float(float(row["median_nit"]), 2),
                fmt_float(float(row["median_linear_iterations_total"]), 1),
                fmt_float(float(row["median_energy"]), 6),
                fmt_float(float(row["median_omega"]), 6),
                fmt_float(float(row["median_u_max"]), 6),
            ]
            for row in ablation_rows
        ],
    )

    fairness = dict(jax_fem_baseline["fairness_gate"])
    final_metrics = dict(jax_fem_baseline["final_metrics"])
    timing = dict(jax_fem_baseline["timing_medians_s"])
    write_tablex(
        "jax_fem_hyperelastic_baseline.tex",
        "@{}" + xcol(1.25) + xcol(1.55) + "@{}",
        ["Metric", "Value"],
        [
            ["matched problem contract", "yes" if bool(fairness["checks"]["same_mesh_path"] and fairness["checks"]["same_schedule"] and fairness["checks"]["same_constitutive_law"]) else "no"],
            ["final energy relative difference", fmt_sci(float(final_metrics["energy_rel_diff"]))],
            ["full-field displacement relative L2", fmt_sci(float(final_metrics["field_relative_l2"]))],
            ["centerline relative L2", fmt_sci(float(final_metrics["centerline_relative_l2"]))],
            ["$u_{\\max}$ curve relative L2", fmt_sci(float(final_metrics["umax_curve_relative_l2"]))],
            ["repo median wall time [s]", fmt_float(float(timing["repo_serial_direct"]), 3)],
            ["JAX-FEM median wall time [s]", fmt_float(float(timing["jax_fem_umfpack_serial"]), 3)],
            ["fairness gate", "pass" if bool(fairness["passed"]) else "artifact-only"],
        ],
    )

    payload = {
        "plasticity3d_recommended_scaling_rows": [
            {
                "ranks": int(row["ranks"]),
                "wall_time_s": float(row["wall_time_s"]),
                "solve_time_s": float(row["solve_time_s"]),
                "nit": int(row["nit"]),
                "linear_iterations_total": int(row["linear_iterations_total"]),
                "final_metric": float(row["final_metric"]),
            }
            for row in local_scaling_rows
        ],
        "plasticity3d_local_vs_source_rows": [
            {
                "ranks": int(lrow["ranks"]),
                "local_wall_time_s": float(lrow["wall_time_s"]),
                "source_wall_time_s": float(srow["wall_time_s"]),
                "local_solve_time_s": float(lrow["solve_time_s"]),
                "source_solve_time_s": float(srow["solve_time_s"]),
            }
            for lrow, srow in zip(mixed_local_rows, mixed_source_rows, strict=True)
        ],
        "plasticity3d_validation": {
            "layer1a_work_rel": float(layer1a_metrics["work_relative_difference"]),
            "layer2_acceptance": bool(layer2_metrics["acceptance"]["overall_pass"]),
        },
        "jax_fem_hyperelastic_baseline": {
            "fairness_gate_passed": bool(fairness["passed"]),
            "energy_rel_diff": float(final_metrics["energy_rel_diff"]),
        },
    }
    write_json(REPO_ROOT / "paper/build/tables_summary.json", payload)


if __name__ == "__main__":
    main()
