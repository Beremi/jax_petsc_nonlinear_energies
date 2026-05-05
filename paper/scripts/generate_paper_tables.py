#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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
HE_KAROLINA_PMG_SCALING = (
    REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/karolina_l5_pmg_scaling.csv"
)
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

IMPLEMENTATION_LABELS = {
    "fenics_custom": "FEniCS custom Newton",
    "jax_petsc_element": "JAX+PETSc element AD",
    "jax_petsc_local_sfd": "JAX+PETSc colored SFD",
    "jax_serial": "serial JAX",
    LOCAL_IMPL: "constitutive-AD PMG solver",
    SOURCE_IMPL: "source-operator PMG variant",
    LOCAL_SOURCEFIXED_IMPL: "constitutive-AD PMG solver",
    SOURCE_SOURCEFIXED_IMPL: "source-assembly PMG variant",
}

MESH_ALIAS_MATH = {
    "L1": "L_{1}",
    "L1_2": "L_{2}",
    "L1_2_3": "L_{3}",
    "L1_2_3_4": "L_{4}",
}


def _trim_decimal(text: str) -> str:
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def num(text: str) -> str:
    return rf"\num{{{text}}}"


def fmt_float(value: float, digits: int = 3) -> str:
    return num(_trim_decimal(f"{float(value):.{digits}f}"))


def fmt_int(value: object) -> str:
    return str(int(float(value)))


def fmt_count(value: object) -> str:
    return fmt_int(value)


def fmt_dofs(value: object) -> str:
    return num(str(int(float(value))))


def fmt_wall_time(value: float) -> str:
    value = float(value)
    if abs(value) >= 100:
        return fmt_float(value, 0)
    if abs(value) >= 10:
        return fmt_float(value, 1)
    if abs(value) >= 1:
        return fmt_float(value, 2)
    if abs(value) >= 0.1:
        return fmt_float(value, 3)
    return fmt_float(value, 4)


def fmt_energy(value: float, *, precision: int | None = None) -> str:
    value = float(value)
    if precision is not None:
        return fmt_float(value, precision)
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return fmt_float(value, 0)
    if magnitude >= 1_000:
        return fmt_float(value, 1)
    if magnitude >= 100:
        return fmt_float(value, 3)
    if magnitude >= 1:
        return fmt_float(value, 6)
    return fmt_float(value, 10)


def fmt_sig(value: float, sig: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    digits = max(sig - 1 - int(math.floor(math.log10(abs(value)))), 0)
    return fmt_float(value, digits)


def fmt_sci(value: float, sig: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return num("0")
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    digits = max(sig - 1, 0)
    return rf"$\num{{{_trim_decimal(f'{mantissa:.{digits}f}')}}}\times 10^{{{exponent}}}$"


def implementation_label(name: object) -> str:
    key = str(name)
    if key in IMPLEMENTATION_LABELS:
        return IMPLEMENTATION_LABELS[key]
    if "local_constitutiveAD" in key and "local_pmg" in key:
        return "constitutive-AD PMG solver"
    if "sourcefixed" in key:
        return "source-assembly PMG variant"
    if key.startswith("source") or "_source" in key:
        return "source-operator PMG variant"
    return key.replace("_", r"\_")


def mesh_label(alias: object) -> str:
    key = str(alias)
    if key in MESH_ALIAS_MATH:
        return rf"${MESH_ALIAS_MATH[key]}$"
    if key.startswith("L") and key[1:].isdigit():
        return rf"$L_{{{key[1:]}}}$"
    return key.replace("_", r"\_")


def _math_mesh(alias: object) -> str:
    key = str(alias)
    if key in MESH_ALIAS_MATH:
        return MESH_ALIAS_MATH[key]
    if key.startswith("L") and key[1:].isdigit():
        return f"L_{{{key[1:]}}}"
    return key.replace("_", r"\_")


def degree_label(degree: object) -> str:
    key = str(degree)
    if key.startswith("P") and key[1:].isdigit():
        return rf"$P_{{{key[1:]}}}$"
    return key.replace("_", r"\_")


def element_label(degree: object, mesh_alias: object) -> str:
    key = str(degree)
    if key.startswith("P") and key[1:].isdigit():
        return rf"$P_{{{key[1:]}}}({_math_mesh(mesh_alias)})$"
    return f"{degree_label(degree)} {mesh_label(mesh_alias)}"


def find_csv_row(rows: list[dict[str, str]], solver: str, ranks: int) -> dict[str, str]:
    return next(row for row in rows if row.get("solver") == solver and int(row["nprocs"]) == ranks)


def xcol(weight: float, align: str = "RaggedRight") -> str:
    return rf">{{\hsize={float(weight):.3f}\hsize\linewidth=\hsize\{align}\arraybackslash}}X"


def xspec(*columns: tuple[float, str]) -> str:
    """Return normalized tabularx X columns.

    The hsize weights must sum to the number of X columns; otherwise tabularx
    can stretch the table poorly and emit alignment warnings.
    """
    total = sum(weight for weight, _align in columns)
    scale = len(columns) / total
    return "".join(xcol(weight * scale, align) for weight, align in columns)


def fill_spec(columns: str) -> str:
    """Return a tabular* spec with compact outer edges and stretched interiors."""
    parts = columns.split()
    if not parts:
        raise ValueError("tabular* spec needs at least one column")
    rest = " ".join(parts[1:])
    return "@{}" + parts[0] + r"@{\extracolsep{\fill}}" + (f" {rest}" if rest else "") + "@{}"


def pcol(width: str, align: str = "RaggedRight") -> str:
    return rf">{{\{align}\arraybackslash}}p{{{width}}}"


LatexRow = list[str] | str


def _latex_lines(header: list[str], rows: list[LatexRow]) -> list[str]:
    lines = [r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    return lines


def latex_table(spec: str, header: list[str], rows: list[LatexRow]) -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", *_latex_lines(header, rows), r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_tabularstar(spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> str:
    lines = [
        rf"\begin{{tabular*}}{{{width}}}{{{spec}}}",
        *_latex_lines(header, rows),
        r"\end{tabular*}",
    ]
    return "\n".join(lines) + "\n"


def latex_tabularx(spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> str:
    lines = [
        rf"\begin{{tabularx}}{{{width}}}{{{spec}}}",
        *_latex_lines(header, rows),
        r"\end{tabularx}",
    ]
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


def write_table(name: str, spec: str, header: list[str], rows: list[LatexRow]) -> None:
    write_text(TABLES_ROOT / name, latex_table(spec, header, rows))


def write_table_star(
    name: str, spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth"
) -> None:
    write_text(TABLES_ROOT / name, latex_tabularstar(spec, header, rows, width=width))


def write_tablex(name: str, spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> None:
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
            "label": element_label("P4", "L5"),
            "free_dofs": int(showcase["mesh"]["free_dofs"]),
            "energy": float(l5_result["energy"]),
            "total_time_s": float(showcase["timings"]["total_time"]),
            "status": "endpoint converged",
            "note": "curated showcase",
        }
    ]
    for path, ranks, label in (
        (P2D_L6_SUMMARY, 8, element_label("P4", "L6")),
        (P2D_L7_SUMMARY, 16, element_label("P4", "L7")),
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


def pass_fail(value: object) -> str:
    return "pass" if bool(value) else "fail"


def _layer2_gate_status(layer2_metrics: dict[str, object], key: str) -> str:
    acceptance = dict(layer2_metrics.get("acceptance", {}))
    if key not in acceptance:
        return "--"
    return pass_fail(acceptance[key])


def final_metric_header(rows: list[dict[str, object]]) -> str:
    names = {str(row.get("final_metric_name", "")).strip() for row in rows if row.get("final_metric_name")}
    if names == {"grad_norm"}:
        return "Final gradient norm"
    if names == {"relative_correction"}:
        return "Final relative correction"
    if len(names) == 1:
        return "Final " + next(iter(names)).replace("_", " ")
    return "Final convergence metric"


def sourcefixed_long_rows(
    local_rows: list[dict[str, object]], source_rows: list[dict[str, object]]
) -> list[list[str]]:
    rows_by_key = {
        (int(row["ranks"]), str(row["implementation"])): row
        for row in [*local_rows, *source_rows]
    }
    table_rows: list[list[str]] = []
    for rank in (4, 8, 16, 32):
        for implementation in (LOCAL_SOURCEFIXED_IMPL, SOURCE_SOURCEFIXED_IMPL):
            row = rows_by_key.get((rank, implementation))
            if row is None:
                table_rows.append([fmt_count(rank), implementation_label(implementation), "--", "--", "--", "--", "not run"])
                continue
            table_rows.append(
                [
                    fmt_count(rank),
                    implementation_label(implementation),
                    fmt_wall_time(float(row["wall_time_s"])),
                    fmt_count(row["nit"]),
                    fmt_count(row["linear_iterations_total"]),
                    fmt_sci(float(row["final_metric"])),
                    str(row["status"]),
                ]
            )
    return table_rows


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
    he_karolina_rows = [
        row for row in read_csv_rows(HE_KAROLINA_PMG_SCALING) if row.get("result", "completed") == "completed"
    ]
    he_karolina_rows.sort(key=lambda row: int(row["ranks"]))
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

    pl_highlight = find_csv_row(pl_rows, "jax_petsc_element", 32)
    gl_highlight = find_csv_row(gl_rows, "jax_petsc_element", 32)
    he_highlight = find_csv_row(he_rows, "jax_petsc_element", 32)
    p2d_highlight = p2d_rows[-1]
    p3d_highlight = next(row for row in local_scaling_rows if int(row["ranks"]) == 32)
    topo_highlight = next(row for row in topo_rows if row["solver"] == "jax_parallel" and int(row["ranks"]) == 32)

    write_tablex(
        "family_highlights.tex",
        "@{}"
        + xspec((0.68, "RaggedRight"), (1.55, "RaggedRight"), (2.10, "RaggedRight"))
        + "@{}",
        ["Family", "Representative maintained result", "Highlight"],
        [
            [
                "$p$-Laplace",
                f"JAX+PETSc element AD, {mesh_label('L9')}, 32 ranks: {fmt_wall_time(float(pl_highlight['total_time_s']))} s",
                "Exact element Hessians are competitive with FEniCS and faster than colored SFD on the hardest maintained case.",
            ],
            [
                "Ginzburg--Landau",
                f"JAX+PETSc element AD, {mesh_label('L9')}, 32 ranks: {fmt_wall_time(float(gl_highlight['total_time_s']))} s",
                "Element AD remains effectively tied with FEniCS custom Newton on the fine-grid maintained benchmark.",
            ],
            [
                "Hyperelasticity",
                f"JAX+PETSc element AD, {mesh_label('L4')}, 32 ranks: {fmt_wall_time(float(he_highlight['total_time_s']))} s",
                "Hybrid trust-region/line-search globalization sustains large-deformation solves in distributed mode.",
            ],
            [
                "Plasticity (2D)",
                f"JAX+PETSc deep-tail PMG, {p2d_highlight['label']}, 16 ranks: {fmt_wall_time(float(p2d_highlight['total_time_s']))} s",
                "The dominant bottlenecks shift from the coarse end to the top smoother and repeated Krylov work.",
            ],
            [
                "Plasticity3D",
                f"constitutive-AD PMG solver, {element_label('P4', 'L1_2')}, $\\lambda_{{\\mathrm{{sr}}}}=\\num{{1.0}}$, 32 ranks: {fmt_wall_time(float(p3d_highlight['wall_time_s']))} s",
                "Historical timing context for the promoted load factor; the main glued-bottom discretization study uses $\\lambda_{\\mathrm{sr}}=\\num{1.55}$.",
            ],
            [
                "Topology",
                f"parallel JAX+PETSc, $768\\times384$, 32 ranks: {fmt_wall_time(float(topo_highlight['wall_time_s']))} s",
                "Distributed design updates and PETSc mechanics deliver a stable fine-grid workflow while pure JAX remains the serial formulation reference.",
            ],
        ],
    )

    write_table(
        "implementation_capability_matrix.tex",
        "@{}l c c c " + pcol(r"0.38\textwidth") + "@{}",
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

    write_table(
        "benchmark_specification_matrix.tex",
        "@{}"
        + "l c "
        + pcol(r"0.14\textwidth")
        + " "
        + pcol(r"0.23\textwidth")
        + " "
        + pcol(r"0.26\textwidth")
        + "@{}",
        ["Family", "Grid / mesh", "Stop rule", "Compared paths", "Main difficulty"],
        [
            ["$p$-Laplace", mesh_label("L9"), "Newton + line search", "FEniCS, pure JAX, JAX+PETSc", "nonlinear elliptic solve with exact sparse Hessians"],
            ["Ginzburg--Landau", mesh_label("L9"), "Newton + line search", "FEniCS, JAX+PETSc", "indefinite local curvature from the double well"],
            ["Hyperelasticity", f"{mesh_label('L4')}, 24 steps", "trust-region path", "FEniCS, pure JAX, JAX+PETSc", "nonconvex large-deformation mechanics"],
            ["Plasticity2D", f"{element_label('P4', 'L5')}--{element_label('P4', 'L7')}", "endpoint or fixed work", "JAX+PETSc only", "same-mesh PMG and nonlinear tail behavior"],
            ["Plasticity3D", f"{degree_label('P1')}/{degree_label('P2')}/{degree_label('P4')}", "$\\|g\\| < \\num{0.01}$", "constitutive-AD and source PMG variants", "heterogeneous 3D Mohr--Coulomb with constitutive AD"],
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

    write_table_star(
        "sota_framework_comparison.tex",
        fill_spec(
            " ".join(
                [
                    pcol(r"0.16\textwidth"),
                    pcol(r"0.13\textwidth"),
                    pcol(r"0.15\textwidth"),
                    pcol(r"0.15\textwidth"),
                    pcol(r"0.14\textwidth"),
                    pcol(r"0.18\textwidth"),
                ]
            )
        ),
        ["Family", "Modeling", "Differentiation", "Second-order route", "Parallel", "Closest overlap"],
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
                "\\shortstack[l]{JAX-FEM\\\\Xue 2026\\\\AutoPDEx\\\\\\citep{xue2023jaxfem,xue2026implicit,bode2025autopdex}}",
                "JAX-native PDE / FE programs",
                "Program-level forward and reverse AD",
                "Higher-order JAX and implicit Hessian-vector derivatives",
                "JAX execution; PETSc optional in AutoPDEx",
                "Nonlinear mechanics, inverse design, and FE differentiable physics",
            ],
            [
                "\\shortstack[l]{JetSCI\\\\\\citep{cattaneo2026jetsci}}",
                "JAX local discretizations plus PETSc sparse solves",
                "JAX-differentiated discretization kernels",
                "Differentiable simulation kernels with PETSc solves",
                "JAX/GPU within node; PETSc MPI across nodes",
                "Heterogeneous micromechanics",
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
                "This work",
                "JAX local energies plus PETSc sparse solvers",
                "Element AD, constitutive AD, and colored SFD",
                "Local Hessians/tangents or sparse recovery",
                "PETSc MPI vectors, matrices, SNES, and multigrid",
                "$p$-Laplace, GL, hyperelasticity, plasticity, topology",
            ],
        ],
    )

    write_table_star(
        "plaplace_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=9),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in pl_showcase
        ],
    )

    write_table_star(
        "ginzburg_landau_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=10),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in gl_showcase
        ],
    )

    write_table_star(
        "hyperelasticity_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Steps", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=6),
                fmt_count(row["completed_steps"]),
                fmt_count(row["total_linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in he_showcase
        ],
    )

    write_table_star(
        "hyperelasticity_karolina_pmg_scaling.tex",
        fill_spec("c c c c c c c c"),
        [
            "Ranks",
            "Nodes",
            "Coarse groups",
            "Ranks/group",
            "Solver total [s]",
            "First step [s]",
            "Newton iters",
            "Krylov iters",
        ],
        [
            [
                fmt_count(row["ranks"]),
                fmt_count(row["nodes"]),
                fmt_count(row["coarse_groups"]),
                fmt_count(row["coarse_group_ranks"]),
                fmt_wall_time(float(row["solver_total_s"])),
                fmt_wall_time(float(row["first_step_s"])),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
            ]
            for row in he_karolina_rows
        ],
    )

    write_table(
        "plasticity2d_benchmark_summary.tex",
        "@{}l c c c l@{}",
        ["Case", "Free DOFs", "Energy", "Wall time [s]", "Note"],
        [
            [
                str(row["label"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_wall_time(float(row["total_time_s"])),
                str(row["note"]),
            ]
            for row in p2d_rows
        ],
    )

    write_table_star(
        "plasticity3d_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Element", "Free DOFs", "Energy", "$\\|g\\|_{\\mathrm{final}}$", "Wall time [s]"],
        [
            [
                element_label(row["degree_line"], row["mesh_alias"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_sci(float(row["final_grad_norm"])),
                fmt_wall_time(float(row["total_time_s"])),
            ]
            for row in sorted(
                degree_energy_rows,
                key=lambda row: (int(str(row["degree_line"]).replace("P", "")), int(row["free_dofs"])),
            )
        ],
    )

    write_table_star(
        "topology_benchmark_summary.tex",
        fill_spec("l c c c c c"),
        ["Case", "Ranks", "Outer iters", "Compliance", "Volume fraction", "Wall time [s]"],
        [
            [
                row["mesh"].replace("x", r"$\times$"),
                fmt_count(row["ranks"]),
                fmt_count(row["outer_iterations"]),
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in topo_benchmark_rows
        ],
    )

    write_table_star(
        "plasticity3d_recommended_scaling.tex",
        fill_spec("c c c c c c c"),
        ["Ranks", "Wall time [s]", "Solve time [s]", "Speedup", "Efficiency", "Newton iters", "Krylov iters"],
        [
            [
                fmt_count(row["ranks"]),
                fmt_wall_time(float(row["wall_time_s"])),
                fmt_wall_time(float(row["solve_time_s"])),
                fmt_float(float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])),
                fmt_float((float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])) / float(row["ranks"])),
                fmt_count(row["nit"]),
                fmt_count(row["linear_iterations_total"]),
            ]
            for row in local_scaling_rows
        ],
    )

    write_table_star(
        "plasticity3d_local_vs_source.tex",
        fill_spec("c c c c c c"),
        ["Ranks", "Constitutive wall [s]", "Source wall [s]", "Constitutive solve [s]", "Source solve [s]", "Ratio"],
        [
            [
                fmt_count(lrow["ranks"]),
                fmt_wall_time(float(lrow["wall_time_s"])),
                fmt_wall_time(float(srow["wall_time_s"])),
                fmt_wall_time(float(lrow["solve_time_s"])),
                fmt_wall_time(float(srow["solve_time_s"])),
                fmt_float(float(lrow["wall_time_s"]) / float(srow["wall_time_s"])),
            ]
            for lrow, srow in zip(mixed_local_rows, mixed_source_rows, strict=True)
        ],
    )

    write_table_star(
        "plasticity3d_fixed_source_operator_pmg.tex",
        fill_spec("c l c c c c c"),
        [
            "Ranks",
            "Route",
            "Wall time [s]",
            "Newton iters",
            "Krylov iters",
            final_metric_header(sourcefixed_local_rows + sourcefixed_source_rows),
            "Status",
        ],
        sourcefixed_long_rows(sourcefixed_local_rows, sourcefixed_source_rows),
    )

    write_table(
        "plasticity3d_degree_energy_study.tex",
        "@{}l c c c c@{}",
        ["Element", "Free DOFs", "Energy", "Wall time [s]", "Status"],
        [
            [
                element_label(row["degree_line"], row["mesh_alias"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_wall_time(float(row["total_time_s"])),
                "reused" if bool(row.get("reused", False)) else str(row["status"]),
            ]
            for row in degree_energy_rows
        ],
    )

    topo_best = [row for row in topo_rows if row["result"] == "completed"]
    topo_best.sort(key=lambda row: int(row["ranks"]))
    write_table_star(
        "topology_summary.tex",
        fill_spec("c c c c c c c"),
        ["Ranks", "Wall time [s]", "Solve time [s]", "Outer iters", "$p$", "Compliance", "Volume fraction"],
        [
            [
                fmt_count(row["ranks"]),
                fmt_wall_time(float(row["wall_time_s"])),
                fmt_wall_time(float(row["solve_time_s"])),
                fmt_count(row["outer_iterations"]),
                fmt_float(float(row["final_p_penal"]), 2),
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
            ]
            for row in topo_best
        ],
    )

    write_table_star(
        "source_continuation_compare.tex",
        fill_spec("l c c c c c"),
        ["Policy", "Ranks", "Runtime [s]", "Init Krylov iters", "Continuation Krylov iters", "Final $\\lambda$"],
        [
            [
                "fixed PMG-shell smoother",
                "8",
                fmt_wall_time(float(source8["run_info"]["runtime_seconds"])),
                fmt_count(source8["timings"]["linear"]["init_linear_iterations"]),
                fmt_count(source8["timings"]["linear"]["attempt_linear_iterations_total"]),
                fmt_float(float(source8_progress["lambda_last"]), 6),
            ],
            [
                "fixed PMG-shell smoother",
                "32",
                fmt_wall_time(float(source32["run_info"]["runtime_seconds"])),
                fmt_count(source32["timings"]["linear"]["init_linear_iterations"]),
                fmt_count(source32["timings"]["linear"]["attempt_linear_iterations_total"]),
                fmt_float(float(source32_progress["lambda_last"]), 6),
            ],
        ],
    )

    layer1a_metrics = p3d_validation["layer1a"]["final_metrics"]
    layer2_metrics = p3d_validation["layer2"]
    endpoint_dev = layer2_metrics.get("endpoint_deviatoric_strain_relative_l2")
    write_table_star(
        "plasticity3d_validation_summary.tex",
        "@{}c@{\\hspace{1.0em}}" + pcol(r"0.36\textwidth") + r"@{\extracolsep{\fill}}c c@{}",
        ["Layer", "Comparison", "Relative difference", "Status"],
        [
            ["1A", "work", fmt_sci(float(layer1a_metrics["work_relative_difference"])), "--"],
            ["1A", "displacement relative $L^2$", fmt_sci(float(layer1a_metrics["displacement_relative_l2"])), "--"],
            [
                "1A",
                "deviatoric-strain relative $L^2$",
                fmt_sci(float(layer1a_metrics["deviatoric_strain_relative_l2"])),
                "--",
            ],
            [
                "2",
                "highest-successful $\\lambda$",
                fmt_sci(float(layer2_metrics["critical_lambda_schedule_proxy"]["relative_difference"])),
                _layer2_gate_status(layer2_metrics, "critical_lambda_pass"),
            ],
            [
                "2",
                "$u_{\\max}(\\lambda)$ relative $L^2$",
                fmt_sci(float(layer2_metrics["umax_curve_relative_l2"])),
                _layer2_gate_status(layer2_metrics, "umax_curve_pass"),
            ],
            [
                "2",
                "endpoint displacement relative $L^2$",
                fmt_sci(float(layer2_metrics["endpoint_displacement_relative_l2"])),
                _layer2_gate_status(layer2_metrics, "endpoint_disp_pass"),
            ],
            [
                "2",
                "endpoint deviatoric-strain relative $L^2$",
                fmt_sci(float(endpoint_dev)) if endpoint_dev is not None else "--",
                "diagnostic",
            ],
            [
                "2",
                "boundary profile relative $L^2$",
                fmt_sci(float(layer2_metrics["boundary_profile_relative_l2"])),
                "diagnostic",
            ],
            [
                "2",
                "acceptance gate",
                "--",
                pass_fail(layer2_metrics["acceptance"]["overall_pass"]),
            ],
        ],
    )

    ablation_rows = [dict(row) for row in p3d_ablation["rows"]]
    write_table_star(
        "plasticity3d_derivative_ablation.tex",
        fill_spec("l c c c c c c c"),
        ["Route", "Wall time [s]", "Solve time [s]", "Newton iters", "Krylov iters", "Energy", "$\\omega$", "$u_{\\max}$"],
        [
            [
                str(row["display_label"]),
                fmt_wall_time(float(row["median_wall_time_s"])),
                fmt_wall_time(float(row["median_solve_time_s"])),
                fmt_count(row["median_nit"]),
                fmt_count(row["median_linear_iterations_total"]),
                fmt_energy(float(row["median_energy"])),
                fmt_energy(float(row["median_omega"])),
                fmt_float(float(row["median_u_max"]), 6),
            ]
            for row in ablation_rows
        ],
    )

    fairness = dict(jax_fem_baseline["fairness_gate"])
    final_metrics = dict(jax_fem_baseline["final_metrics"])
    timing = dict(jax_fem_baseline["timing_medians_s"])
    fairness_checks = dict(fairness["checks"])
    agreement_pass = all(
        bool(fairness_checks[key])
        for key in (
            "energy_rel_diff_le_5pct",
            "field_relative_l2_le_5pct",
            "centerline_relative_l2_le_5pct",
            "umax_curve_relative_l2_le_5pct",
        )
    )
    write_table_star(
        "jax_fem_hyperelastic_baseline.tex",
        "@{}"
        + pcol(r"0.15\textwidth")
        + "@{\\hspace{1.0em}}"
        + pcol(r"0.30\textwidth")
        + r"@{\extracolsep{\fill}}c c c@{}",
        ["Group", "Quantity", "Relative difference", "Median wall time [s]", "Status"],
        [
            ["Agreement", "final energy", fmt_sci(float(final_metrics["energy_rel_diff"])), "--", "--"],
            ["Agreement", "full-field displacement relative $L^2$", fmt_sci(float(final_metrics["field_relative_l2"])), "--", "--"],
            ["Agreement", "centerline relative $L^2$", fmt_sci(float(final_metrics["centerline_relative_l2"])), "--", "--"],
            ["Agreement", "$u_{\\max}$ curve relative $L^2$", fmt_sci(float(final_metrics["umax_curve_relative_l2"])), "--", "--"],
            r"\addlinespace",
            ["Timing", "this work serial direct", "--", fmt_wall_time(float(timing["repo_serial_direct"])), "--"],
            ["Timing", "JAX-FEM UMFPACK serial", "--", fmt_wall_time(float(timing["jax_fem_umfpack_serial"])), "--"],
            r"\addlinespace",
            ["Contract/gate", "same mesh path", "--", "--", pass_fail(fairness_checks["same_mesh_path"])],
            ["Contract/gate", "same displacement schedule", "--", "--", pass_fail(fairness_checks["same_schedule"])],
            ["Contract/gate", "agreement thresholds", "--", "--", pass_fail(agreement_pass)],
            ["Contract/gate", "repository-energy post-comparison", "--", "--", "used"],
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
                "final_metric_name": str(row.get("final_metric_name", "")),
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
        "hyperelasticity_karolina_pmg_scaling": [
            {
                "nodes": int(row["nodes"]),
                "ranks": int(row["ranks"]),
                "coarse_groups": int(row["coarse_groups"]),
                "coarse_group_ranks": int(row["coarse_group_ranks"]),
                "solver_total_s": float(row["solver_total_s"]),
                "first_step_s": float(row["first_step_s"]),
                "newton_iters": int(row["newton_iters"]),
                "linear_iters": int(row["linear_iters"]),
                "energy": float(row["energy"]),
            }
            for row in he_karolina_rows
        ],
    }
    write_json(REPO_ROOT / "paper/build/tables_summary.json", payload)


if __name__ == "__main__":
    main()
