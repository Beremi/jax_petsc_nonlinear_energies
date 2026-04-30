#!/usr/bin/env python3
"""Generate reports and plots for the Plasticity3D validation package."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from experiments.analysis.generate_plasticity3d_p2_lambda1p6_docs_assets import (
    _interpolate_planar_slice as interpolate_planar_slice,
)
from experiments.analysis.generate_plasticity3d_p4_l1_docs_assets import (
    _apply_showcase_camera,
    _surface_plot_arrays,
)
from experiments.analysis.plasticity3d_validation_utils import (
    acceptance_flags,
    compute_boundary_profile,
    critical_lambda_schedule_proxy,
    curve_relative_l2,
    parse_markdown_pipe_table,
    relative_l2,
    write_report,
)
from src.problems.slope_stability_3d.support.mesh import load_case_hdf5, same_mesh_case_hdf5_path
from src.problems.slope_stability_3d.support.simplex_lagrange import evaluate_tetra_lagrange_basis


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "raw_results" / "plasticity3d_validation" / "validation_manifest.json"
SUMMARY_NAME = "comparison_summary.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _repo_path(rel: str) -> Path:
    return (REPO_ROOT / str(rel)).resolve()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=260)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _quadrature_points_tetra_p2() -> np.ndarray:
    return np.array(
        [
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.0714285714285714,
                0.399403576166799,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
                0.100596423833201,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
            ],
            [
                1.0 / 4.0,
                0.0714285714285714,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
            ],
        ],
        dtype=np.float64,
    )


def _deviatoric_strain_norm(strain6: np.ndarray) -> np.ndarray:
    dev_metric = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ) / 3.0
    arr = np.asarray(strain6, dtype=np.float64)
    proj = arr @ dev_metric.T
    return np.sqrt(np.maximum(0.0, np.sum(arr * proj, axis=1)))


def _compute_case_qfields(case, coords_final: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(case.elems_scalar, dtype=np.int64)
    u_elem = np.asarray(displacement[elems], dtype=np.float64)
    ux = u_elem[:, :, 0]
    uy = u_elem[:, :, 1]
    uz = u_elem[:, :, 2]
    dphix = np.asarray(case.dphix, dtype=np.float64)
    dphiy = np.asarray(case.dphiy, dtype=np.float64)
    dphiz = np.asarray(case.dphiz, dtype=np.float64)
    e_xx = np.einsum("eqp,ep->eq", dphix, ux)
    e_yy = np.einsum("eqp,ep->eq", dphiy, uy)
    e_zz = np.einsum("eqp,ep->eq", dphiz, uz)
    g_xy = np.einsum("eqp,ep->eq", dphiy, ux) + np.einsum("eqp,ep->eq", dphix, uy)
    g_yz = np.einsum("eqp,ep->eq", dphiz, uy) + np.einsum("eqp,ep->eq", dphiy, uz)
    g_xz = np.einsum("eqp,ep->eq", dphiz, ux) + np.einsum("eqp,ep->eq", dphix, uz)
    strain = np.stack((e_xx, e_yy, e_zz, g_xy, g_yz, g_xz), axis=-1)
    dev = _deviatoric_strain_norm(strain.reshape(-1, 6)).reshape(strain.shape[:2])
    xi = _quadrature_points_tetra_p2()
    hatp = np.asarray(evaluate_tetra_lagrange_basis(2, xi)[0], dtype=np.float64)
    qcoords = np.einsum("pq,epd->eqd", hatp, np.asarray(coords_final[elems], dtype=np.float64))
    return qcoords, dev


def _surface_plot(ax, coords_final: np.ndarray, surface_faces: np.ndarray, nodal_values: np.ndarray, *, title: str, scalar_label: str, cmap_name: str, clim: tuple[float, float]) -> None:
    coords_plot, tri_plot, values = _surface_plot_arrays(
        coords_final,
        surface_faces,
        nodal_values,
        degree=2,
        subdivisions=2,
    )
    tri_vals = np.mean(values[np.asarray(tri_plot, dtype=np.int64)], axis=1)
    norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    cmap = matplotlib.colormaps[cmap_name]
    poly = Poly3DCollection(
        coords_plot[np.asarray(tri_plot, dtype=np.int64)],
        facecolors=cmap(norm(tri_vals)),
        linewidths=0.0,
        edgecolors="none",
        alpha=1.0,
    )
    poly.set_rasterized(True)
    ax.add_collection3d(poly)
    _apply_showcase_camera(ax, coords_plot)
    ax.set_axis_off()
    ax.set_title(title, pad=5.0)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    return sm


def _plot_surface_compare(
    *,
    coords_source: np.ndarray,
    coords_candidate: np.ndarray,
    surface_faces: np.ndarray,
    values_source: np.ndarray,
    values_candidate: np.ndarray,
    out_base: Path,
    title_prefix: str,
) -> None:
    diff = np.abs(np.asarray(values_candidate, dtype=np.float64) - np.asarray(values_source, dtype=np.float64))
    finite_main = np.concatenate([np.asarray(values_source, dtype=np.float64), np.asarray(values_candidate, dtype=np.float64)])
    main_vmax = float(np.quantile(finite_main, 0.995))
    diff_vmax = float(max(np.quantile(diff, 0.995), 1.0e-12))
    fig = plt.figure(figsize=(13.8, 4.8), dpi=170)
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    sm_main = _surface_plot(
        axes[0],
        np.asarray(coords_source, dtype=np.float64),
        np.asarray(surface_faces, dtype=np.int64),
        np.asarray(values_source, dtype=np.float64),
        title=f"{title_prefix}: reference",
        scalar_label="u magnitude",
        cmap_name="viridis",
        clim=(0.0, max(main_vmax, 1.0e-12)),
    )
    _surface_plot(
        axes[1],
        np.asarray(coords_candidate, dtype=np.float64),
        np.asarray(surface_faces, dtype=np.int64),
        np.asarray(values_candidate, dtype=np.float64),
        title=f"{title_prefix}: maintained",
        scalar_label="u magnitude",
        cmap_name="viridis",
        clim=(0.0, max(main_vmax, 1.0e-12)),
    )
    sm_diff = _surface_plot(
        axes[2],
        np.asarray(coords_candidate, dtype=np.float64),
        np.asarray(surface_faces, dtype=np.int64),
        np.asarray(diff, dtype=np.float64),
        title=f"{title_prefix}: abs diff",
        scalar_label="|du|",
        cmap_name="cividis",
        clim=(0.0, diff_vmax),
    )
    fig.colorbar(sm_main, ax=axes[:2], fraction=0.018, pad=0.01, label="u magnitude")
    fig.colorbar(sm_diff, ax=axes[2], fraction=0.025, pad=0.01, label="|du|")
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_slice_compare(
    *,
    points_source: np.ndarray,
    points_candidate: np.ndarray,
    values_source: np.ndarray,
    values_candidate: np.ndarray,
    footprint_points: np.ndarray,
    footprint_tetrahedra: np.ndarray,
    axis: int,
    out_base: Path,
    title_prefix: str,
) -> None:
    axis = int(axis)
    axis_min = float(np.min(points_candidate[:, axis]))
    axis_max = float(np.max(points_candidate[:, axis]))
    center = float(0.5 * (axis_min + axis_max))
    half = float(0.02 * max(axis_max - axis_min, 1.0e-12))
    source_slice = interpolate_planar_slice(
        points_source,
        values_source,
        axis=axis,
        center=center,
        half_thickness=half,
        footprint_points=footprint_points,
        footprint_tetrahedra=footprint_tetrahedra,
        resolution=760,
    )
    candidate_slice = interpolate_planar_slice(
        points_candidate,
        values_candidate,
        axis=axis,
        center=center,
        half_thickness=half,
        footprint_points=footprint_points,
        footprint_tetrahedra=footprint_tetrahedra,
        resolution=760,
    )
    source_image = np.asarray(source_slice["image"], dtype=np.float64)
    candidate_image = np.asarray(candidate_slice["image"], dtype=np.float64)
    if source_image.shape != candidate_image.shape:
        ny = min(int(source_image.shape[0]), int(candidate_image.shape[0]))
        nx = min(int(source_image.shape[1]), int(candidate_image.shape[1]))
        source_image = source_image[:ny, :nx]
        candidate_image = candidate_image[:ny, :nx]
    diff_image = candidate_image - source_image
    finite_main = np.isfinite(source_image) & np.isfinite(candidate_image)
    vmax = float(
        np.quantile(
            np.concatenate(
                [
                    source_image[finite_main],
                    candidate_image[finite_main],
                ]
            ),
            0.995,
        )
    )
    vmax = max(vmax, 1.0e-12)
    diff_clip = float(np.quantile(np.abs(diff_image[np.isfinite(diff_image)]), 0.995))
    diff_clip = max(diff_clip, 1.0e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=180)
    images = [
        (source_image, "reference", "magma", 0.0, vmax),
        (candidate_image, "maintained", "magma", 0.0, vmax),
        (np.asarray(diff_image, dtype=np.float64), "diff", "coolwarm", -diff_clip, diff_clip),
    ]
    main_im = None
    diff_im = None
    for ax, (image, subtitle, cmap_name, vmin, vmax_use) in zip(axes, images, strict=True):
        im = ax.imshow(
            image,
            origin="lower",
            extent=tuple(source_slice["extent"]),
            cmap=cmap_name,
            vmin=float(vmin),
            vmax=float(vmax_use),
            interpolation="bilinear",
            aspect="equal",
        )
        ax.set_xlabel(str(source_slice["xlabel"]))
        ax.set_ylabel(str(source_slice["ylabel"]))
        ax.set_title(f"{title_prefix}: {subtitle}")
        if subtitle == "diff":
            diff_im = im
        else:
            main_im = im
    if main_im is not None:
        fig.colorbar(main_im, ax=axes[:2], fraction=0.025, pad=0.02, label="deviatoric strain")
    if diff_im is not None:
        fig.colorbar(diff_im, ax=axes[2], fraction=0.04, pad=0.02, label="diff")
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_scalar_curve(x: np.ndarray, y_ref: np.ndarray, y_candidate: np.ndarray, out_base: Path, *, ylabel: str, title: str, ref_label: str = "reference", cand_label: str = "maintained") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    ax.plot(x, y_ref, marker="o", label=ref_label)
    ax.plot(x, y_candidate, marker="s", label=cand_label)
    ax.set_xlabel("lambda")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_boundary_profile(profile_ref: dict[str, np.ndarray], profile_candidate: dict[str, np.ndarray], out_base: Path, *, title: str, ref_label: str = "reference", cand_label: str = "maintained") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    ax.plot(profile_ref["x"], profile_ref["u_mag"], label=ref_label)
    ax.plot(profile_candidate["x"], profile_candidate["u_mag"], label=cand_label)
    ax.set_xlabel("x")
    ax.set_ylabel("boundary displacement magnitude")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_base)


def _source_mesh_mapping(
    source_root: Path,
    maintained_coords_ref: np.ndarray,
    *,
    boundary_type: int,
) -> dict[str, np.ndarray | float | bool]:
    sys.path.insert(0, str((source_root / "src").resolve()))
    try:
        from slope_stability.mesh import load_mesh_from_file, reorder_mesh_nodes  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Loading the source PETSc mesh requires the source package and the "
            "`meshio` dependency inside the main repo environment."
        ) from exc
    mesh_path = source_root / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
    mesh = load_mesh_from_file(mesh_path, boundary_type=int(boundary_type), elem_type="P2")
    reordered = reorder_mesh_nodes(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, strategy="block_xyz")
    coords_source = np.asarray(reordered.coord, dtype=np.float64).T
    dist, src_to_maint = cKDTree(np.asarray(maintained_coords_ref, dtype=np.float64)).query(coords_source, k=1)
    inv = np.empty_like(src_to_maint)
    inv[src_to_maint] = np.arange(src_to_maint.size, dtype=np.int64)
    return {
        "coords_source": coords_source,
        "inv": inv,
        "max_node_distance": float(np.max(dist)),
        "q_mask_source": np.asarray(reordered.q_mask, dtype=bool).T,
    }


def _load_layer2_source_displacement(npz_path: Path, inv: np.ndarray) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    u = np.asarray(data["U"], dtype=np.float64).T
    return np.asarray(u[np.asarray(inv, dtype=np.int64)], dtype=np.float64)


def _load_layer1a_source_displacement(final_state_mat: Path, inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat = scipy.io.loadmat(final_state_mat)
    disp = np.asarray(mat["U_final"], dtype=np.float64).T[np.asarray(inv, dtype=np.int64)]
    q_src = np.asarray(mat["Q"], dtype=bool).T.reshape(-1)
    force_src = np.asarray(mat["f_V"].toarray(), dtype=np.float64).T.reshape((-1, 3))[np.asarray(inv, dtype=np.int64)].reshape(-1)
    return (
        disp,
        np.asarray(mat["coord"], dtype=np.float64).T,
        q_src,
        force_src,
    )


def _build_layer1a(layer_cfg: dict[str, object], out_dir: Path) -> dict[str, object]:
    source_branch = _repo_path(str(layer_cfg["source_branch_dir"]))
    jax_branch = _repo_path(str(layer_cfg["jax_branch_dir"]))
    source_summary = _read_json(source_branch / "branch_summary.json")
    jax_summary = _read_json(jax_branch / "branch_summary.json")
    case = load_case_hdf5(same_mesh_case_hdf5_path("hetero_ssr_L1", 2))
    coords_ref = np.asarray(case.nodes, dtype=np.float64)
    source_final_mat = Path(str(source_summary["final_state_mat"]))
    final_jax_step = dict(jax_summary["steps"][-1])
    jax_state = np.load(Path(final_jax_step["state_npz"]))
    coords_jax_ref = np.asarray(jax_state["coords_ref"], dtype=np.float64)
    dist, src_to_jax = cKDTree(coords_jax_ref).query(np.asarray(scipy.io.loadmat(source_final_mat)["coord"], dtype=np.float64).T, k=1)
    inv = np.empty_like(src_to_jax)
    inv[src_to_jax] = np.arange(src_to_jax.size, dtype=np.int64)
    source_disp, _source_coord_ref, q_src, force_src = _load_layer1a_source_displacement(source_final_mat, inv)
    source_coords_final = coords_ref + source_disp
    candidate_disp = np.asarray(jax_state["displacement"], dtype=np.float64)
    candidate_coords_final = np.asarray(jax_state["coords_final"], dtype=np.float64)
    qcoords_source, dev_source = _compute_case_qfields(case, source_coords_final, source_disp)
    qcoords_candidate, dev_candidate = _compute_case_qfields(case, candidate_coords_final, candidate_disp)
    q_jax = np.zeros(coords_ref.shape[0] * 3, dtype=bool)
    q_jax[np.asarray(case.freedofs, dtype=np.int64)] = True
    force_jax = np.asarray(case.force, dtype=np.float64)
    work_rel = abs(float(source_summary["final_work"]) - float(final_jax_step["work"])) / max(abs(float(source_summary["final_work"])), 1.0e-30)
    disp_rel = relative_l2(source_disp, candidate_disp)
    dev_rel = relative_l2(dev_source, dev_candidate)

    assets_dir = out_dir / "layer1a" / "assets"
    _plot_scalar_curve(
        np.asarray(source_summary["lambda_hist"], dtype=np.float64),
        np.asarray(source_summary["work_hist"], dtype=np.float64),
        np.asarray(jax_summary["work_hist"], dtype=np.float64),
        assets_dir / "branch_work_history",
        ylabel="external work",
        title="Layer 1A: direct-branch work history",
        ref_label="Octave",
        cand_label="JAX",
    )
    _plot_surface_compare(
        coords_source=source_coords_final,
        coords_candidate=candidate_coords_final,
        surface_faces=np.asarray(jax_state["surface_faces"], dtype=np.int64),
        values_source=np.linalg.norm(source_disp, axis=1),
        values_candidate=np.linalg.norm(candidate_disp, axis=1),
        out_base=assets_dir / "deformed_boundary_compare",
        title_prefix="Layer 1A boundary",
    )
    for axis, axis_name in enumerate(("x", "y", "z")):
        _plot_slice_compare(
            points_source=np.asarray(qcoords_source, dtype=np.float64).reshape(-1, 3),
            points_candidate=np.asarray(qcoords_candidate, dtype=np.float64).reshape(-1, 3),
            values_source=np.asarray(dev_source, dtype=np.float64).reshape(-1),
            values_candidate=np.asarray(dev_candidate, dtype=np.float64).reshape(-1),
            footprint_points=np.asarray(candidate_coords_final, dtype=np.float64),
            footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
            axis=axis,
            out_base=assets_dir / f"deviatoric_strain_slice_{axis_name}_compare",
            title_prefix=f"Layer 1A {axis_name}-slice",
        )

    baseline = dict(layer_cfg["accepted_baseline"])
    return {
        "kind": "exact_source_faithfulness",
        "schedule": [float(v) for v in source_summary["lambda_hist"]],
        "structural_checks": {
            "node_map_max_abs_diff": float(np.max(dist)),
            "free_mask_exact": bool(np.array_equal(q_src.reshape((-1, 3))[inv].reshape(-1), q_jax)),
            "force_relative_diff": relative_l2(force_jax, force_src),
        },
        "final_metrics": {
            "work_relative_difference": float(work_rel),
            "displacement_relative_l2": float(disp_rel),
            "deviatoric_strain_relative_l2": float(dev_rel),
        },
        "accepted_baseline": baseline,
        "assets": {
            "branch_history": str((assets_dir / "branch_work_history.pdf").relative_to(out_dir)),
            "deformed_boundary_compare": str((assets_dir / "deformed_boundary_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_x_compare": str((assets_dir / "deviatoric_strain_slice_x_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_y_compare": str((assets_dir / "deviatoric_strain_slice_y_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_z_compare": str((assets_dir / "deviatoric_strain_slice_z_compare.pdf").relative_to(out_dir)),
        },
    }


def _build_layer1b(layer_cfg: dict[str, object], out_dir: Path) -> dict[str, object]:
    report_path = _repo_path(str(layer_cfg["report_md"]))
    report_text = report_path.read_text(encoding="utf-8")
    summary_rows = parse_markdown_pipe_table(report_text, "## Summary")
    step_rows = parse_markdown_pipe_table(report_text, "## Accepted-Step Table")
    parsed_summary = {row["Metric"]: row["MATLAB"] if "MATLAB" in row else row["value"] for row in summary_rows}
    lambdas_matlab = np.asarray([float(row["MATLAB lambda"]) for row in step_rows if row["MATLAB lambda"] != "-"], dtype=np.float64)
    lambdas_petsc = np.asarray([float(row["PETSc lambda"]) for row in step_rows if row["PETSc lambda"] != "-"], dtype=np.float64)
    omega_matlab = np.asarray([float(row["MATLAB omega"]) for row in step_rows if row["MATLAB omega"] != "-"], dtype=np.float64)
    omega_petsc = np.asarray([float(row["PETSc omega"]) for row in step_rows if row["PETSc omega"] != "-"], dtype=np.float64)
    assets_dir = out_dir / "layer1b" / "assets"
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), dpi=180)
    axes[0].plot(lambdas_matlab, marker="o", label="MATLAB")
    axes[0].plot(lambdas_petsc, marker="s", label="PETSc")
    axes[0].set_title("Accepted lambda history")
    axes[0].set_xlabel("step")
    axes[0].legend(loc="best")
    axes[1].plot(omega_matlab, marker="o", label="MATLAB")
    axes[1].plot(omega_petsc, marker="s", label="PETSc")
    axes[1].set_title("Omega history")
    axes[1].set_xlabel("step")
    final_umax_matlab = float(next(row["MATLAB"] for row in summary_rows if row["Metric"] == "Final Umax"))
    final_umax_petsc = float(next(row["PETSc"] for row in summary_rows if row["Metric"] == "Final Umax"))
    axes[2].bar([0, 1], [final_umax_matlab, final_umax_petsc], color=("#4c78a8", "#f58518"))
    axes[2].set_xticks([0, 1], ["MATLAB", "PETSc"])
    axes[2].set_title("Final Umax")
    axes[2].set_ylabel("Umax")
    fig.tight_layout()
    _save_figure(fig, assets_dir / "source_family_history")
    return {
        "kind": "published_source_family_triangulation",
        "report_md": _repo_rel(report_path),
        "summary": {
            "runtime_matlab_s": float(parsed_summary["Runtime [s]"]),
            "runtime_petsc_s": float(next(row["PETSc"] for row in summary_rows if row["Metric"] == "Runtime [s]")),
            "relative_lambda_history_error": float(next(row["MATLAB"] for row in summary_rows if row["Metric"] == "Relative lambda history error")),
            "relative_omega_history_error": float(next(row["MATLAB"] for row in summary_rows if row["Metric"] == "Relative omega history error")),
            "relative_umax_history_error": float(next(row["MATLAB"] for row in summary_rows if row["Metric"] == "Relative Umax history error")),
        },
        "assets": {
            "source_family_history": str((assets_dir / "source_family_history.pdf").relative_to(out_dir)),
        },
        "notes": "Context / published source-family evidence only; not the same glued-bottom Plasticity3D case.",
    }


def _build_layer2(manifest: dict[str, object], out_dir: Path) -> dict[str, object]:
    rows_cfg = list(manifest["layer2"]["rows"])
    case = load_case_hdf5(same_mesh_case_hdf5_path("hetero_ssr_L1", 2, "glued_bottom"))
    coords_ref = np.asarray(case.nodes, dtype=np.float64)
    source_boundary_type = int(
        dict(manifest.get("validation_contract", {})).get("source_mesh_boundary_type", 0)
    )
    mapping = _source_mesh_mapping(
        _repo_path(str(manifest["source_root"])),
        coords_ref,
        boundary_type=source_boundary_type,
    )
    inv = np.asarray(mapping["inv"], dtype=np.int64)
    q_jax = np.zeros(coords_ref.shape[0] * 3, dtype=bool)
    q_jax[np.asarray(case.freedofs, dtype=np.int64)] = True

    rows: list[dict[str, object]] = []
    endpoint_bundle: dict[str, np.ndarray] | None = None
    for cfg in rows_cfg:
        lam = float(cfg["lambda_value"])
        maintained_output = _read_json(_repo_path(str(cfg["maintained"]["output_json"])))
        maintained_state = np.load(_repo_path(str(cfg["maintained"]["state_npz"])))
        source_output = _read_json(_repo_path(str(cfg["source_reference"]["output_json"])))
        source_disp = _load_layer2_source_displacement(_repo_path(str(cfg["source_reference"]["petsc_run_npz"])), inv)
        maintained_disp = np.asarray(maintained_state["displacement"], dtype=np.float64)
        maintained_coords_final = np.asarray(maintained_state["coords_final"], dtype=np.float64)
        source_coords_final = coords_ref + source_disp
        qcoords_source, dev_source = _compute_case_qfields(case, source_coords_final, source_disp)
        qcoords_candidate, dev_candidate = _compute_case_qfields(case, maintained_coords_final, maintained_disp)
        row = {
            "lambda_value": float(lam),
            "source_solver_success": bool(source_output.get("solver_success", False)),
            "maintained_solver_success": bool(maintained_output.get("solver_success", False)),
            "source_energy": float(source_output.get("energy", float("nan"))),
            "maintained_energy": float(maintained_output.get("energy", float("nan"))),
            "source_omega": float(source_output.get("omega", float("nan"))),
            "maintained_omega": float(maintained_output.get("omega", float("nan"))),
            "source_u_max": float(source_output.get("u_max", float("nan"))),
            "maintained_u_max": float(maintained_output.get("u_max", float("nan"))),
            "displacement_relative_l2": float(relative_l2(source_disp, maintained_disp)),
            "deviatoric_strain_relative_l2": float(relative_l2(dev_source, dev_candidate)),
        }
        rows.append(row)
        if math.isclose(lam, float(manifest["validation_contract"]["schedule"][-1]), rel_tol=0.0, abs_tol=1.0e-12):
            endpoint_bundle = {
                "source_disp": source_disp,
                "maintained_disp": maintained_disp,
                "source_coords_final": source_coords_final,
                "maintained_coords_final": maintained_coords_final,
                "qcoords_source": qcoords_source.reshape(-1, 3),
                "qcoords_candidate": qcoords_candidate.reshape(-1, 3),
                "dev_source": dev_source.reshape(-1),
                "dev_candidate": dev_candidate.reshape(-1),
                "surface_faces": np.asarray(maintained_state["surface_faces"], dtype=np.int64),
            }

    schedule = np.asarray([row["lambda_value"] for row in rows], dtype=np.float64)
    u_source = np.asarray([row["source_u_max"] for row in rows], dtype=np.float64)
    u_candidate = np.asarray([row["maintained_u_max"] for row in rows], dtype=np.float64)
    critical_source = critical_lambda_schedule_proxy(
        [{"lambda_value": row["lambda_value"], "solver_success": row["source_solver_success"]} for row in rows]
    )
    critical_candidate = critical_lambda_schedule_proxy(
        [{"lambda_value": row["lambda_value"], "solver_success": row["maintained_solver_success"]} for row in rows]
    )
    critical_rel = abs(float(critical_source) - float(critical_candidate)) / max(abs(float(critical_source)), 1.0e-30)
    umax_rel = curve_relative_l2(schedule, u_source, schedule, u_candidate)

    endpoint_disp_rel = float("nan")
    endpoint_dev_rel = float("nan")
    boundary_rel = float("nan")
    assets: dict[str, str] = {}
    if endpoint_bundle is not None:
        endpoint_disp_rel = relative_l2(endpoint_bundle["source_disp"], endpoint_bundle["maintained_disp"])
        endpoint_dev_rel = relative_l2(endpoint_bundle["dev_source"], endpoint_bundle["dev_candidate"])
        profile_source = compute_boundary_profile(coords_ref, endpoint_bundle["source_coords_final"])
        profile_candidate = compute_boundary_profile(coords_ref, endpoint_bundle["maintained_coords_final"])
        boundary_rel = relative_l2(profile_source["u_mag"], np.interp(profile_source["x"], profile_candidate["x"], profile_candidate["u_mag"]))
        assets_dir = out_dir / "layer2" / "assets"
        _plot_scalar_curve(
            schedule,
            u_source,
            u_candidate,
            assets_dir / "umax_curve",
            ylabel="u_max",
            title="Layer 2: u_max(lambda)",
            ref_label="source operator reference",
            cand_label="maintained surrogate",
        )
        _plot_boundary_profile(
            profile_source,
            profile_candidate,
            assets_dir / "boundary_profile",
            title="Layer 2: boundary displacement profile",
            ref_label="source operator reference",
            cand_label="maintained surrogate",
        )
        _plot_surface_compare(
            coords_source=endpoint_bundle["source_coords_final"],
            coords_candidate=endpoint_bundle["maintained_coords_final"],
            surface_faces=endpoint_bundle["surface_faces"],
            values_source=np.linalg.norm(endpoint_bundle["source_disp"], axis=1),
            values_candidate=np.linalg.norm(endpoint_bundle["maintained_disp"], axis=1),
            out_base=assets_dir / "deformed_boundary_compare",
            title_prefix="Layer 2 boundary",
        )
        for axis, axis_name in enumerate(("x", "y", "z")):
            _plot_slice_compare(
                points_source=endpoint_bundle["qcoords_source"],
                points_candidate=endpoint_bundle["qcoords_candidate"],
                values_source=endpoint_bundle["dev_source"],
                values_candidate=endpoint_bundle["dev_candidate"],
                footprint_points=endpoint_bundle["maintained_coords_final"],
                footprint_tetrahedra=np.asarray(case.elems_scalar, dtype=np.int64),
                axis=axis,
                out_base=assets_dir / f"deviatoric_strain_slice_{axis_name}_compare",
                title_prefix=f"Layer 2 {axis_name}-slice",
            )
        assets = {
            "umax_curve": str((assets_dir / "umax_curve.pdf").relative_to(out_dir)),
            "boundary_profile": str((assets_dir / "boundary_profile.pdf").relative_to(out_dir)),
            "deformed_boundary_compare": str((assets_dir / "deformed_boundary_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_x_compare": str((assets_dir / "deviatoric_strain_slice_x_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_y_compare": str((assets_dir / "deviatoric_strain_slice_y_compare.pdf").relative_to(out_dir)),
            "deviatoric_strain_slice_z_compare": str((assets_dir / "deviatoric_strain_slice_z_compare.pdf").relative_to(out_dir)),
        }

    flags = acceptance_flags(
        critical_lambda_rel_diff=float(critical_rel),
        umax_curve_rel_l2=float(umax_rel),
        endpoint_disp_rel_l2=float(endpoint_disp_rel),
    )
    return {
        "kind": "fixed_lambda_source_operator_validation",
        "mapping_checks": {
            "node_map_max_abs_diff": float(mapping["max_node_distance"]),
            "free_mask_exact": bool(
                np.array_equal(
                    np.asarray(mapping["q_mask_source"], dtype=bool)[inv].reshape(-1),
                    q_jax,
                )
            ),
        },
        "rows": rows,
        "critical_lambda_schedule_proxy": {
            "source_reference": float(critical_source),
            "maintained_surrogate": float(critical_candidate),
            "relative_difference": float(critical_rel),
        },
        "umax_curve_relative_l2": float(umax_rel),
        "endpoint_displacement_relative_l2": float(endpoint_disp_rel),
        "endpoint_deviatoric_strain_relative_l2": float(endpoint_dev_rel),
        "boundary_profile_relative_l2": float(boundary_rel),
        "acceptance": {
            **flags,
            "overall_pass": bool(all(flags.values())),
        },
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = _read_json(Path(args.manifest_json))
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else Path(args.manifest_json).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    layer1a = _build_layer1a(dict(manifest["layer1a"]), out_dir)
    layer1b = _build_layer1b(dict(manifest["layer1b"]), out_dir)
    layer2 = _build_layer2(manifest, out_dir)

    summary = {
        "runner": str(manifest["runner"]),
        "manifest_json": str(Path(args.manifest_json).resolve()),
        "layer1a": layer1a,
        "layer1b": layer1b,
        "layer2": layer2,
    }
    _write_json(out_dir / SUMMARY_NAME, summary)

    report_lines = [
        "# Plasticity3D validation package",
        "",
        "## Layer 1A: exact source-faithfulness",
        "",
        f"- accepted schedule: `{layer1a['schedule']}`",
        f"- work relative difference: `{layer1a['final_metrics']['work_relative_difference']:.6e}`",
        f"- displacement relative L2: `{layer1a['final_metrics']['displacement_relative_l2']:.6e}`",
        f"- deviatoric-strain relative L2: `{layer1a['final_metrics']['deviatoric_strain_relative_l2']:.6e}`",
        "",
        "## Layer 1B: published source-family triangulation",
        "",
        f"- relative lambda history error: `{layer1b['summary']['relative_lambda_history_error']:.6e}`",
        f"- relative omega history error: `{layer1b['summary']['relative_omega_history_error']:.6e}`",
        f"- relative Umax history error: `{layer1b['summary']['relative_umax_history_error']:.6e}`",
        "",
        "## Layer 2: fixed-lambda source-operator validation",
        "",
        f"- highest-successful-lambda schedule proxy relative difference: `{layer2['critical_lambda_schedule_proxy']['relative_difference']:.6e}`",
        f"- u_max(lambda) relative L2: `{layer2['umax_curve_relative_l2']:.6e}`",
        f"- endpoint displacement relative L2: `{layer2['endpoint_displacement_relative_l2']:.6e}`",
        f"- endpoint deviatoric-strain relative L2: `{layer2['endpoint_deviatoric_strain_relative_l2']:.6e}`",
        f"- boundary-profile relative L2: `{layer2['boundary_profile_relative_l2']:.6e}`",
        f"- acceptance: `{layer2['acceptance']}`",
        "",
    ]
    write_report(out_dir / "REPORT.md", report_lines)
    print(out_dir / SUMMARY_NAME)


if __name__ == "__main__":
    main()
