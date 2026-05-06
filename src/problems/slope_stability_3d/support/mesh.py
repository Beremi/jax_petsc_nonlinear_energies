"""Mesh import, HDF5 IO, and same-mesh FE helpers for 3D heterogeneous SSR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import runpy
import shutil

import h5py
from mpi4py import MPI
import numpy as np
import scipy.sparse as sp

from src.core.problem_data.hdf5 import (
    MESH_DATA_ROOT,
    load_problem_hdf5,
    load_problem_hdf5_fields,
)
from src.core.petsc.dof_partition import petsc_ownership_range
from src.core.petsc.reordered_element_base import (
    _owned_pattern_from_local_elems,
    _owned_pattern_from_local_scalar_elems,
    inverse_permutation,
    select_permutation,
)
from src.problems.slope_stability_3d.support.materials import (
    MaterialSpec,
    heterogenous_materials_qp,
)
from src.problems.slope_stability_3d.support.simplex_lagrange import (
    evaluate_tetra_lagrange_basis,
    tetra_reference_nodes,
    triangle_lagrange_interior_tuples,
    triangle_lagrange_node_tuples,
)


COORD_DECIMALS = 12
DEFAULT_MESH_NAME = "hetero_ssr_L1"
RAW_MESH_ROOT = MESH_DATA_ROOT / "SlopeStability3D" / "hetero_ssr"
DEFINITION_PATH = RAW_MESH_ROOT / "definition.py"
_MESH_CASE_RE = re.compile(r"^SSR_hetero_(?P<kind>ada_L\d+|uni)\.msh$")
_REFINED_MESH_RE = re.compile(r"^(?P<base>hetero_ssr_L\d+)(?P<suffix>(?:_\d+)*)$")
SOURCE_INTERNAL_AXIS_ORDER = "xyz"
SOURCE_INTERNAL_AXIS_PERM = np.asarray([0, 1, 2], dtype=np.int64)
SAME_MESH_HDF5_SCHEMA_VERSION = 6
PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM = "componentwise_bottom"
PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM = "glued_bottom"
DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT = PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM
_PLASTICITY3D_CONSTRAINT_VARIANTS = frozenset(
    {
        PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM,
        PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM,
    }
)
_LIGHT_CASE_CACHE: dict[tuple[str, int, str], dict[str, object]] = {}
_RANK_LOCAL_LIGHT_CACHE: dict[tuple[str, int, str, str, int, int], dict[str, object]] = {}
_RANK_LOCAL_HEAVY_CACHE: dict[tuple[str, int, str, str, int, int], dict[str, object]] = {}


def clear_same_mesh_case_hdf5_caches() -> None:
    """Release in-process same-mesh HDF5 payload caches."""

    _LIGHT_CASE_CACHE.clear()
    _RANK_LOCAL_LIGHT_CACHE.clear()
    _RANK_LOCAL_HEAVY_CACHE.clear()


@dataclass(frozen=True)
class SlopeStability3DCaseData:
    case_name: str
    mesh_name: str
    degree: int
    raw_mesh_filename: str
    constraint_variant: str

    nodes: np.ndarray
    elems_scalar: np.ndarray
    elems: np.ndarray
    surf: np.ndarray
    boundary_label: np.ndarray
    q_mask: np.ndarray
    freedofs: np.ndarray

    dphix: np.ndarray
    dphiy: np.ndarray
    dphiz: np.ndarray
    quad_weight: np.ndarray

    force: np.ndarray
    u_0: np.ndarray

    material_id: np.ndarray
    c0_q: np.ndarray
    phi_q: np.ndarray
    psi_q: np.ndarray
    shear_q: np.ndarray
    bulk_q: np.ndarray
    lame_q: np.ndarray
    gamma_q: np.ndarray
    eps_p_old: np.ndarray

    adjacency: sp.coo_matrix | None
    elastic_kernel: np.ndarray
    macro_parent: np.ndarray
    macro_parent_mesh_name: str

    davis_type: str = "B"
    lambda_target_default: float = 1.0
    gravity_axis: int = 1

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elems_scalar.shape[0])

    @property
    def n_q(self) -> int:
        return int(self.quad_weight.shape[1])


@dataclass(frozen=True)
class _MacroMeshData:
    mesh_name: str
    raw_mesh_filename: str
    nodes: np.ndarray
    elems: np.ndarray
    surf: np.ndarray
    boundary_label: np.ndarray
    material_id: np.ndarray
    macro_parent: np.ndarray
    macro_parent_mesh_name: str


def normalize_constraint_variant(constraint_variant: str | None) -> str:
    variant = str(
        DEFAULT_PLASTICITY3D_CONSTRAINT_VARIANT if constraint_variant is None else constraint_variant
    ).strip()
    if variant not in _PLASTICITY3D_CONSTRAINT_VARIANTS:
        supported = ", ".join(sorted(_PLASTICITY3D_CONSTRAINT_VARIANTS))
        raise ValueError(
            f"Unsupported Plasticity3D constraint variant {constraint_variant!r}; "
            f"expected one of {supported}"
        )
    return variant


def _point_key(point: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(point, dtype=np.float64), COORD_DECIMALS).tolist())


def _parse_refined_mesh_name(mesh_name: str) -> tuple[str, int] | None:
    mesh_name = str(mesh_name)
    match = _REFINED_MESH_RE.fullmatch(mesh_name)
    if match is None:
        return None
    base = str(match.group("base"))
    suffix = str(match.group("suffix") or "")
    if not suffix:
        return base, 0
    tokens = [int(token) for token in suffix.split("_") if token]
    expected = list(range(2, 2 + len(tokens)))
    if tokens != expected:
        raise ValueError(
            f"Unsupported refined 3D slope mesh name {mesh_name!r}; "
            f"expected sequential suffixes {expected!r}"
        )
    return base, len(tokens)


def mesh_name_with_uniform_refinements(base_mesh_name: str, steps: int) -> str:
    base_mesh_name = str(base_mesh_name)
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if steps == 0:
        return base_mesh_name
    if re.fullmatch(r"hetero_ssr_L\d+", base_mesh_name) is None:
        raise ValueError(f"Unsupported 3D slope mesh base name {base_mesh_name!r}")
    suffix = "".join(f"_{level}" for level in range(2, 2 + steps))
    return f"{base_mesh_name}{suffix}"


def macro_parent_mesh_name_for_name(mesh_name: str) -> str:
    mesh_name = str(mesh_name)
    parsed = _parse_refined_mesh_name(mesh_name)
    if parsed is None:
        return mesh_name
    base_mesh_name, refinement_steps = parsed
    if refinement_steps <= 0:
        return str(base_mesh_name)
    return mesh_name_with_uniform_refinements(str(base_mesh_name), refinement_steps - 1)


def supported_mesh_names() -> tuple[str, ...]:
    names: list[str] = []
    for path in sorted(RAW_MESH_ROOT.glob("*.msh")):
        try:
            names.append(mesh_name_from_raw_filename(path.name))
        except ValueError:
            continue
    for base in tuple(names):
        if re.fullmatch(r"hetero_ssr_L\d+", str(base)):
            for steps in range(1, 4):
                names.append(mesh_name_with_uniform_refinements(str(base), steps))
    return tuple(dict.fromkeys(names))


def base_mesh_name_for_name(mesh_name: str) -> str:
    mesh_name = str(mesh_name)
    parsed = _parse_refined_mesh_name(mesh_name)
    if parsed is not None:
        return str(parsed[0])
    return mesh_name


def uniform_refinement_steps_for_name(mesh_name: str) -> int:
    parsed = _parse_refined_mesh_name(str(mesh_name))
    return int(parsed[1]) if parsed is not None else 0


def mesh_name_from_raw_filename(filename: str) -> str:
    match = _MESH_CASE_RE.match(str(filename))
    if match is None:
        raise ValueError(f"Unsupported raw 3D slope mesh {filename!r}")
    kind = str(match.group("kind"))
    if kind == "uni":
        return "hetero_ssr_uni"
    return f"hetero_ssr_{kind.split('_', 1)[1]}"


def raw_mesh_filename_for_name(mesh_name: str) -> str:
    mesh_name = base_mesh_name_for_name(str(mesh_name))
    if mesh_name == "hetero_ssr_uni":
        return "SSR_hetero_uni.msh"
    if re.fullmatch(r"hetero_ssr_L\d+", mesh_name):
        return f"SSR_hetero_ada_{mesh_name.split('_')[-1]}.msh"
    raise ValueError(f"Unsupported 3D slope mesh name {mesh_name!r}")


def raw_mesh_path_for_name(mesh_name: str) -> Path:
    return RAW_MESH_ROOT / raw_mesh_filename_for_name(mesh_name)


def legacy_unversioned_same_mesh_case_name(mesh_name: str, degree: int) -> str:
    return f"{str(mesh_name)}_p{int(degree)}_same_mesh"


def legacy_unversioned_same_mesh_case_hdf5_path(mesh_name: str, degree: int) -> Path:
    return RAW_MESH_ROOT / f"{legacy_unversioned_same_mesh_case_name(mesh_name, degree)}.h5"


def same_mesh_case_name(mesh_name: str, degree: int, constraint_variant: str | None = None) -> str:
    variant = normalize_constraint_variant(constraint_variant)
    return f"{str(mesh_name)}_p{int(degree)}_same_mesh_{variant}"


def same_mesh_case_hdf5_path(
    mesh_name: str,
    degree: int,
    constraint_variant: str | None = None,
) -> Path:
    return RAW_MESH_ROOT / f"{same_mesh_case_name(mesh_name, degree, constraint_variant)}.h5"


def _load_definition() -> dict[str, object]:
    raw = runpy.run_path(str(DEFINITION_PATH))
    definition = raw.get("DEFINITION")
    if not isinstance(definition, dict):
        raise RuntimeError(f"{DEFINITION_PATH} does not define DEFINITION")
    return definition


def _definition_materials() -> list[MaterialSpec]:
    definition = _load_definition()
    out: list[MaterialSpec] = []
    for item in definition["materials"]:
        out.append(
            MaterialSpec(
                c0=float(item["c0"]),
                phi=float(item["phi"]),
                psi=float(item["psi"]),
                young=float(item["young"]),
                poisson=float(item["poisson"]),
                gamma_sat=float(item["gamma_sat"]),
                gamma_unsat=float(item["gamma_unsat"]),
            )
        )
    return out


def _definition_dirichlet_labels() -> dict[str, tuple[int, ...]]:
    definition = _load_definition()
    raw = dict(definition.get("dirichlet_labels", {}))
    return {
        "x": tuple(int(v) for v in raw.get("x", (1, 2))),
        "y": tuple(int(v) for v in raw.get("y", (5,))),
        "z": tuple(int(v) for v in raw.get("z", (3, 4))),
    }


def _decode_hdf5_string(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _collect_meshio_blocks(mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ValueError("Gmsh mesh must carry 'gmsh:physical' cell data")

    cells_out: list[np.ndarray] = []
    tags_out: list[np.ndarray] = []
    for block, physical in zip(mesh.cells, physical_blocks, strict=False):
        if str(block.type) != str(cell_type):
            continue
        cells_out.append(np.asarray(block.data, dtype=np.int64))
        tags_out.append(np.asarray(physical, dtype=np.int64).ravel())

    if not cells_out:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(cells_out), np.concatenate(tags_out)


def _physical_group_name_map(
    field_data: dict[str, np.ndarray],
    dim: int,
    prefix: str,
) -> dict[int, int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    mapping: dict[int, int] = {}
    for name, meta in field_data.items():
        arr = np.asarray(meta, dtype=np.int64).ravel()
        if arr.size < 2 or int(arr[1]) != int(dim):
            continue
        match = pattern.match(str(name).strip().lower())
        if match is None:
            continue
        mapping[int(arr[0])] = int(match.group(1))
    return mapping


def _map_physical_ids(
    physical_ids: np.ndarray,
    field_data: dict[str, np.ndarray],
    dim: int,
    prefix: str,
) -> np.ndarray:
    ids = np.asarray(physical_ids, dtype=np.int64).ravel()
    mapping = _physical_group_name_map(field_data, dim, prefix)
    if not mapping:
        return ids
    missing = sorted(int(v) for v in np.unique(ids) if int(v) not in mapping)
    if missing:
        raise ValueError(
            f"Physical group mapping for prefix {prefix!r} is incomplete; missing "
            f"logical ids for physical tags {missing}."
        )
    return np.asarray([mapping[int(v)] for v in ids], dtype=np.int64)


def _load_macro_mesh_from_msh(mesh_path: str | Path, mesh_name: str) -> _MacroMeshData:
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Reading .msh files requires the 'meshio' package") from exc

    path = Path(mesh_path)
    msh = meshio.read(path)
    tetra_cells, tetra_tags = _collect_meshio_blocks(msh, "tetra")
    tri_cells, tri_tags = _collect_meshio_blocks(msh, "triangle")
    if tetra_cells.size == 0:
        raise ValueError(f"No tetrahedral cells found in {path}")

    points = np.asarray(msh.points, dtype=np.float64)
    # Source parity: the dry 3D heterogeneous SSR family stores the raw .msh
    # points directly in the same coordinate frame used by load_mesh_P2() in
    # the source benchmark, with the second coordinate as the vertical
    # direction.
    nodes = np.asarray(points[:, SOURCE_INTERNAL_AXIS_PERM], dtype=np.float64)
    material_id = _map_physical_ids(tetra_tags, msh.field_data, 3, "material")
    boundary_label = _map_physical_ids(tri_tags, msh.field_data, 2, "boundary")

    return _MacroMeshData(
        mesh_name=str(mesh_name),
        raw_mesh_filename=path.name,
        nodes=nodes,
        elems=np.asarray(tetra_cells, dtype=np.int64),
        surf=np.asarray(tri_cells, dtype=np.int64),
        boundary_label=np.asarray(boundary_label, dtype=np.int64),
        material_id=np.asarray(material_id, dtype=np.int64),
        macro_parent=np.arange(tetra_cells.shape[0], dtype=np.int64),
        macro_parent_mesh_name=str(mesh_name),
    )


def _macro_midpoint_index(
    nodes: np.ndarray,
    edge_map: dict[tuple[int, int], int],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
) -> int:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    idx = edge_map.get(key)
    if idx is not None:
        return int(idx)
    idx = int(nodes.shape[0] + len(extra_points))
    edge_map[key] = idx
    extra_points.append(0.5 * (nodes[key[0]] + nodes[key[1]]))
    return idx


def _tet_signed_volume6(nodes: np.ndarray, tet: tuple[int, int, int, int]) -> float:
    v0, v1, v2, v3 = (int(v) for v in tet)
    mat = np.column_stack(
        (
            np.asarray(nodes[v1] - nodes[v0], dtype=np.float64),
            np.asarray(nodes[v2] - nodes[v0], dtype=np.float64),
            np.asarray(nodes[v3] - nodes[v0], dtype=np.float64),
        )
    )
    return float(np.linalg.det(mat))


def _ensure_positive_tet_orientation(
    nodes: np.ndarray,
    tet: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if _tet_signed_volume6(nodes, tet) >= 0.0:
        return tuple(int(v) for v in tet)
    a, b, c, d = (int(v) for v in tet)
    return (a, c, b, d)


def _uniform_refine_macro_mesh_once(
    macro: _MacroMeshData,
    *,
    mesh_name: str,
) -> _MacroMeshData:
    nodes = np.asarray(macro.nodes, dtype=np.float64)
    edge_map: dict[tuple[int, int], int] = {}
    extra_points: list[np.ndarray] = []

    refined_tets: list[tuple[int, int, int, int]] = []
    refined_materials: list[int] = []
    refined_parent: list[int] = []

    for tet_id, tet in enumerate(np.asarray(macro.elems, dtype=np.int64)):
        v0, v1, v2, v3 = (int(v) for v in tet)
        m01 = _macro_midpoint_index(nodes, edge_map, extra_points, v0, v1)
        m12 = _macro_midpoint_index(nodes, edge_map, extra_points, v1, v2)
        m02 = _macro_midpoint_index(nodes, edge_map, extra_points, v0, v2)
        m03 = _macro_midpoint_index(nodes, edge_map, extra_points, v0, v3)
        m13 = _macro_midpoint_index(nodes, edge_map, extra_points, v1, v3)
        m23 = _macro_midpoint_index(nodes, edge_map, extra_points, v2, v3)
        children = (
            (v0, m01, m02, m03),
            (m01, v1, m12, m13),
            (m02, m12, v2, m23),
            (m03, m13, m23, v3),
            (m01, m02, m03, m23),
            (m01, m02, m12, m23),
            (m01, m12, m13, m23),
            (m01, m03, m13, m23),
        )
        for child in children:
            refined_tets.append(tuple(int(v) for v in child))
            refined_materials.append(int(macro.material_id[tet_id]))
            refined_parent.append(int(tet_id))

    refined_nodes = (
        np.vstack((nodes, np.asarray(extra_points, dtype=np.float64)))
        if extra_points
        else np.asarray(nodes, dtype=np.float64).copy()
    )
    refined_elems = np.asarray(
        [_ensure_positive_tet_orientation(refined_nodes, tet) for tet in refined_tets],
        dtype=np.int64,
    )

    refined_surf: list[tuple[int, int, int]] = []
    refined_labels: list[int] = []
    for face_id, face in enumerate(np.asarray(macro.surf, dtype=np.int64)):
        v0, v1, v2 = (int(v) for v in face)
        m01 = _macro_midpoint_index(nodes, edge_map, extra_points, v0, v1)
        m12 = _macro_midpoint_index(nodes, edge_map, extra_points, v1, v2)
        m02 = _macro_midpoint_index(nodes, edge_map, extra_points, v0, v2)
        children = (
            (v0, m01, m02),
            (m01, v1, m12),
            (m02, m12, v2),
            (m01, m12, m02),
        )
        for child in children:
            refined_surf.append(tuple(int(v) for v in child))
            refined_labels.append(int(macro.boundary_label[face_id]))

    refined_nodes = (
        np.vstack((nodes, np.asarray(extra_points, dtype=np.float64)))
        if len(extra_points) + nodes.shape[0] != refined_nodes.shape[0]
        else refined_nodes
    )
    return _MacroMeshData(
        mesh_name=str(mesh_name),
        raw_mesh_filename=str(macro.raw_mesh_filename),
        nodes=np.asarray(refined_nodes, dtype=np.float64),
        elems=np.asarray(refined_elems, dtype=np.int64),
        surf=np.asarray(refined_surf, dtype=np.int64),
        boundary_label=np.asarray(refined_labels, dtype=np.int64),
        material_id=np.asarray(refined_materials, dtype=np.int64),
        macro_parent=np.asarray(refined_parent, dtype=np.int64),
        macro_parent_mesh_name=str(macro.mesh_name),
    )


def build_same_mesh_tetra_connectivity(
    macro_nodes: np.ndarray,
    macro_tets: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[float, float, float], int]]:
    degree = int(degree)
    macro_nodes = np.asarray(macro_nodes, dtype=np.float64)
    macro_tets = np.asarray(macro_tets, dtype=np.int64)
    if degree < 1:
        raise ValueError("degree must be >= 1")

    ref = tetra_reference_nodes(degree).T
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_out: list[np.ndarray] = []
    elems_out = np.empty((macro_tets.shape[0], ref.shape[0]), dtype=np.int64)

    for e, tet in enumerate(macro_tets):
        verts = macro_nodes[np.asarray(tet, dtype=np.int64)]
        for a, xi in enumerate(ref):
            l0 = 1.0 - xi[0] - xi[1] - xi[2]
            x = (
                l0 * verts[0]
                + xi[0] * verts[1]
                + xi[1] * verts[2]
                + xi[2] * verts[3]
            )
            key = _point_key(x)
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_out)
                node_map[key] = idx
                nodes_out.append(np.asarray(x, dtype=np.float64))
            elems_out[e, a] = int(idx)

    return np.asarray(nodes_out, dtype=np.float64), elems_out, node_map


def _midpoint_node_index(
    coord: np.ndarray,
    edge_map: dict[tuple[int, int], int],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
) -> int:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    idx = edge_map.get(key)
    if idx is not None:
        return int(idx)
    idx = int(coord.shape[1] + len(extra_points))
    edge_map[key] = idx
    extra_points.append(0.5 * (coord[:, key[0]] + coord[:, key[1]]))
    return idx


def _edge_lagrange_node_indices(
    coord: np.ndarray,
    edge_map: dict[tuple[int, int], tuple[int, ...]],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
    *,
    order: int,
) -> tuple[int, ...]:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    stored = edge_map.get(key)
    if stored is None:
        lo, hi = key
        ids: list[int] = []
        for step in range(1, int(order)):
            idx = int(coord.shape[1] + len(extra_points))
            alpha = float(int(order) - step) / float(order)
            beta = float(step) / float(order)
            extra_points.append(alpha * coord[:, lo] + beta * coord[:, hi])
            ids.append(idx)
        stored = tuple(ids)
        edge_map[key] = stored
    if (i, j) == key:
        return stored
    return tuple(reversed(stored))


def _face_interior_node_indices(
    coord: np.ndarray,
    face_map: dict[tuple[int, int, int], dict[tuple[int, int, int], int]],
    extra_points: list[np.ndarray],
    verts: tuple[int, int, int],
    *,
    order: int,
) -> tuple[int, ...]:
    local_verts = tuple(int(v) for v in verts)
    canonical = tuple(sorted(local_verts))
    stored = face_map.get(canonical)
    if stored is None:
        stored = {}
        for tri_counts in triangle_lagrange_interior_tuples(int(order)):
            point = np.zeros(coord.shape[0], dtype=np.float64)
            for count, node in zip(tri_counts, canonical, strict=False):
                point += (float(count) / float(order)) * coord[:, int(node)]
            idx = int(coord.shape[1] + len(extra_points))
            extra_points.append(point)
            stored[tuple(int(v) for v in tri_counts)] = idx
        face_map[canonical] = stored

    local_to_canonical = [canonical.index(v) for v in local_verts]
    out: list[int] = []
    for tri_counts in triangle_lagrange_interior_tuples(int(order)):
        canonical_counts = [0, 0, 0]
        for local_idx, canonical_idx in enumerate(local_to_canonical):
            canonical_counts[canonical_idx] = int(tri_counts[local_idx])
        out.append(int(stored[tuple(canonical_counts)]))
    return tuple(out)


def _elevate_macro_mesh_to_degree(
    macro_nodes: np.ndarray,
    macro_tets: np.ndarray,
    macro_surf: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = int(degree)
    coord = np.asarray(macro_nodes, dtype=np.float64).T
    tet4 = np.asarray(macro_tets, dtype=np.int64).T
    tri3 = np.asarray(macro_surf, dtype=np.int64).T

    if degree == 1:
        return np.asarray(coord.T, dtype=np.float64), np.asarray(tet4.T, dtype=np.int64), np.asarray(tri3.T, dtype=np.int64)

    if degree == 2:
        edge_map: dict[tuple[int, int], int] = {}
        extra_points: list[np.ndarray] = []

        tet10 = np.empty((10, tet4.shape[1]), dtype=np.int64)
        tet10[:4, :] = tet4
        for idx in range(tet4.shape[1]):
            v0, v1, v2, v3 = (int(v) for v in tet4[:, idx])
            tet10[4, idx] = _midpoint_node_index(coord, edge_map, extra_points, v0, v1)
            tet10[5, idx] = _midpoint_node_index(coord, edge_map, extra_points, v1, v2)
            tet10[6, idx] = _midpoint_node_index(coord, edge_map, extra_points, v0, v2)
            tet10[7, idx] = _midpoint_node_index(coord, edge_map, extra_points, v1, v3)
            tet10[8, idx] = _midpoint_node_index(coord, edge_map, extra_points, v2, v3)
            tet10[9, idx] = _midpoint_node_index(coord, edge_map, extra_points, v0, v3)

        tri6 = np.empty((6, tri3.shape[1]), dtype=np.int64)
        if tri3.shape[1]:
            tri6[:3, :] = tri3
            for idx in range(tri3.shape[1]):
                v0, v1, v2 = (int(v) for v in tri3[:, idx])
                tri6[3, idx] = _midpoint_node_index(coord, edge_map, extra_points, v0, v1)
                tri6[4, idx] = _midpoint_node_index(coord, edge_map, extra_points, v1, v2)
                tri6[5, idx] = _midpoint_node_index(coord, edge_map, extra_points, v0, v2)

        coord_new = (
            np.hstack((coord, np.column_stack(extra_points)))
            if extra_points
            else coord.copy()
        )
        return (
            np.asarray(coord_new.T, dtype=np.float64),
            np.asarray(tet10.T, dtype=np.int64),
            np.asarray(tri6.T, dtype=np.int64),
        )

    if degree == 4:
        edge_map: dict[tuple[int, int], tuple[int, ...]] = {}
        face_map: dict[tuple[int, int, int], dict[tuple[int, int, int], int]] = {}
        extra_points: list[np.ndarray] = []

        tet35 = np.empty((35, tet4.shape[1]), dtype=np.int64)
        tet35[:4, :] = tet4
        for idx in range(tet4.shape[1]):
            v0, v1, v2, v3 = (int(v) for v in tet4[:, idx])

            tet35[4:7, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v0, v1, order=4)
            tet35[7:10, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v1, v2, order=4)
            tet35[10:13, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v0, v2, order=4)
            tet35[13:16, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v1, v3, order=4)
            tet35[16:19, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v2, v3, order=4)
            tet35[19:22, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v0, v3, order=4)

            faces = (
                (v0, v1, v2),
                (v0, v1, v3),
                (v0, v2, v3),
                (v1, v2, v3),
            )
            cursor = 22
            for face in faces:
                interior = _face_interior_node_indices(
                    coord,
                    face_map,
                    extra_points,
                    face,
                    order=4,
                )
                tet35[cursor : cursor + len(interior), idx] = interior
                cursor += len(interior)

            centroid_idx = int(coord.shape[1] + len(extra_points))
            extra_points.append(
                0.25
                * (
                    coord[:, v0]
                    + coord[:, v1]
                    + coord[:, v2]
                    + coord[:, v3]
                )
            )
            tet35[34, idx] = centroid_idx

        tri15 = np.empty((15, tri3.shape[1]), dtype=np.int64)
        if tri3.shape[1]:
            tri15[:3, :] = tri3
            for idx in range(tri3.shape[1]):
                v0, v1, v2 = (int(v) for v in tri3[:, idx])
                tri15[3:6, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v0, v1, order=4)
                tri15[6:9, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v1, v2, order=4)
                tri15[9:12, idx] = _edge_lagrange_node_indices(coord, edge_map, extra_points, v0, v2, order=4)
                tri15[12:15, idx] = _face_interior_node_indices(
                    coord,
                    face_map,
                    extra_points,
                    (v0, v1, v2),
                    order=4,
                )

        coord_new = (
            np.hstack((coord, np.column_stack(extra_points)))
            if extra_points
            else coord.copy()
        )
        return (
            np.asarray(coord_new.T, dtype=np.float64),
            np.asarray(tet35.T, dtype=np.int64),
            np.asarray(tri15.T, dtype=np.int64),
        )

    raise ValueError(f"Unsupported degree {degree!r}; expected 1, 2, or 4")


def _lift_boundary_faces(
    macro_nodes: np.ndarray,
    macro_surf: np.ndarray,
    *,
    degree: int,
    node_lookup: dict[tuple[float, float, float], int],
) -> np.ndarray:
    degree = int(degree)
    tuples = triangle_lagrange_node_tuples(degree)
    surf = np.empty((macro_surf.shape[0], len(tuples)), dtype=np.int64)
    for face_idx, face in enumerate(np.asarray(macro_surf, dtype=np.int64)):
        verts = np.asarray(macro_nodes[np.asarray(face, dtype=np.int64)], dtype=np.float64)
        for a, counts in enumerate(tuples):
            point = (
                (float(counts[0]) / float(degree)) * verts[0]
                + (float(counts[1]) / float(degree)) * verts[1]
                + (float(counts[2]) / float(degree)) * verts[2]
            )
            key = _point_key(point)
            try:
                surf[face_idx, a] = int(node_lookup[key])
            except KeyError as exc:
                raise KeyError(
                    "Boundary face lifting could not locate degree-aware face node "
                    f"{point.tolist()} in the same-mesh node map"
                ) from exc
    return surf


def expand_tetra_connectivity_to_dofs(elems_scalar: np.ndarray) -> np.ndarray:
    elems_scalar = np.asarray(elems_scalar, dtype=np.int64)
    n_elem, n_p = elems_scalar.shape
    elems = np.empty((n_elem, 3 * n_p), dtype=np.int64)
    elems[:, 0::3] = 3 * elems_scalar
    elems[:, 1::3] = 3 * elems_scalar + 1
    elems[:, 2::3] = 3 * elems_scalar + 2
    return elems


def ownership_block_size_3d(freedofs: np.ndarray) -> int:
    freedofs_arr = np.asarray(freedofs, dtype=np.int64).ravel()
    if freedofs_arr.size == 0:
        return 1
    counts = np.bincount(freedofs_arr // 3)
    if counts.size and np.all(counts[np.nonzero(counts)] == 3):
        return 3
    return 1


def select_reordered_perm_3d(
    reorder_mode: str,
    *,
    adjacency: sp.spmatrix | None,
    coords_all: np.ndarray,
    freedofs: np.ndarray,
    n_parts: int,
) -> np.ndarray:
    mode = str(reorder_mode)
    freedofs_arr = np.asarray(freedofs, dtype=np.int64).ravel()
    if mode == "block_xyz":
        node_ids = freedofs_arr // 3
        comps = freedofs_arr % 3
        coords = np.asarray(coords_all[node_ids], dtype=np.float64)
        order = np.lexsort(
            (
                comps,
                coords[:, 2],
                coords[:, 1],
                coords[:, 0],
            )
        )
        return np.asarray(order, dtype=np.int64)

    block_size = ownership_block_size_3d(freedofs_arr)
    return select_permutation(
        mode,
        adjacency=adjacency,
        coords_all=np.asarray(coords_all, dtype=np.float64),
        freedofs=freedofs_arr,
        n_parts=int(n_parts),
        block_size=int(block_size),
    )


def _build_q_mask(
    nodes: np.ndarray,
    n_nodes: int,
    surf: np.ndarray,
    boundary_label: np.ndarray,
    *,
    constraint_variant: str | None = None,
) -> np.ndarray:
    variant = normalize_constraint_variant(constraint_variant)
    q_mask = np.ones((int(n_nodes), 3), dtype=bool)
    labels = _definition_dirichlet_labels()
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        constrained = tuple(int(v) for v in labels[axis_name])
        if not constrained:
            continue
        mask = np.isin(np.asarray(boundary_label, dtype=np.int64), np.asarray(constrained, dtype=np.int64))
        if np.any(mask):
            q_mask[np.asarray(surf[mask], dtype=np.int64).ravel(), axis_idx] = False
    if variant == PLASTICITY3D_CONSTRAINT_VARIANT_GLUED_BOTTOM:
        node_coords = np.asarray(nodes, dtype=np.float64).reshape((int(n_nodes), 3))
        glued = np.abs(node_coords[:, 1]) <= 1.0e-12
        if np.any(glued):
            q_mask[glued, :] = False
    return q_mask


def _build_free_dofs(q_mask: np.ndarray) -> np.ndarray:
    q_mask = np.asarray(q_mask, dtype=bool)
    dof_ids = np.arange(3 * q_mask.shape[0], dtype=np.int64).reshape(q_mask.shape[0], 3)
    return dof_ids[q_mask].astype(np.int64)


def _build_dof_adjacency(elems: np.ndarray, freedofs: np.ndarray) -> sp.coo_matrix:
    n_free = int(freedofs.size)
    if n_free == 0:
        return sp.coo_matrix((0, 0), dtype=np.float64)
    full_to_free = np.full(int(np.max(elems)) + 1, -1, dtype=np.int64)
    full_to_free[freedofs] = np.arange(n_free, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for elem_dofs in np.asarray(elems, dtype=np.int64):
        local = full_to_free[elem_dofs]
        local = local[local >= 0]
        if local.size == 0:
            continue
        rows.append(np.repeat(local, local.size))
        cols.append(np.tile(local, local.size))

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.ones(row.size, dtype=np.float64)
    else:
        row = np.empty(0, dtype=np.int64)
        col = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float64)
    adjacency = sp.coo_matrix((data, (row, col)), shape=(n_free, n_free))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency


def build_near_nullspace_modes_3d(nodes: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    freedofs = np.asarray(freedofs, dtype=np.int64)
    n_nodes = int(nodes.shape[0])

    full = np.zeros((3 * n_nodes, 6), dtype=np.float64)
    center = np.mean(nodes, axis=0)
    x = nodes[:, 0] - center[0]
    y = nodes[:, 1] - center[1]
    z = nodes[:, 2] - center[2]

    full[0::3, 0] = 1.0
    full[1::3, 1] = 1.0
    full[2::3, 2] = 1.0

    full[1::3, 3] = -z
    full[2::3, 3] = y

    full[0::3, 4] = z
    full[2::3, 4] = -x

    full[0::3, 5] = -y
    full[1::3, 5] = x
    return full[freedofs, :]


def _quadrature_volume_3d(degree: int) -> tuple[np.ndarray, np.ndarray]:
    degree = int(degree)
    if degree == 1:
        xi = np.array([[1.0 / 4.0], [1.0 / 4.0], [1.0 / 4.0]], dtype=np.float64)
        wf = np.array([1.0 / 6.0], dtype=np.float64)
        return xi, wf
    if degree == 2:
        xi = np.array(
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
        wf = np.array(
            [
                -0.013155555555555,
                0.007622222222222,
                0.007622222222222,
                0.007622222222222,
                0.007622222222222,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
            ],
            dtype=np.float64,
        )
        return xi, wf
    if degree == 4:
        xi = np.array(
            [
                [
                    0.3561913862225449,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.8779781243961660,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0329863295731731,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                ],
                [
                    0.2146028712591517,
                    0.2146028712591517,
                    0.2146028712591517,
                    0.3561913862225449,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.8779781243961660,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.0329863295731731,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                ],
                [
                    0.2146028712591517,
                    0.2146028712591517,
                    0.3561913862225449,
                    0.2146028712591517,
                    0.0406739585346113,
                    0.0406739585346113,
                    0.8779781243961660,
                    0.0406739585346113,
                    0.3223378901422757,
                    0.3223378901422757,
                    0.0329863295731731,
                    0.3223378901422757,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.0636610018750175,
                    0.0636610018750175,
                    0.6030056647916491,
                    0.6030056647916491,
                    0.0636610018750175,
                    0.2696723314583159,
                    0.2696723314583159,
                    0.6030056647916491,
                    0.0636610018750175,
                ],
            ],
            dtype=np.float64,
        )
        wf = np.array(
            [
                0.0399227502581679,
                0.0399227502581679,
                0.0399227502581679,
                0.0399227502581679,
                0.0100772110553207,
                0.0100772110553207,
                0.0100772110553207,
                0.0100772110553207,
                0.0553571815436544,
                0.0553571815436544,
                0.0553571815436544,
                0.0553571815436544,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
                0.0482142857142857,
            ],
            dtype=np.float64,
        ) / 6.0
        return xi, wf
    raise ValueError(f"Unsupported tetra degree {degree!r}; expected 1, 2, or 4")


def _assemble_local_tet_ops(
    nodes: np.ndarray,
    elems_scalar: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi, wf = _quadrature_volume_3d(int(degree))
    hatp, dhat1, dhat2, dhat3 = evaluate_tetra_lagrange_basis(int(degree), xi)
    elem_coords = np.asarray(nodes[np.asarray(elems_scalar, dtype=np.int64)], dtype=np.float64)
    n_elem = int(elems_scalar.shape[0])
    n_q = int(xi.shape[1])
    n_p = int(elems_scalar.shape[1])

    dphix = np.empty((n_elem, n_q, n_p), dtype=np.float64)
    dphiy = np.empty((n_elem, n_q, n_p), dtype=np.float64)
    dphiz = np.empty((n_elem, n_q, n_p), dtype=np.float64)
    quad_weight = np.empty((n_elem, n_q), dtype=np.float64)

    xcoord = elem_coords[:, :, 0]
    ycoord = elem_coords[:, :, 1]
    zcoord = elem_coords[:, :, 2]
    for q in range(n_q):
        dh1 = np.asarray(dhat1[:, q], dtype=np.float64)
        dh2 = np.asarray(dhat2[:, q], dtype=np.float64)
        dh3 = np.asarray(dhat3[:, q], dtype=np.float64)

        j11 = xcoord @ dh1
        j12 = ycoord @ dh1
        j13 = zcoord @ dh1
        j21 = xcoord @ dh2
        j22 = ycoord @ dh2
        j23 = zcoord @ dh2
        j31 = xcoord @ dh3
        j32 = ycoord @ dh3
        j33 = zcoord @ dh3

        det_j = (
            j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31)
        )
        inv_det = 1.0 / det_j

        dphix[:, q, :] = (
            ((j22 * j33 - j23 * j32)[:, None] * dh1[None, :])
            - ((j12 * j33 - j13 * j32)[:, None] * dh2[None, :])
            + ((j12 * j23 - j13 * j22)[:, None] * dh3[None, :])
        ) * inv_det[:, None]
        dphiy[:, q, :] = (
            (-(j21 * j33 - j23 * j31)[:, None] * dh1[None, :])
            + ((j11 * j33 - j13 * j31)[:, None] * dh2[None, :])
            - ((j11 * j23 - j13 * j21)[:, None] * dh3[None, :])
        ) * inv_det[:, None]
        dphiz[:, q, :] = (
            ((j21 * j32 - j22 * j31)[:, None] * dh1[None, :])
            - ((j11 * j32 - j12 * j31)[:, None] * dh2[None, :])
            + ((j11 * j22 - j12 * j21)[:, None] * dh3[None, :])
        ) * inv_det[:, None]
        quad_weight[:, q] = np.abs(det_j) * float(wf[q])

    return dphix, dphiy, dphiz, quad_weight, hatp


def _assemble_gravity_load(
    elems_scalar: np.ndarray,
    quad_weight: np.ndarray,
    hatp: np.ndarray,
    gamma_q: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    force = np.zeros(3 * int(n_nodes), dtype=np.float64)
    local_y = -np.einsum(
        "eq,aq,eq->ea",
        np.asarray(quad_weight, dtype=np.float64),
        np.asarray(hatp, dtype=np.float64),
        np.asarray(gamma_q, dtype=np.float64),
    )
    dofs_y = 3 * np.asarray(elems_scalar, dtype=np.int64) + 1
    np.add.at(force, dofs_y.ravel(), local_y.ravel())
    return force


def _macro_mesh_for_case(mesh_path: str | Path, *, mesh_name: str) -> _MacroMeshData:
    base_mesh_name = base_mesh_name_for_name(str(mesh_name))
    macro = _load_macro_mesh_from_msh(mesh_path, base_mesh_name)
    refinement_steps = int(uniform_refinement_steps_for_name(str(mesh_name)))
    for step in range(1, refinement_steps + 1):
        refined_mesh_name = mesh_name_with_uniform_refinements(base_mesh_name, step)
        macro = _uniform_refine_macro_mesh_once(macro, mesh_name=str(refined_mesh_name))
    if str(macro.mesh_name) != str(mesh_name):
        macro = _MacroMeshData(
            mesh_name=str(mesh_name),
            raw_mesh_filename=str(macro.raw_mesh_filename),
            nodes=np.asarray(macro.nodes, dtype=np.float64),
            elems=np.asarray(macro.elems, dtype=np.int64),
            surf=np.asarray(macro.surf, dtype=np.int64),
            boundary_label=np.asarray(macro.boundary_label, dtype=np.int64),
            material_id=np.asarray(macro.material_id, dtype=np.int64),
            macro_parent=np.asarray(macro.macro_parent, dtype=np.int64),
            macro_parent_mesh_name=str(macro.macro_parent_mesh_name),
        )
    return macro


def build_case_data_from_raw_mesh(
    mesh_path: str | Path,
    *,
    mesh_name: str,
    degree: int,
    constraint_variant: str | None = None,
) -> SlopeStability3DCaseData:
    degree = int(degree)
    if degree not in {1, 2, 4}:
        raise ValueError(f"Unsupported degree {degree!r}; expected 1, 2, or 4")
    variant = normalize_constraint_variant(constraint_variant)

    macro = _macro_mesh_for_case(mesh_path, mesh_name=str(mesh_name))
    nodes, elems_scalar, surf = _elevate_macro_mesh_to_degree(
        macro.nodes,
        macro.elems,
        macro.surf,
        degree=degree,
    )
    q_mask = _build_q_mask(
        nodes,
        nodes.shape[0],
        surf,
        macro.boundary_label,
        constraint_variant=variant,
    )
    freedofs = _build_free_dofs(q_mask)
    elems = expand_tetra_connectivity_to_dofs(elems_scalar)
    dphix, dphiy, dphiz, quad_weight, hatp = _assemble_local_tet_ops(
        nodes,
        elems_scalar,
        degree=degree,
    )

    materials = _definition_materials()
    c0_q, phi_q, psi_q, shear_q, bulk_q, lame_q, gamma_q = heterogenous_materials_qp(
        macro.material_id,
        n_q=int(quad_weight.shape[1]),
        materials=materials,
    )
    force = _assemble_gravity_load(
        elems_scalar,
        quad_weight,
        hatp,
        gamma_q,
        n_nodes=int(nodes.shape[0]),
    )
    build_adjacency = int(elems.shape[0]) <= 50000
    adjacency = _build_dof_adjacency(elems, freedofs) if build_adjacency else None
    elastic_kernel = build_near_nullspace_modes_3d(nodes, freedofs)

    return SlopeStability3DCaseData(
        case_name=same_mesh_case_name(mesh_name, degree, variant),
        mesh_name=str(mesh_name),
        degree=int(degree),
        raw_mesh_filename=str(Path(mesh_path).name),
        constraint_variant=str(variant),
        nodes=np.asarray(nodes, dtype=np.float64),
        elems_scalar=np.asarray(elems_scalar, dtype=np.int64),
        elems=np.asarray(elems, dtype=np.int64),
        surf=np.asarray(surf, dtype=np.int64),
        boundary_label=np.asarray(macro.boundary_label, dtype=np.int64),
        q_mask=np.asarray(q_mask, dtype=bool),
        freedofs=np.asarray(freedofs, dtype=np.int64),
        dphix=np.asarray(dphix, dtype=np.float64),
        dphiy=np.asarray(dphiy, dtype=np.float64),
        dphiz=np.asarray(dphiz, dtype=np.float64),
        quad_weight=np.asarray(quad_weight, dtype=np.float64),
        force=np.asarray(force, dtype=np.float64),
        u_0=np.zeros(3 * nodes.shape[0], dtype=np.float64),
        material_id=np.asarray(macro.material_id, dtype=np.int64),
        c0_q=np.asarray(c0_q, dtype=np.float64),
        phi_q=np.asarray(phi_q, dtype=np.float64),
        psi_q=np.asarray(psi_q, dtype=np.float64),
        shear_q=np.asarray(shear_q, dtype=np.float64),
        bulk_q=np.asarray(bulk_q, dtype=np.float64),
        lame_q=np.asarray(lame_q, dtype=np.float64),
        gamma_q=np.asarray(gamma_q, dtype=np.float64),
        eps_p_old=np.zeros(
            (elems_scalar.shape[0], quad_weight.shape[1], 6),
            dtype=np.float64,
        ),
        adjacency=adjacency,
        elastic_kernel=np.asarray(elastic_kernel, dtype=np.float64),
        macro_parent=np.asarray(macro.macro_parent, dtype=np.int64),
        macro_parent_mesh_name=str(macro.macro_parent_mesh_name),
        davis_type="B",
        lambda_target_default=1.0,
        gravity_axis=1,
    )


def _broadcast_case_data(factory, *, build_mode: str, comm: MPI.Comm | None):
    build_mode = str(build_mode)
    if build_mode == "replicated" or comm is None or int(comm.size) <= 1:
        return factory()
    if build_mode == "root_bcast":
        payload = factory() if int(comm.rank) == 0 else None
        return comm.bcast(payload, root=0)
    if build_mode == "rank_local":
        return factory()
    raise ValueError(f"Unsupported build_mode {build_mode!r}")


def build_same_mesh_lagrange_case_data(
    mesh_name: str = DEFAULT_MESH_NAME,
    *,
    degree: int,
    constraint_variant: str | None = None,
    build_mode: str = "replicated",
    comm: MPI.Comm | None = None,
) -> SlopeStability3DCaseData:
    mesh_name = str(mesh_name)
    degree = int(degree)
    variant = normalize_constraint_variant(constraint_variant)
    hdf5_path = ensure_same_mesh_case_hdf5(mesh_name, degree, constraint_variant=variant)
    return _broadcast_case_data(
        lambda: (
            load_case_hdf5(hdf5_path)
            if hdf5_path.exists()
            else build_case_data_from_raw_mesh(
                raw_mesh_path_for_name(mesh_name),
                mesh_name=mesh_name,
                degree=degree,
                constraint_variant=variant,
            )
        ),
        build_mode=build_mode,
        comm=comm,
    )


def write_case_hdf5(path: str | Path, case_data: SlopeStability3DCaseData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adjacency = None if case_data.adjacency is None else case_data.adjacency.tocoo()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("case_name", data=np.bytes_(case_data.case_name))
        handle.create_dataset("mesh_name", data=np.bytes_(case_data.mesh_name))
        handle.create_dataset("raw_mesh_filename", data=np.bytes_(case_data.raw_mesh_filename))
        handle.create_dataset("constraint_variant", data=np.bytes_(case_data.constraint_variant))
        handle.create_dataset("schema_version", data=int(SAME_MESH_HDF5_SCHEMA_VERSION))
        handle.create_dataset("source_internal_axis_order", data=np.bytes_(SOURCE_INTERNAL_AXIS_ORDER))
        handle.create_dataset("degree", data=int(case_data.degree))
        handle.create_dataset("nodes", data=case_data.nodes)
        handle.create_dataset("elems_scalar", data=case_data.elems_scalar)
        handle.create_dataset("elems", data=case_data.elems)
        handle.create_dataset("surf", data=case_data.surf)
        handle.create_dataset("boundary_label", data=case_data.boundary_label)
        handle.create_dataset("q_mask", data=case_data.q_mask.astype(np.uint8))
        handle.create_dataset("freedofs", data=case_data.freedofs)
        handle.create_dataset("dphix", data=case_data.dphix)
        handle.create_dataset("dphiy", data=case_data.dphiy)
        handle.create_dataset("dphiz", data=case_data.dphiz)
        handle.create_dataset("quad_weight", data=case_data.quad_weight)
        handle.create_dataset("force", data=case_data.force)
        handle.create_dataset("u_0", data=case_data.u_0)
        handle.create_dataset("material_id", data=case_data.material_id)
        handle.create_dataset("c0_q", data=case_data.c0_q)
        handle.create_dataset("phi_q", data=case_data.phi_q)
        handle.create_dataset("psi_q", data=case_data.psi_q)
        handle.create_dataset("shear_q", data=case_data.shear_q)
        handle.create_dataset("bulk_q", data=case_data.bulk_q)
        handle.create_dataset("lame_q", data=case_data.lame_q)
        handle.create_dataset("gamma_q", data=case_data.gamma_q)
        handle.create_dataset("eps_p_old", data=case_data.eps_p_old)
        handle.create_dataset("elastic_kernel", data=case_data.elastic_kernel)
        handle.create_dataset("macro_parent", data=case_data.macro_parent)
        handle.create_dataset(
            "macro_parent_mesh_name",
            data=np.bytes_(case_data.macro_parent_mesh_name),
        )
        handle.create_dataset("davis_type", data=np.bytes_(case_data.davis_type))
        handle.create_dataset(
            "lambda_target_default",
            data=float(case_data.lambda_target_default),
        )
        handle.create_dataset("gravity_axis", data=int(case_data.gravity_axis))
        if adjacency is not None:
            grp = handle.create_group("adjacency")
            grp.create_dataset("data", data=adjacency.data)
            grp.create_dataset("row", data=adjacency.row)
            grp.create_dataset("col", data=adjacency.col)
            grp.create_dataset("shape", data=np.asarray(adjacency.shape, dtype=np.int64))


def load_case_hdf5(path: str | Path) -> SlopeStability3DCaseData:
    raw, adjacency = load_problem_hdf5(str(path))
    if adjacency is None:
        adjacency = _build_dof_adjacency(
            np.asarray(raw["elems"], dtype=np.int64),
            np.asarray(raw["freedofs"], dtype=np.int64),
        )
    return SlopeStability3DCaseData(
        case_name=_decode_hdf5_string(raw["case_name"]),
        mesh_name=_decode_hdf5_string(raw["mesh_name"]),
        degree=int(raw["degree"]),
        raw_mesh_filename=_decode_hdf5_string(raw["raw_mesh_filename"]),
        constraint_variant=_decode_hdf5_string(
            raw.get("constraint_variant", np.bytes_(PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM))
        ),
        nodes=np.asarray(raw["nodes"], dtype=np.float64),
        elems_scalar=np.asarray(raw["elems_scalar"], dtype=np.int64),
        elems=np.asarray(raw["elems"], dtype=np.int64),
        surf=np.asarray(raw["surf"], dtype=np.int64),
        boundary_label=np.asarray(raw["boundary_label"], dtype=np.int64),
        q_mask=np.asarray(raw["q_mask"], dtype=bool),
        freedofs=np.asarray(raw["freedofs"], dtype=np.int64),
        dphix=np.asarray(raw["dphix"], dtype=np.float64),
        dphiy=np.asarray(raw["dphiy"], dtype=np.float64),
        dphiz=np.asarray(raw["dphiz"], dtype=np.float64),
        quad_weight=np.asarray(raw["quad_weight"], dtype=np.float64),
        force=np.asarray(raw["force"], dtype=np.float64),
        u_0=np.asarray(raw["u_0"], dtype=np.float64),
        material_id=np.asarray(raw["material_id"], dtype=np.int64),
        c0_q=np.asarray(raw["c0_q"], dtype=np.float64),
        phi_q=np.asarray(raw["phi_q"], dtype=np.float64),
        psi_q=np.asarray(raw["psi_q"], dtype=np.float64),
        shear_q=np.asarray(raw["shear_q"], dtype=np.float64),
        bulk_q=np.asarray(raw["bulk_q"], dtype=np.float64),
        lame_q=np.asarray(raw["lame_q"], dtype=np.float64),
        gamma_q=np.asarray(raw["gamma_q"], dtype=np.float64),
        eps_p_old=np.asarray(raw["eps_p_old"], dtype=np.float64),
        adjacency=adjacency,
        elastic_kernel=np.asarray(raw["elastic_kernel"], dtype=np.float64),
        macro_parent=np.asarray(raw["macro_parent"], dtype=np.int64),
        macro_parent_mesh_name=_decode_hdf5_string(raw["macro_parent_mesh_name"]),
        davis_type=_decode_hdf5_string(raw["davis_type"]),
        lambda_target_default=float(raw["lambda_target_default"]),
        gravity_axis=int(raw["gravity_axis"]),
    )


def load_case_hdf5_fields(
    path: str | Path,
    *,
    fields: list[str] | tuple[str, ...] | set[str],
    load_adjacency: bool = False,
) -> tuple[dict[str, object], sp.coo_matrix | None]:
    raw, adjacency = load_problem_hdf5_fields(
        str(path),
        fields=fields,
        load_adjacency=bool(load_adjacency),
    )
    for key in (
        "case_name",
        "mesh_name",
        "raw_mesh_filename",
        "constraint_variant",
        "macro_parent_mesh_name",
        "davis_type",
    ):
        if key in raw:
            raw[key] = _decode_hdf5_string(raw[key])
    return raw, adjacency


_SAME_MESH_LIGHT_FIELDS = (
    "case_name",
    "mesh_name",
    "raw_mesh_filename",
    "constraint_variant",
    "degree",
    "nodes",
    "elems_scalar",
    "elems",
    "surf",
    "boundary_label",
    "q_mask",
    "freedofs",
    "force",
    "u_0",
    "material_id",
    "elastic_kernel",
    "macro_parent",
    "macro_parent_mesh_name",
    "davis_type",
    "lambda_target_default",
    "gravity_axis",
)


def load_same_mesh_case_hdf5_light(
    mesh_name: str,
    degree: int,
    *,
    constraint_variant: str | None = None,
) -> dict[str, object]:
    variant = normalize_constraint_variant(constraint_variant)
    cache_key = (str(mesh_name), int(degree), str(variant))
    cached = _LIGHT_CASE_CACHE.get(cache_key)
    if cached is not None:
        raw = dict(cached)
        raw["elem_type"] = f"P{int(degree)}"
        raw["element_degree"] = int(degree)
        return raw
    path = ensure_same_mesh_case_hdf5(str(mesh_name), int(degree), constraint_variant=variant)
    raw, _ = load_case_hdf5_fields(path, fields=_SAME_MESH_LIGHT_FIELDS, load_adjacency=False)
    _LIGHT_CASE_CACHE[cache_key] = dict(raw)
    raw["elem_type"] = f"P{int(degree)}"
    raw["element_degree"] = int(degree)
    return raw


def _build_rank_local_partition_metadata(
    raw: dict[str, object],
    *,
    reorder_mode: str,
    comm: MPI.Comm,
) -> dict[str, object]:
    nodes = np.asarray(raw["nodes"], dtype=np.float64)
    elems = np.asarray(raw["elems"], dtype=np.int64)
    elems_scalar = np.asarray(raw["elems_scalar"], dtype=np.int64)
    freedofs = np.asarray(raw["freedofs"], dtype=np.int64)
    u_0 = np.asarray(raw["u_0"], dtype=np.float64)
    n_free = int(freedofs.size)
    index_dtype = (
        np.int32
        if int(n_free) <= int(np.iinfo(np.int32).max)
        else np.int64
    )

    perm = select_reordered_perm_3d(
        str(reorder_mode),
        adjacency=None,
        coords_all=nodes,
        freedofs=freedofs,
        n_parts=int(comm.size),
    )
    iperm = inverse_permutation(np.asarray(perm, dtype=np.int64))
    ownership_block_size = int(ownership_block_size_3d(freedofs))
    lo, hi = petsc_ownership_range(
        n_free,
        int(comm.rank),
        int(comm.size),
        block_size=ownership_block_size,
    )

    total_to_free_orig = np.full(len(u_0), -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)
    total_to_free_reord = np.full(len(u_0), -1, dtype=np.int64)
    free_mask = total_to_free_orig >= 0
    total_to_free_reord[free_mask] = iperm[total_to_free_orig[free_mask]]
    free_reord_by_node = np.asarray(total_to_free_reord.reshape((-1, 3)), dtype=np.int64)
    owned_node_mask = np.any(
        (free_reord_by_node >= int(lo)) & (free_reord_by_node < int(hi)),
        axis=1,
    )
    local_elem_mask = np.any(owned_node_mask[elems_scalar], axis=1)
    local_elem_idx = np.where(local_elem_mask)[0].astype(np.int64)
    local_elems_total = np.asarray(elems[local_elem_idx], dtype=np.int64)
    local_scalar_total = np.asarray(elems_scalar[local_elem_idx], dtype=np.int64)
    local_elems_reordered = np.asarray(total_to_free_reord[local_elems_total], dtype=np.int64)
    owned_rows, owned_cols = _owned_pattern_from_local_scalar_elems(
        local_scalar_total,
        free_reord_by_node=free_reord_by_node,
        lo=int(lo),
        hi=int(hi),
        n_free=int(n_free),
    )

    masked = np.where(
        local_elems_reordered >= 0,
        local_elems_reordered,
        np.int64(n_free),
    )
    elem_min = np.min(masked, axis=1)
    valid = elem_min < int(n_free)
    local_elem_owner = np.full(len(local_elem_idx), -1, dtype=np.int64)
    if np.any(valid):
        from src.core.petsc.dof_partition import _rank_of_dof_vec

        local_elem_owner[valid] = _rank_of_dof_vec(
            elem_min[valid],
            int(n_free),
            int(comm.size),
            block_size=ownership_block_size,
        )
    local_energy_weights = (local_elem_owner == int(comm.rank)).astype(np.float64)
    local_total_nodes, inverse = np.unique(
        local_elems_total.ravel(),
        return_inverse=True,
    )
    elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

    return {
        "_distributed_perm": np.asarray(perm, dtype=index_dtype),
        "_distributed_iperm": np.asarray(iperm, dtype=index_dtype),
        "_distributed_lo": int(lo),
        "_distributed_hi": int(hi),
        "_distributed_ownership_block_size": int(ownership_block_size),
        "_distributed_total_to_free_reord": np.asarray(total_to_free_reord, dtype=index_dtype),
        "_distributed_local_elem_idx": np.asarray(local_elem_idx, dtype=index_dtype),
        "_distributed_local_elems_total": np.asarray(local_elems_total, dtype=index_dtype),
        "_distributed_local_elems_reordered": np.asarray(local_elems_reordered, dtype=index_dtype),
        "_distributed_owned_rows": np.asarray(owned_rows, dtype=index_dtype),
        "_distributed_owned_cols": np.asarray(owned_cols, dtype=index_dtype),
        "_distributed_local_total_nodes": np.asarray(local_total_nodes, dtype=index_dtype),
        "_distributed_elems_local_np": np.asarray(elems_local_np, dtype=np.int32),
        "_distributed_energy_weights": np.asarray(local_energy_weights, dtype=np.float64),
    }


def load_same_mesh_case_hdf5_rank_local_light(
    mesh_name: str,
    degree: int,
    *,
    constraint_variant: str | None = None,
    reorder_mode: str,
    comm: MPI.Comm,
    block_size: int = 3,
) -> dict[str, object]:
    del block_size
    variant = normalize_constraint_variant(constraint_variant)
    cache_key = (
        str(mesh_name),
        int(degree),
        str(variant),
        str(reorder_mode),
        int(comm.size),
        int(comm.rank),
    )
    cached = _RANK_LOCAL_LIGHT_CACHE.get(cache_key)
    if cached is not None:
        raw = dict(cached)
        raw["elem_type"] = f"P{int(degree)}"
        raw["element_degree"] = int(degree)
        return raw
    raw = load_same_mesh_case_hdf5_light(
        str(mesh_name),
        int(degree),
        constraint_variant=variant,
    )
    raw.update(_build_rank_local_partition_metadata(raw, reorder_mode=str(reorder_mode), comm=comm))
    _RANK_LOCAL_LIGHT_CACHE[cache_key] = dict(raw)
    raw["elem_type"] = f"P{int(degree)}"
    raw["element_degree"] = int(degree)
    return raw


def load_same_mesh_case_hdf5_rank_local(
    mesh_name: str,
    degree: int,
    *,
    constraint_variant: str | None = None,
    reorder_mode: str,
    comm: MPI.Comm,
    block_size: int = 3,
) -> dict[str, object]:
    variant = normalize_constraint_variant(constraint_variant)
    cache_key = (
        str(mesh_name),
        int(degree),
        str(variant),
        str(reorder_mode),
        int(comm.size),
        int(comm.rank),
    )
    cached = _RANK_LOCAL_HEAVY_CACHE.get(cache_key)
    if cached is not None:
        raw = dict(cached)
        raw["elem_type"] = f"P{int(degree)}"
        raw["element_degree"] = int(degree)
        return raw
    path = ensure_same_mesh_case_hdf5(str(mesh_name), int(degree), constraint_variant=variant)
    raw = load_same_mesh_case_hdf5_rank_local_light(
        str(mesh_name),
        int(degree),
        constraint_variant=variant,
        reorder_mode=str(reorder_mode),
        comm=comm,
        block_size=int(block_size),
    )
    local_elem_idx = np.asarray(raw["_distributed_local_elem_idx"], dtype=np.int64)

    heavy_fields = (
        "dphix",
        "dphiy",
        "dphiz",
        "quad_weight",
        "c0_q",
        "phi_q",
        "psi_q",
        "shear_q",
        "bulk_q",
        "lame_q",
        "gamma_q",
        "eps_p_old",
    )
    with h5py.File(path, "r") as handle:
        for key in heavy_fields:
            raw[f"_distributed_{key}"] = np.asarray(handle[key][local_elem_idx], dtype=np.float64)

    raw["elem_type"] = f"P{int(degree)}"
    raw["element_degree"] = int(degree)
    raw["_distributed_local_data"] = True
    _RANK_LOCAL_HEAVY_CACHE[cache_key] = dict(raw)
    return raw


def _materialize_legacy_componentwise_case_if_needed(mesh_name: str, degree: int, out_path: Path) -> bool:
    out_path = Path(out_path)
    if out_path.exists():
        return True
    legacy_path = legacy_unversioned_same_mesh_case_hdf5_path(mesh_name, degree)
    if not legacy_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, out_path)
    with h5py.File(out_path, "r+") as handle:
        for key, value in (
            ("case_name", same_mesh_case_name(mesh_name, degree, PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM)),
            ("constraint_variant", PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM),
            ("schema_version", int(SAME_MESH_HDF5_SCHEMA_VERSION)),
            ("macro_parent_mesh_name", macro_parent_mesh_name_for_name(mesh_name)),
            ("source_internal_axis_order", SOURCE_INTERNAL_AXIS_ORDER),
        ):
            if key in handle:
                del handle[key]
            if isinstance(value, str):
                handle.create_dataset(key, data=np.bytes_(value))
            else:
                handle.create_dataset(key, data=value)
    return True


def ensure_same_mesh_case_hdf5(
    mesh_name: str,
    degree: int,
    *,
    constraint_variant: str | None = None,
) -> Path:
    mesh_name = str(mesh_name)
    degree = int(degree)
    variant = normalize_constraint_variant(constraint_variant)
    path = same_mesh_case_hdf5_path(mesh_name, degree, variant)
    if (
        variant == PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM
        and _materialize_legacy_componentwise_case_if_needed(mesh_name, degree, path)
        and _same_mesh_hdf5_is_current(
            path,
            mesh_name=mesh_name,
            degree=degree,
            constraint_variant=variant,
        )
    ):
        return path
    if not _same_mesh_hdf5_is_current(
        path,
        mesh_name=mesh_name,
        degree=degree,
        constraint_variant=variant,
    ):
        case_data = build_case_data_from_raw_mesh(
            raw_mesh_path_for_name(mesh_name),
            mesh_name=mesh_name,
            degree=degree,
            constraint_variant=variant,
        )
        write_case_hdf5(path, case_data)
    return path


def _same_mesh_hdf5_is_current(
    path: str | Path,
    *,
    mesh_name: str,
    degree: int,
    constraint_variant: str | None = None,
) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    variant = normalize_constraint_variant(constraint_variant)
    try:
        with h5py.File(path, "r") as handle:
            stored_degree = int(handle["degree"][()])
            stored_mesh_name = _decode_hdf5_string(handle["mesh_name"][()])
            stored_raw_mesh = _decode_hdf5_string(handle["raw_mesh_filename"][()])
            schema_obj = handle.get("schema_version")
            axis_obj = handle.get("source_internal_axis_order")
            macro_parent_obj = handle.get("macro_parent_mesh_name")
            stored_schema = int(schema_obj[()]) if schema_obj is not None else 0
            stored_axis_order = (
                _decode_hdf5_string(axis_obj[()]) if axis_obj is not None else ""
            )
            stored_macro_parent_mesh_name = (
                _decode_hdf5_string(macro_parent_obj[()]) if macro_parent_obj is not None else ""
            )
            constraint_obj = handle.get("constraint_variant")
            stored_constraint_variant = (
                _decode_hdf5_string(constraint_obj[()])
                if constraint_obj is not None
                else PLASTICITY3D_CONSTRAINT_VARIANT_COMPONENTWISE_BOTTOM
            )
    except Exception:
        return False
    return (
        stored_degree == int(degree)
        and stored_mesh_name == str(mesh_name)
        and stored_raw_mesh == raw_mesh_filename_for_name(mesh_name)
        and stored_schema == int(SAME_MESH_HDF5_SCHEMA_VERSION)
        and stored_axis_order == SOURCE_INTERNAL_AXIS_ORDER
        and stored_macro_parent_mesh_name == macro_parent_mesh_name_for_name(mesh_name)
        and stored_constraint_variant == str(variant)
    )
