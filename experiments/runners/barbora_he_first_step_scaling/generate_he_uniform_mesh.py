#!/usr/bin/env python3
"""Generate structured HyperElasticity beam HDF5 meshes.

The checked-in HyperElasticity meshes use a regular beam grid with six
tetrahedra per brick.  This helper reproduces that layout so missing refinement
levels can be generated in the same HDF5 schema as levels 1--4.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from src.core.problem_data.hdf5 import mesh_data_path


LENGTH_X = 0.4
HALF_WIDTH = 0.005
C1 = 38461538.461538464
D1 = 83333333.33333333

# Cube node order:
# n000, n100, n010, n110, n001, n101, n011, n111.
TET_TEMPLATE = np.array(
    [
        [0, 1, 2, 5],
        [0, 2, 4, 5],
        [2, 4, 5, 6],
        [1, 3, 2, 5],
        [3, 5, 7, 2],
        [2, 5, 7, 6],
    ],
    dtype=np.int64,
)


def dimensions_for_level(level: int) -> tuple[int, int, int]:
    """Return brick counts in x, y, z for a HyperElasticity level."""
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    nx = 80 * (2 ** (level - 1))
    ny = 2**level
    nz = 2**level
    return nx, ny, nz


def generate_nodes(nx: int, ny: int, nz: int) -> np.ndarray:
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1
    x = np.linspace(0.0, LENGTH_X, nx1, dtype=np.float64)
    y = np.linspace(-HALF_WIDTH, HALF_WIDTH, ny1, dtype=np.float64)
    z = np.linspace(-HALF_WIDTH, HALF_WIDTH, nz1, dtype=np.float64)

    coords = np.empty((nx1 * ny1 * nz1, 3), dtype=np.float64)
    coords[:, 0] = np.tile(x, ny1 * nz1)
    coords[:, 1] = np.tile(np.repeat(y, nx1), nz1)
    coords[:, 2] = np.repeat(z, nx1 * ny1)
    return coords


def generate_elements(nx: int, ny: int, nz: int) -> np.ndarray:
    nx1, ny1 = nx + 1, ny + 1
    ix = np.arange(nx, dtype=np.int64)
    iy = np.arange(ny, dtype=np.int64) * nx1
    iz = np.arange(nz, dtype=np.int64) * nx1 * ny1
    base = (iz[:, None, None] + iy[None, :, None] + ix[None, None, :]).ravel()

    cube = np.empty((base.size, 8), dtype=np.int64)
    cube[:, 0] = base
    cube[:, 1] = base + 1
    cube[:, 2] = base + nx1
    cube[:, 3] = base + nx1 + 1
    cube[:, 4] = base + nx1 * ny1
    cube[:, 5] = base + nx1 * ny1 + 1
    cube[:, 6] = base + nx1 * ny1 + nx1
    cube[:, 7] = base + nx1 * ny1 + nx1 + 1
    return cube[:, TET_TEMPLATE].reshape((-1, 4))


def generate_reference_gradients(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = LENGTH_X / float(nx)
    dy = (2.0 * HALF_WIDTH) / float(ny)
    dz = (2.0 * HALF_WIDTH) / float(nz)
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [dx, 0.0, 0.0],
            [0.0, dy, 0.0],
            [dx, dy, 0.0],
            [0.0, 0.0, dz],
            [dx, 0.0, dz],
            [0.0, dy, dz],
            [dx, dy, dz],
        ],
        dtype=np.float64,
    )

    gradients = np.empty((TET_TEMPLATE.shape[0], 4, 3), dtype=np.float64)
    volumes = np.empty(TET_TEMPLATE.shape[0], dtype=np.float64)
    for idx, tet in enumerate(TET_TEMPLATE):
        vertices = cube[tet]
        matrix = np.ones((4, 4), dtype=np.float64)
        matrix[:, 1:] = vertices
        gradients[idx] = np.linalg.inv(matrix)[1:, :].T
        jac = np.column_stack(
            [vertices[1] - vertices[0], vertices[2] - vertices[0], vertices[3] - vertices[0]]
        )
        volumes[idx] = abs(np.linalg.det(jac)) / 6.0

    gradients[np.abs(gradients) < 1e-12] = 0.0
    return gradients[:, :, 0], gradients[:, :, 1], gradients[:, :, 2], volumes


def generate_element_data(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = nx * ny * nz
    dphix_pattern, dphiy_pattern, dphiz_pattern, vol_pattern = generate_reference_gradients(
        nx, ny, nz
    )
    dphix = np.tile(dphix_pattern, (cells, 1))
    dphiy = np.tile(dphiy_pattern, (cells, 1))
    dphiz = np.tile(dphiz_pattern, (cells, 1))
    vol = np.tile(vol_pattern, cells)
    return dphix, dphiy, dphiz, vol


def generate_free_dofs(nx: int, ny: int, nz: int) -> np.ndarray:
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1
    free_x = np.arange(1, nx, dtype=np.int64)
    yz_planes = np.arange(ny1 * nz1, dtype=np.int64)
    free_nodes = (yz_planes[:, None] * nx1 + free_x[None, :]).ravel()
    offsets = np.arange(3, dtype=np.int64)
    return (3 * free_nodes[:, None] + offsets[None, :]).ravel()


def _compact_free_nodes(elems: np.ndarray, nx: int) -> np.ndarray:
    nx1 = nx + 1
    ix = elems % nx1
    compact = (elems // nx1) * (nx - 1) + (ix - 1)
    compact = compact.astype(np.int32, copy=False)
    compact[(ix == 0) | (ix == nx)] = -1
    return compact


def build_node_adjacency(elems: np.ndarray, nx: int, n_free_nodes: int) -> sp.csr_matrix:
    compact = _compact_free_nodes(elems, nx)
    rows = np.repeat(compact, 4, axis=1).reshape(-1)
    cols = np.tile(compact, (1, 4)).reshape(-1)
    keep = (rows >= 0) & (cols >= 0)
    rows = rows[keep]
    cols = cols[keep]
    data = np.ones(rows.shape[0], dtype=np.uint8)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(n_free_nodes, n_free_nodes)).tocsr()
    adjacency.sum_duplicates()
    adjacency.data = np.ones(adjacency.nnz, dtype=np.uint8)
    adjacency.eliminate_zeros()
    return adjacency


def write_dof_adjacency(group: h5py.Group, node_adjacency: sp.csr_matrix) -> int:
    node_coo = node_adjacency.tocoo()
    nnz = int(node_coo.nnz * 9)
    chunk = min(1_000_000, max(1, nnz))

    row_ds = group.create_dataset(
        "row", shape=(nnz,), dtype=np.int32, compression="gzip", chunks=(chunk,)
    )
    col_ds = group.create_dataset(
        "col", shape=(nnz,), dtype=np.int32, compression="gzip", chunks=(chunk,)
    )
    data_ds = group.create_dataset(
        "data", shape=(nnz,), dtype=np.float64, compression="gzip", chunks=(chunk,)
    )
    group.create_dataset(
        "shape",
        data=np.array([node_adjacency.shape[0] * 3, node_adjacency.shape[1] * 3], dtype=np.int64),
    )

    row_offsets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    col_offsets = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    write_at = 0
    pair_chunk = max(1, chunk // 9)
    for start in range(0, node_coo.nnz, pair_chunk):
        stop = min(start + pair_chunk, node_coo.nnz)
        node_rows = node_coo.row[start:stop].astype(np.int32, copy=False)
        node_cols = node_coo.col[start:stop].astype(np.int32, copy=False)
        rows = (3 * node_rows[:, None] + row_offsets[None, :]).reshape(-1)
        cols = (3 * node_cols[:, None] + col_offsets[None, :]).reshape(-1)
        size = rows.shape[0]
        row_ds[write_at : write_at + size] = rows
        col_ds[write_at : write_at + size] = cols
        data_ds[write_at : write_at + size] = np.ones(size, dtype=np.float64)
        write_at += size

    if write_at != nnz:
        raise RuntimeError(f"wrote {write_at} adjacency entries, expected {nnz}")
    return nnz


def create_dataset(handle: h5py.File, name: str, data: np.ndarray | float) -> None:
    if np.isscalar(data):
        handle.create_dataset(name, data=data)
    else:
        handle.create_dataset(name, data=data, compression="gzip")


def write_mesh(path: Path, level: int, *, force: bool = False) -> dict[str, object]:
    nx, ny, nz = dimensions_for_level(level)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    coords = generate_nodes(nx, ny, nz)
    elems = generate_elements(nx, ny, nz)
    dphix, dphiy, dphiz, vol = generate_element_data(nx, ny, nz)
    u0 = coords.ravel()
    dofs = generate_free_dofs(nx, ny, nz)
    n_free_nodes = dofs.size // 3
    node_adjacency = build_node_adjacency(elems, nx, n_free_nodes)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with h5py.File(tmp_path, "w") as handle:
            create_dataset(handle, "C1", np.float64(C1))
            create_dataset(handle, "D1", np.float64(D1))
            create_dataset(handle, "nodes2coord", coords)
            create_dataset(handle, "elems2nodes", elems)
            create_dataset(handle, "dphix", dphix)
            create_dataset(handle, "dphiy", dphiy)
            create_dataset(handle, "dphiz", dphiz)
            create_dataset(handle, "vol", vol)
            create_dataset(handle, "u0", u0)
            create_dataset(handle, "dofsMinim", dofs)
            adjacency_group = handle.create_group("adjacency")
            adjacency_nnz = write_dof_adjacency(adjacency_group, node_adjacency)
        tmp_path.replace(path)
        path.chmod(0o644)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    return {
        "level": level,
        "path": str(path),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nodes": int(coords.shape[0]),
        "tetrahedra": int(elems.shape[0]),
        "total_dofs": int(u0.size),
        "free_dofs": int(dofs.size),
        "node_adjacency_nnz": int(node_adjacency.nnz),
        "adjacency_nnz": int(adjacency_nnz),
        "file_size_bytes": int(path.stat().st_size),
        "elapsed_s": elapsed,
    }


def validate_level(level: int) -> dict[str, object]:
    path = Path(mesh_data_path("HyperElasticity", f"HyperElasticity_level{level}.h5"))
    nx, ny, nz = dimensions_for_level(level)
    coords = generate_nodes(nx, ny, nz)
    elems = generate_elements(nx, ny, nz)
    dphix, dphiy, dphiz, vol = generate_element_data(nx, ny, nz)
    dofs = generate_free_dofs(nx, ny, nz)
    node_adjacency = build_node_adjacency(elems, nx, dofs.size // 3)

    with h5py.File(path, "r") as handle:
        checks = {
            "nodes2coord": bool(np.allclose(coords, handle["nodes2coord"][:])),
            "elems2nodes": bool(np.array_equal(elems, handle["elems2nodes"][:])),
            "dphix": bool(np.allclose(dphix, handle["dphix"][:], rtol=1e-12, atol=1e-12)),
            "dphiy": bool(np.allclose(dphiy, handle["dphiy"][:], rtol=1e-12, atol=1e-12)),
            "dphiz": bool(np.allclose(dphiz, handle["dphiz"][:], rtol=1e-12, atol=1e-12)),
            "vol": bool(np.allclose(vol, handle["vol"][:], rtol=1e-12, atol=1e-18)),
            "u0": bool(np.allclose(coords.ravel(), handle["u0"][:])),
            "dofsMinim": bool(np.array_equal(dofs, handle["dofsMinim"][:])),
            "adjacency_nnz": bool(node_adjacency.nnz * 9 == handle["adjacency/data"].shape[0]),
            "adjacency_shape": bool(
                list(handle["adjacency/shape"][:]) == [int(dofs.size), int(dofs.size)]
            ),
            "C1": bool(np.isclose(float(handle["C1"][()]), C1)),
            "D1": bool(np.isclose(float(handle["D1"][()]), D1)),
        }
    ok = all(checks.values())
    return {
        "level": level,
        "path": str(path),
        "ok": ok,
        "checks": checks,
        "node_adjacency_nnz": int(node_adjacency.nnz),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HDF5 path. Defaults to data/meshes/HyperElasticity/HyperElasticity_level<level>.h5.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate generated levels 1--4 against checked-in HDF5 files before writing.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run existing-level validation and do not write an output mesh.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path for generation metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"validation": []}

    if args.validate_existing or args.validate_only:
        validation = [validate_level(level) for level in range(1, 5)]
        payload["validation"] = validation
        failed = [entry for entry in validation if not entry["ok"]]
        if failed:
            print(json.dumps(payload, indent=2))
            raise SystemExit("existing-level validation failed")
        if args.validate_only:
            print(json.dumps(payload, indent=2))
            return

    out_path = args.out or Path(
        mesh_data_path("HyperElasticity", f"HyperElasticity_level{args.level}.h5")
    )
    payload["generated"] = write_mesh(out_path, args.level, force=args.force)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
