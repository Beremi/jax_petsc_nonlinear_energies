"""Shared HDF5 mesh/problem-data loading helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[3]
MESH_DATA_ROOT = REPO_ROOT / "data" / "meshes"


def mesh_data_path(*relative_parts: str) -> str:
    """Return an absolute path inside the canonical mesh-data tree."""
    return str(MESH_DATA_ROOT.joinpath(*relative_parts))


def _load_adjacency_group(group: h5py.Group) -> sp.coo_matrix:
    """Load an HDF5 adjacency group as a sparsity pattern.

    The checked-in adjacency ``data`` datasets are all ones and are not used as
    numerical matrix values by the solvers.  Avoid reading that dense float64
    vector for large meshes; level-5 HyperElasticity stores roughly 1.4 GiB
    there after decompression.
    """
    shape = tuple(int(v) for v in group["shape"][:])
    index_dtype = (
        np.int32
        if max(shape, default=0) <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    row = np.asarray(group["row"][:], dtype=index_dtype)
    col = np.asarray(group["col"][:], dtype=index_dtype)
    data = np.ones(row.shape, dtype=np.bool_)
    return sp.coo_matrix((data, (row, col)), shape=shape)


def load_problem_hdf5(filename: str) -> tuple[dict[str, object], sp.coo_matrix | None]:
    """Load a problem HDF5 file and reconstruct the optional COO adjacency."""
    params: dict[str, object] = {}
    adjacency = None
    with h5py.File(filename, "r") as handle:
        for key in handle:
            if key == "adjacency":
                adjacency = _load_adjacency_group(handle[key])
                continue
            dataset = handle[key]
            params[key] = dataset[()] if dataset.shape == () else dataset[:]
    return params, adjacency


def load_problem_hdf5_fields(
    filename: str,
    *,
    fields: list[str] | tuple[str, ...] | set[str] | None = None,
    load_adjacency: bool = False,
) -> tuple[dict[str, object], sp.coo_matrix | None]:
    """Load only selected fields from a problem HDF5 file.

    This is useful for large same-mesh assets where most ranks only need the
    lightweight metadata and not the full dense operator datasets.
    """
    params: dict[str, object] = {}
    adjacency = None
    wanted = None if fields is None else {str(key) for key in fields}
    with h5py.File(filename, "r") as handle:
        if load_adjacency and "adjacency" in handle:
            adjacency = _load_adjacency_group(handle["adjacency"])
        for key in handle:
            if key == "adjacency":
                continue
            if wanted is not None and key not in wanted:
                continue
            dataset = handle[key]
            params[key] = dataset[()] if dataset.shape == () else dataset[:]
    return params, adjacency


def jaxify_problem_data(
    params: dict[str, object],
    *,
    arrays: dict[str, object],
    scalars: dict[str, type] | None = None,
) -> dict[str, object]:
    """Convert a raw parameter dictionary to JAX arrays and Python scalars."""
    import jax.numpy as jnp
    from jax import config

    config.update("jax_enable_x64", True)

    converted: dict[str, object] = {}
    for key, dtype in arrays.items():
        converted[key] = jnp.asarray(params[key], dtype=dtype)
    for key, scalar_type in (scalars or {}).items():
        converted[key] = scalar_type(params[key])
    return converted
