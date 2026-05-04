"""Shared reordered overlap-domain element assembler scaffold."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from src.core.petsc.dof_partition import _rank_of_dof_vec, petsc_ownership_range


@dataclass
class GlobalLayout:
    perm: np.ndarray
    iperm: np.ndarray
    lo: int
    hi: int
    n_free: int
    total_to_free_reord: np.ndarray
    coo_rows: np.ndarray
    coo_cols: np.ndarray
    owned_mask: np.ndarray
    owned_rows: np.ndarray
    owned_cols: np.ndarray
    owned_keys_sorted: np.ndarray
    owned_pos_sorted: np.ndarray
    elem_owner: np.ndarray


@dataclass
class LocalOverlapData:
    local_elem_idx: np.ndarray
    local_total_nodes: np.ndarray
    elems_local_np: np.ndarray
    elems_reordered: np.ndarray
    local_elem_data: dict[str, np.ndarray]
    energy_weights: np.ndarray


@dataclass
class ScatterData:
    owned_local_pos: np.ndarray
    vec_e: np.ndarray
    vec_i: np.ndarray
    vec_positions: np.ndarray
    hess_e: np.ndarray | None
    hess_i: np.ndarray | None
    hess_j: np.ndarray | None
    hess_positions: np.ndarray | None


def _owned_local_pattern_from_local_elems(
    local_elems_local_free: np.ndarray,
    *,
    owned_free_mask: np.ndarray,
    n_local_free: int,
    chunk_elems: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    index_dtype = (
        np.int32
        if int(n_local_free) <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    elems_arr = np.asarray(local_elems_local_free, dtype=index_dtype)
    owned_mask_arr = np.asarray(owned_free_mask, dtype=bool)
    key_base = np.int64(n_local_free)
    all_keys = np.zeros(0, dtype=np.int64)
    for start in range(0, int(elems_arr.shape[0]), int(chunk_elems)):
        block = elems_arr[start : start + int(chunk_elems)]
        rows = block[:, :, None]
        cols = block[:, None, :]
        valid = (rows >= 0) & (cols >= 0) & owned_mask_arr[rows]
        if not np.any(valid):
            continue
        row_vals = np.broadcast_to(rows, valid.shape)[valid].astype(np.int64, copy=False)
        col_vals = np.broadcast_to(cols, valid.shape)[valid].astype(np.int64, copy=False)
        keys = np.unique(row_vals * key_base + col_vals)
        if keys.size == 0:
            continue
        if all_keys.size == 0:
            all_keys = np.asarray(keys, dtype=np.int64)
        else:
            all_keys = np.union1d(all_keys, np.asarray(keys, dtype=np.int64))
    if all_keys.size == 0:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty
    return (
        np.asarray(all_keys // key_base, dtype=index_dtype),
        np.asarray(all_keys % key_base, dtype=index_dtype),
    )


def _array_nbytes(arr: np.ndarray | None) -> int:
    if arr is None:
        return 0
    return int(np.asarray(arr).nbytes)


def _build_block_graph(adjacency: sparse.spmatrix, block_size: int) -> sparse.csr_matrix:
    rows, cols = adjacency.nonzero()
    br = rows // block_size
    bc = cols // block_size
    n_blocks = adjacency.shape[0] // block_size
    block = sparse.coo_matrix(
        (np.ones_like(br, dtype=np.int8), (br, bc)),
        shape=(n_blocks, n_blocks),
    ).tocsr()
    block.data[:] = 1
    block.eliminate_zeros()
    return block


def _expand_block_perm(block_perm: np.ndarray, block_size: int) -> np.ndarray:
    if int(block_size) <= 1:
        return np.asarray(block_perm, dtype=np.int64)
    perm = np.empty(block_perm.size * block_size, dtype=np.int64)
    for comp in range(block_size):
        perm[comp::block_size] = block_perm * block_size + comp
    return perm


def perm_identity(n_free: int) -> np.ndarray:
    return np.arange(n_free, dtype=np.int64)


def perm_block_rcm(adjacency: sparse.spmatrix, block_size: int) -> np.ndarray:
    graph = (
        adjacency.tocsr()
        if int(block_size) <= 1
        else _build_block_graph(adjacency, int(block_size))
    )
    block_perm = np.asarray(
        reverse_cuthill_mckee(graph, symmetric_mode=True), dtype=np.int64
    )
    return _expand_block_perm(block_perm, int(block_size))


def perm_block_xyz(
    coords_all: np.ndarray,
    freedofs: np.ndarray,
    block_size: int,
) -> np.ndarray:
    block_size = int(block_size)
    freedofs_arr = np.asarray(freedofs, dtype=np.int64)
    node_ids = freedofs_arr[::block_size] // block_size
    coords = np.asarray(coords_all[node_ids], dtype=np.float64)
    sort_keys = tuple(coords[:, dim] for dim in reversed(range(coords.shape[1])))
    block_perm = np.lexsort(sort_keys).astype(np.int64)
    return _expand_block_perm(block_perm, block_size)


def perm_block_metis(
    adjacency: sparse.spmatrix,
    block_size: int,
    n_parts: int,
) -> np.ndarray:
    import pymetis

    graph = (
        adjacency.tocsr()
        if int(block_size) <= 1
        else _build_block_graph(adjacency, int(block_size))
    )
    _, part = pymetis.part_graph(n_parts, xadj=graph.indptr, adjncy=graph.indices)
    part = np.asarray(part, dtype=np.int64)
    block_ids = np.arange(graph.shape[0], dtype=np.int64)
    block_perm = np.lexsort((block_ids, part)).astype(np.int64)
    return _expand_block_perm(block_perm, int(block_size))


def select_permutation(
    reorder_mode: str,
    *,
    adjacency: sparse.spmatrix,
    coords_all: np.ndarray,
    freedofs: np.ndarray,
    n_parts: int,
    block_size: int,
) -> np.ndarray:
    if reorder_mode == "none":
        return perm_identity(len(freedofs))
    if reorder_mode == "block_rcm":
        return perm_block_rcm(adjacency, block_size)
    if reorder_mode == "block_xyz":
        return perm_block_xyz(coords_all, freedofs, block_size)
    if reorder_mode == "block_metis":
        return perm_block_metis(adjacency, block_size, n_parts)
    raise ValueError(f"Unsupported element reorder mode: {reorder_mode!r}")


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(perm.size, dtype=np.int64)
    return iperm


def _validate_permutation(perm: np.ndarray, n_free: int) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64).ravel()
    if perm.size != int(n_free):
        raise ValueError(
            f"Permutation size {perm.size} does not match number of free DOFs {n_free}"
        )
    if perm.size and (
        int(np.min(perm)) != 0
        or int(np.max(perm)) != int(n_free) - 1
        or np.unique(perm).size != perm.size
    ):
        raise ValueError("Permutation override must contain each free DOF exactly once")
    return perm


def local_vec_from_full(
    full_reordered: np.ndarray,
    total_to_free_reord: np.ndarray,
    local_total_nodes: np.ndarray,
    dirichlet_full: np.ndarray,
) -> np.ndarray:
    local_reord = total_to_free_reord[local_total_nodes]
    v_local = np.asarray(dirichlet_full[local_total_nodes], dtype=np.float64).copy()
    free_mask = local_reord >= 0
    if np.any(free_mask):
        v_local[free_mask] = full_reordered[local_reord[free_mask]]
    return v_local


def _owned_pattern_from_local_elems(
    local_elems_reordered: np.ndarray,
    *,
    lo: int,
    hi: int,
    n_free: int,
    chunk_elems: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    index_dtype = (
        np.int32
        if int(n_free) <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    key_base = np.int64(n_free)
    elems_arr = np.asarray(local_elems_reordered, dtype=index_dtype)
    all_keys = np.zeros(0, dtype=np.int64)
    for start in range(0, int(elems_arr.shape[0]), int(chunk_elems)):
        block = elems_arr[start : start + int(chunk_elems)]
        rows = block[:, :, None]
        cols = block[:, None, :]
        valid = (rows >= int(lo)) & (rows < int(hi)) & (cols >= 0)
        if not np.any(valid):
            continue
        row_vals = np.broadcast_to(rows, valid.shape)[valid].astype(np.int64, copy=False)
        col_vals = np.broadcast_to(cols, valid.shape)[valid].astype(np.int64, copy=False)
        keys = np.unique(row_vals * key_base + col_vals)
        if keys.size == 0:
            continue
        if all_keys.size == 0:
            all_keys = np.asarray(keys, dtype=np.int64)
        else:
            all_keys = np.union1d(all_keys, np.asarray(keys, dtype=np.int64))
    if all_keys.size == 0:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty
    return (
        np.asarray(all_keys // key_base, dtype=index_dtype),
        np.asarray(all_keys % key_base, dtype=index_dtype),
    )


def _owned_pattern_from_local_scalar_elems(
    local_elems_scalar: np.ndarray,
    *,
    free_reord_by_node: np.ndarray,
    lo: int,
    hi: int,
    n_free: int,
    chunk_elems: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Build owned COO pattern from scalar-node element connectivity.

    This avoids materializing full vector-valued element pair expansions in the
    hot setup path. We first deduplicate scalar node pairs and only then expand
    the surviving node pairs to the active vector components.
    """
    index_dtype = (
        np.int32 if int(n_free) <= int(np.iinfo(np.int32).max) else np.int64
    )
    scalar_elems = np.asarray(local_elems_scalar, dtype=np.int64)
    if scalar_elems.size == 0:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty

    free_by_node = np.asarray(free_reord_by_node, dtype=np.int64)
    owned_node_mask = np.any(
        (free_by_node >= int(lo)) & (free_by_node < int(hi)),
        axis=1,
    )
    free_node_mask = np.any(free_by_node >= 0, axis=1)
    key_base = np.int64(free_by_node.shape[0])
    scalar_key_batches: list[np.ndarray] = []
    pending_batch: list[np.ndarray] = []
    pending_batch_bytes = 0
    max_batch_bytes = 512 * 1024 * 1024

    for start in range(0, int(scalar_elems.shape[0]), int(chunk_elems)):
        block = scalar_elems[start : start + int(chunk_elems)]
        row_nodes = block[:, :, None]
        col_nodes = block[:, None, :]
        valid = owned_node_mask[row_nodes] & free_node_mask[col_nodes]
        if not np.any(valid):
            continue
        row_vals = np.broadcast_to(row_nodes, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        col_vals = np.broadcast_to(col_nodes, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        keys = np.unique(row_vals * key_base + col_vals)
        if keys.size == 0:
            continue
        keys = np.asarray(keys, dtype=np.int64)
        pending_batch.append(keys)
        pending_batch_bytes += int(keys.nbytes)
        if pending_batch_bytes >= max_batch_bytes:
            scalar_key_batches.append(np.unique(np.concatenate(pending_batch, axis=0)))
            pending_batch = []
            pending_batch_bytes = 0

    if pending_batch:
        scalar_key_batches.append(np.unique(np.concatenate(pending_batch, axis=0)))

    if not scalar_key_batches:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty
    scalar_keys = np.unique(np.concatenate(scalar_key_batches, axis=0))

    row_nodes = scalar_keys // key_base
    col_nodes = scalar_keys % key_base
    row_dofs = np.asarray(free_by_node[row_nodes], dtype=np.int64)
    col_dofs = np.asarray(free_by_node[col_nodes], dtype=np.int64)
    row_valid = (row_dofs >= int(lo)) & (row_dofs < int(hi))
    col_valid = col_dofs >= 0
    total_nnz = int(np.sum(np.sum(row_valid, axis=1) * np.sum(col_valid, axis=1)))
    rows = np.empty(total_nnz, dtype=index_dtype)
    cols = np.empty(total_nnz, dtype=index_dtype)
    offset = 0
    for row_comp in range(row_dofs.shape[1]):
        row_mask = row_valid[:, row_comp]
        if not np.any(row_mask):
            continue
        row_vals = np.asarray(row_dofs[:, row_comp], dtype=index_dtype)
        for col_comp in range(col_dofs.shape[1]):
            mask = row_mask & col_valid[:, col_comp]
            if not np.any(mask):
                continue
            count = int(np.count_nonzero(mask))
            rows[offset : offset + count] = row_vals[mask]
            cols[offset : offset + count] = np.asarray(
                col_dofs[:, col_comp], dtype=index_dtype
            )[mask]
            offset += count

    if offset != total_nnz:
        rows = rows[:offset]
        cols = cols[:offset]

    order = np.lexsort((cols, rows))
    return rows[order], cols[order]


def _owned_local_pattern_from_local_scalar_elems(
    local_elems_scalar: np.ndarray,
    *,
    free_local_by_node: np.ndarray,
    owned_free_mask: np.ndarray,
    n_local_free: int,
    chunk_elems: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Build local-index COO pattern from scalar-node element connectivity."""
    index_dtype = (
        np.int32 if int(n_local_free) <= int(np.iinfo(np.int32).max) else np.int64
    )
    scalar_elems = np.asarray(local_elems_scalar, dtype=np.int64)
    if scalar_elems.size == 0:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty

    free_by_node = np.asarray(free_local_by_node, dtype=np.int64)
    owned_mask = np.asarray(owned_free_mask, dtype=bool)
    free_comp_mask = free_by_node >= 0
    owned_comp_mask = np.zeros(free_by_node.shape, dtype=bool)
    if np.any(free_comp_mask):
        owned_comp_mask[free_comp_mask] = owned_mask[free_by_node[free_comp_mask]]
    owned_node_mask = np.any(owned_comp_mask, axis=1)
    free_node_mask = np.any(free_comp_mask, axis=1)
    key_base = np.int64(free_by_node.shape[0])
    scalar_key_batches: list[np.ndarray] = []

    for start in range(0, int(scalar_elems.shape[0]), int(chunk_elems)):
        block = scalar_elems[start : start + int(chunk_elems)]
        row_nodes = block[:, :, None]
        col_nodes = block[:, None, :]
        valid = owned_node_mask[row_nodes] & free_node_mask[col_nodes]
        if not np.any(valid):
            continue
        row_vals = np.broadcast_to(row_nodes, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        col_vals = np.broadcast_to(col_nodes, valid.shape)[valid].astype(
            np.int64, copy=False
        )
        keys = np.unique(row_vals * key_base + col_vals)
        if keys.size:
            scalar_key_batches.append(np.asarray(keys, dtype=np.int64))

    if not scalar_key_batches:
        empty = np.zeros(0, dtype=index_dtype)
        return empty, empty
    scalar_keys = np.unique(np.concatenate(scalar_key_batches, axis=0))

    row_nodes = scalar_keys // key_base
    col_nodes = scalar_keys % key_base
    row_dofs = np.asarray(free_by_node[row_nodes], dtype=np.int64)
    col_dofs = np.asarray(free_by_node[col_nodes], dtype=np.int64)
    row_valid = np.zeros(row_dofs.shape, dtype=bool)
    valid_rows = row_dofs >= 0
    if np.any(valid_rows):
        row_valid[valid_rows] = owned_mask[row_dofs[valid_rows]]
    col_valid = col_dofs >= 0
    total_nnz = int(np.sum(np.sum(row_valid, axis=1) * np.sum(col_valid, axis=1)))
    rows = np.empty(total_nnz, dtype=index_dtype)
    cols = np.empty(total_nnz, dtype=index_dtype)
    offset = 0
    for row_comp in range(row_dofs.shape[1]):
        row_mask = row_valid[:, row_comp]
        if not np.any(row_mask):
            continue
        row_vals = np.asarray(row_dofs[:, row_comp], dtype=index_dtype)
        for col_comp in range(col_dofs.shape[1]):
            mask = row_mask & col_valid[:, col_comp]
            if not np.any(mask):
                continue
            count = int(np.count_nonzero(mask))
            rows[offset : offset + count] = row_vals[mask]
            cols[offset : offset + count] = np.asarray(
                col_dofs[:, col_comp], dtype=index_dtype
            )[mask]
            offset += count

    if offset != total_nnz:
        rows = rows[:offset]
        cols = cols[:offset]

    order = np.lexsort((cols, rows))
    return rows[order], cols[order]


def build_global_layout(
    params: dict,
    adjacency: sparse.spmatrix | None,
    perm: np.ndarray,
    comm: MPI.Comm,
    *,
    block_size: int,
    dirichlet_key: str,
    keep_global_coo: bool = True,
) -> GlobalLayout:
    freedofs = np.asarray(params["freedofs"], dtype=np.int64)
    formula_layout = bool(params.get("_distributed_formula_layout", False))
    elems = None if formula_layout else np.asarray(params["elems"], dtype=np.int64)
    n_total = int(params.get("_distributed_n_total", 0))
    if n_total <= 0:
        n_total = int(len(np.asarray(params[dirichlet_key], dtype=np.float64)))
    n_free = int(params.get("_distributed_n_free", int(freedofs.size)))
    index_dtype = (
        np.int32
        if int(n_free) <= int(np.iinfo(np.int32).max)
        else np.int64
    )
    if formula_layout:
        iperm = np.zeros(0, dtype=index_dtype)
    elif "_distributed_iperm" in params:
        iperm = np.asarray(params["_distributed_iperm"], dtype=index_dtype)
    else:
        iperm = np.asarray(inverse_permutation(perm), dtype=index_dtype)
    if "_distributed_lo" in params and "_distributed_hi" in params:
        lo = int(params["_distributed_lo"])
        hi = int(params["_distributed_hi"])
    else:
        lo, hi = petsc_ownership_range(
            n_free, comm.rank, comm.size, block_size=int(block_size)
        )

    if formula_layout:
        total_to_free_reord = np.zeros(0, dtype=index_dtype)
    elif "_distributed_total_to_free_reord" in params:
        total_to_free_reord = np.asarray(
            params["_distributed_total_to_free_reord"], dtype=np.int64
        )
    else:
        total_to_free_orig = np.full(n_total, -1, dtype=np.int64)
        total_to_free_orig[freedofs] = np.arange(n_free, dtype=np.int64)
        total_to_free_reord = np.full(n_total, -1, dtype=np.int64)
        mask = total_to_free_orig >= 0
        total_to_free_reord[mask] = iperm[total_to_free_orig[mask]]
    key_base = np.int64(n_free)
    if adjacency is not None:
        adjacency_coo = (
            adjacency
            if sparse.isspmatrix_coo(adjacency)
            else adjacency.tocoo(copy=False)
        )
        row_adj = np.asarray(adjacency_coo.row, dtype=index_dtype)
        col_adj = np.asarray(adjacency_coo.col, dtype=index_dtype)
        coo_rows = np.asarray(iperm[row_adj], dtype=index_dtype)
        coo_cols = np.asarray(iperm[col_adj], dtype=index_dtype)
        owned_mask = (coo_rows >= lo) & (coo_rows < hi)
        owned_rows = np.asarray(coo_rows[owned_mask], dtype=index_dtype)
        owned_cols = np.asarray(coo_cols[owned_mask], dtype=index_dtype)
        if not bool(keep_global_coo):
            coo_rows = np.zeros(0, dtype=index_dtype)
            coo_cols = np.zeros(0, dtype=index_dtype)
            owned_mask = np.zeros(0, dtype=bool)

        owned_keys = (
            np.asarray(owned_rows, dtype=np.int64) * key_base
            + np.asarray(owned_cols, dtype=np.int64)
        )
        owned_sort = np.argsort(owned_keys, kind="mergesort")
        owned_keys_sorted = np.asarray(owned_keys[owned_sort], dtype=np.int64)
        owned_pos_sorted = np.asarray(owned_sort, dtype=np.int64)

        elems_reordered = total_to_free_reord[elems]
        masked = np.where(elems_reordered >= 0, elems_reordered, np.int64(n_free))
        elem_min = np.min(masked, axis=1)
        valid = elem_min < n_free
        elem_owner = np.full(len(elems), -1, dtype=np.int64)
        if np.any(valid):
            elem_owner[valid] = _rank_of_dof_vec(
                elem_min[valid],
                n_free,
                comm.size,
                block_size=int(block_size),
            )
    elif "_distributed_local_elem_idx" in params:
        if formula_layout and not bool(keep_global_coo):
            owned_rows = np.zeros(0, dtype=index_dtype)
            owned_cols = np.zeros(0, dtype=index_dtype)
        elif "_distributed_owned_rows" in params and "_distributed_owned_cols" in params:
            owned_rows = np.asarray(params["_distributed_owned_rows"], dtype=np.int64)
            owned_cols = np.asarray(params["_distributed_owned_cols"], dtype=np.int64)
        else:
            local_elem_idx = np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
            if "_distributed_local_elems_reordered" in params:
                local_elems_reordered = np.asarray(
                    params["_distributed_local_elems_reordered"], dtype=np.int64
                )
            else:
                if elems is None:
                    raise ValueError(
                        "Distributed formula layouts require "
                        "'_distributed_local_elems_reordered'"
                    )
                local_elems_total = np.asarray(elems[local_elem_idx], dtype=np.int64)
                local_elems_reordered = np.asarray(
                    total_to_free_reord[local_elems_total], dtype=np.int64
                )
            owned_rows, owned_cols = _owned_pattern_from_local_elems(
                local_elems_reordered,
                lo=int(lo),
                hi=int(hi),
                n_free=int(n_free),
            )
        owned_rows = np.asarray(owned_rows, dtype=index_dtype)
        owned_cols = np.asarray(owned_cols, dtype=index_dtype)
        coo_rows = np.asarray(owned_rows, dtype=index_dtype)
        coo_cols = np.asarray(owned_cols, dtype=index_dtype)
        owned_mask = np.ones(len(owned_rows), dtype=bool)
        owned_keys = (
            np.asarray(owned_rows, dtype=np.int64) * key_base
            + np.asarray(owned_cols, dtype=np.int64)
        )
        owned_sort = np.argsort(owned_keys, kind="mergesort")
        owned_keys_sorted = np.asarray(owned_keys[owned_sort], dtype=np.int64)
        owned_pos_sorted = np.asarray(owned_sort, dtype=np.int64)
        elem_owner = np.zeros(0, dtype=np.int64)
    else:
        raise ValueError(
            "A global adjacency matrix is required unless distributed local element "
            "data is provided via '_distributed_local_elem_idx'"
        )

    return GlobalLayout(
        perm=perm,
        iperm=iperm,
        lo=lo,
        hi=hi,
        n_free=n_free,
        total_to_free_reord=total_to_free_reord,
        coo_rows=np.asarray(coo_rows, dtype=index_dtype),
        coo_cols=np.asarray(coo_cols, dtype=index_dtype),
        owned_mask=np.asarray(owned_mask, dtype=bool),
        owned_rows=owned_rows,
        owned_cols=owned_cols,
        owned_keys_sorted=np.asarray(owned_keys_sorted, dtype=np.int64),
        owned_pos_sorted=np.asarray(owned_pos_sorted, dtype=np.int64),
        elem_owner=elem_owner,
    )


def build_local_overlap_data(
    params: dict,
    layout: GlobalLayout,
    comm: MPI.Comm,
    *,
    elem_data_keys: tuple[str, ...],
    block_size: int,
) -> LocalOverlapData:
    elems = None if "elems" not in params else np.asarray(params["elems"], dtype=np.int64)
    if "_distributed_local_elem_idx" in params:
        local_elem_idx = np.asarray(params["_distributed_local_elem_idx"], dtype=np.int64)
        if "_distributed_local_elems_total" in params:
            local_elems_total = np.asarray(
                params["_distributed_local_elems_total"], dtype=np.int64
            )
        else:
            if elems is None:
                raise ValueError(
                    "Distributed local element data requires "
                    "'_distributed_local_elems_total'"
                )
            local_elems_total = np.asarray(elems[local_elem_idx], dtype=np.int64)
        if "_distributed_local_elems_reordered" in params:
            elem_reordered_local = np.asarray(
                params["_distributed_local_elems_reordered"], dtype=np.int64
            )
        else:
            if layout.total_to_free_reord.size == 0:
                raise ValueError(
                    "Distributed formula layouts require "
                    "'_distributed_local_elems_reordered'"
                )
            elem_reordered_local = np.asarray(
                layout.total_to_free_reord[local_elems_total], dtype=np.int64
            )
        if "_distributed_energy_weights" in params:
            local_energy_weights = np.asarray(
                params["_distributed_energy_weights"], dtype=np.float64
            )
        else:
            masked = np.where(
                elem_reordered_local >= 0,
                elem_reordered_local,
                np.int64(layout.n_free),
            )
            elem_min = np.min(masked, axis=1)
            valid = elem_min < int(layout.n_free)
            local_elem_owner = np.full(len(local_elem_idx), -1, dtype=np.int64)
            if np.any(valid):
                local_elem_owner[valid] = _rank_of_dof_vec(
                    elem_min[valid],
                    int(layout.n_free),
                    int(comm.size),
                    block_size=int(block_size),
                )
            local_energy_weights = (local_elem_owner == int(comm.rank)).astype(np.float64)
        local_elem_data = {}
        for key in elem_data_keys:
            local_key = f"_distributed_{key}"
            if local_key in params:
                local_elem_data[key] = np.asarray(params[local_key], dtype=np.float64)
            else:
                local_elem_data[key] = np.asarray(params[key], dtype=np.float64)[local_elem_idx]
    else:
        if elems is None:
            raise ValueError("Replicated local-overlap setup requires 'elems'")
        elem_reordered = layout.total_to_free_reord[elems]
        local_mask = np.any(
            (elem_reordered >= layout.lo) & (elem_reordered < layout.hi), axis=1
        )
        local_elem_idx = np.where(local_mask)[0].astype(np.int64)
        local_energy_weights = (layout.elem_owner[local_elem_idx] == comm.rank).astype(
            np.float64
        )
        local_elems_total = elems[local_elem_idx]
        elem_reordered_local = np.asarray(elem_reordered[local_elem_idx], dtype=np.int64)
        local_elem_data = {
            key: np.asarray(params[key], dtype=np.float64)[local_elem_idx]
            for key in elem_data_keys
        }

    if "_distributed_local_total_nodes" in params and "_distributed_elems_local_np" in params:
        local_total_nodes = np.asarray(
            params["_distributed_local_total_nodes"], dtype=np.int64
        )
        elems_local_np = np.asarray(params["_distributed_elems_local_np"], dtype=np.int32)
    else:
        local_total_nodes, inverse = np.unique(
            local_elems_total.ravel(), return_inverse=True
        )
        elems_local_np = inverse.reshape(local_elems_total.shape).astype(np.int32)

    return LocalOverlapData(
        local_elem_idx=local_elem_idx,
        local_total_nodes=np.asarray(local_total_nodes, dtype=np.int64),
        elems_local_np=elems_local_np,
        elems_reordered=np.asarray(elem_reordered_local, dtype=np.int64),
        local_elem_data=local_elem_data,
        energy_weights=local_energy_weights,
    )


def build_near_nullspace(
    layout: GlobalLayout,
    params: dict,
    comm: MPI.Comm,
    *,
    kernel_key: str,
) -> PETSc.NullSpace:
    if "_distributed_owned_nullspace" in params:
        kernel = np.asarray(params["_distributed_owned_nullspace"], dtype=np.float64)
    elif kernel_key in params:
        kernel = np.asarray(params[kernel_key], dtype=np.float64)
    elif "nodes" in params and "freedofs" in params:
        nodes = np.asarray(params["nodes"], dtype=np.float64)
        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        owned_orig_free = np.asarray(layout.perm[layout.lo : layout.hi], dtype=np.int64)
        owned_total_dofs = np.asarray(freedofs[owned_orig_free], dtype=np.int64)
        comps = owned_total_dofs % 2
        node_ids = owned_total_dofs // 2
        center = np.mean(nodes, axis=0)
        x = np.asarray(nodes[node_ids, 0], dtype=np.float64) - float(center[0])
        y = np.asarray(nodes[node_ids, 1], dtype=np.float64) - float(center[1])
        kernel = np.zeros((layout.hi - layout.lo, 3), dtype=np.float64)
        kernel[comps == 0, 0] = 1.0
        kernel[comps == 1, 1] = 1.0
        kernel[comps == 0, 2] = -y[comps == 0]
        kernel[comps == 1, 2] = x[comps == 1]
    else:
        raise KeyError(
            f"Missing near-nullspace source {kernel_key!r} and could not derive "
            "rigid modes from nodes/freedofs"
        )
    vecs = []
    for i in range(kernel.shape[1]):
        vec = PETSc.Vec().createMPI((layout.hi - layout.lo, layout.n_free), comm=comm)
        if kernel.shape[0] == int(layout.hi - layout.lo):
            vec.array[:] = np.asarray(kernel[:, i], dtype=np.float64)
        else:
            mode = kernel[:, i][layout.perm]
            vec.array[:] = mode[layout.lo : layout.hi]
        vec.assemble()
        vecs.append(vec)
    return PETSc.NullSpace().create(vectors=vecs)


class ReorderedElementAssemblerBase:
    """Generic reordered overlap-domain PETSc assembler."""

    distribution_strategy = "overlap_allgather"
    block_size = 1
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys: tuple[str, ...] = ()
    near_nullspace_key: str | None = None

    def __init__(
        self,
        params,
        comm,
        adjacency,
        *,
        ksp_rtol=1e-3,
        ksp_type="cg",
        pc_type="gamg",
        ksp_max_it=10000,
        pc_options=None,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
        use_near_nullspace=False,
        perm_override=None,
        distribution_strategy=None,
        reuse_hessian_value_buffers=True,
        assembly_backend="coo",
        petsc_log_events=False,
        memory_guard_total_gib=None,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.params = params
        self.reorder_mode = str(reorder_mode)
        self.local_hessian_mode = str(local_hessian_mode)
        self.use_near_nullspace = bool(use_near_nullspace)
        self._formula_layout = bool(params.get("_distributed_formula_layout", False))
        self.reuse_hessian_value_buffers = bool(reuse_hessian_value_buffers)
        self.assembly_backend_requested = str(assembly_backend or "coo")
        self.assembly_backend = str(
            self._resolve_assembly_backend(self.assembly_backend_requested)
        )
        self._petsc_log_events_enabled = bool(petsc_log_events)
        self.memory_guard_total_gib = (
            None
            if memory_guard_total_gib is None
            else float(memory_guard_total_gib)
        )
        self._petsc_event_cache: dict[str, PETSc.LogEvent] = {}
        self.distribution_strategy = str(
            distribution_strategy or getattr(self, "distribution_strategy", "overlap_allgather")
        )
        if self._formula_layout and self.distribution_strategy != "overlap_p2p":
            raise ValueError(
                "Formula rank-local layouts require distribution_strategy='overlap_p2p'"
            )
        self.iter_timings = []
        self._hvp_eval_mode = "element_overlap"
        self._setup_timings: dict[str, float] = {}
        self._memory_summary: dict[str, float | int] = {}
        self._callback_stats = {
            "energy": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "kernel": 0.0,
                "allreduce": 0.0,
                "load": 0.0,
                "total": 0.0,
            },
            "gradient": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "kernel": 0.0,
                "total": 0.0,
            },
            "hessian": {
                "calls": 0,
                "allgatherv": 0.0,
                "ghost_exchange": 0.0,
                "build_v_local": 0.0,
                "hvp_compute": 0.0,
                "extraction": 0.0,
                "coo_assembly": 0.0,
                "total": 0.0,
            },
        }
        self._debug_stage_log_dir = str(
            os.environ.get("FNE_REORDERED_STAGE_LOG_DIR", "")
        ).strip()

        t_setup_total = time.perf_counter()
        self._emit_debug_stage("init_start")

        freedofs = np.asarray(params["freedofs"], dtype=np.int64)
        t0 = time.perf_counter()
        with self._petsc_event("reordered:setup_permutation"):
            if self._formula_layout:
                perm = np.zeros(0, dtype=np.int64)
            elif perm_override is None:
                if adjacency is None and self.reorder_mode not in {"none", "block_xyz"}:
                    raise ValueError(
                        "Distributed local element mode currently supports only "
                        "reorder modes 'none' and 'block_xyz' without a global adjacency"
                    )
                perm = select_permutation(
                    self.reorder_mode,
                    adjacency=adjacency,
                    coords_all=np.asarray(params[self.coordinate_key], dtype=np.float64),
                    freedofs=freedofs,
                    n_parts=self.size,
                    block_size=int(self.block_size),
                )
            else:
                perm = _validate_permutation(perm_override, len(freedofs))
        self._setup_timings["permutation"] = time.perf_counter() - t0
        self._emit_debug_stage("permutation_ready")

        t0 = time.perf_counter()
        with self._petsc_event("reordered:global_layout"):
            self.layout = build_global_layout(
                params,
                adjacency,
                perm,
                comm,
                block_size=int(self.block_size),
                dirichlet_key=self.dirichlet_key,
                keep_global_coo=self.local_hessian_mode != "element",
            )
        self._setup_timings["global_layout"] = time.perf_counter() - t0
        self._emit_debug_stage("global_layout_ready")
        self.part = SimpleNamespace(
            perm=self.layout.perm,
            iperm=self.layout.iperm,
            lo=self.layout.lo,
            hi=self.layout.hi,
            n_free=self.layout.n_free,
            n_owned=self.layout.hi - self.layout.lo,
        )
        t0 = time.perf_counter()
        with self._petsc_event("reordered:local_overlap"):
            self.local_data = build_local_overlap_data(
                params,
                self.layout,
                comm,
                elem_data_keys=self.local_elem_data_keys,
                block_size=int(self.block_size),
            )
        self._setup_timings["local_overlap"] = time.perf_counter() - t0
        self._emit_debug_stage("local_overlap_ready")
        if "_distributed_dirichlet_ref_local" in params:
            self.dirichlet_full = None
            self._dirichlet_local_template = np.asarray(
                params["_distributed_dirichlet_ref_local"], dtype=np.float64
            )
        else:
            self.dirichlet_full = np.asarray(params[self.dirichlet_key], dtype=np.float64)
            self._dirichlet_local_template = None

        t0 = time.perf_counter()
        with self._petsc_event("reordered:distribution_setup"):
            self._setup_distribution_exchange()
        self._setup_timings["distribution_setup"] = time.perf_counter() - t0
        self._emit_debug_stage("distribution_setup_ready")
        t0 = time.perf_counter()
        with self._petsc_event("reordered:matrix_backend_setup"):
            self._setup_matrix_backend_state()
        self._setup_timings["matrix_backend_setup"] = time.perf_counter() - t0
        self._emit_debug_stage("matrix_backend_setup_ready")

        t0 = time.perf_counter()
        with self._petsc_event("reordered:kernel_build"):
            (
                self._energy_jit,
                self._grad_jit,
                self._elem_hess_jit,
                self._local_grad_raw,
            ) = self._make_local_element_kernels()
        self._setup_timings["kernel_build"] = time.perf_counter() - t0
        self._emit_debug_stage("kernel_build_ready")
        t0 = time.perf_counter()
        with self._petsc_event("reordered:scatter_build"):
            self._scatter = self._build_scatter_data()
        self._setup_timings["scatter_build"] = time.perf_counter() - t0
        self._emit_debug_stage("scatter_build_ready")
        t0 = time.perf_counter()
        with self._petsc_event("reordered:rhs_build"):
            self._f_owned = np.asarray(self._build_rhs_owned(), dtype=np.float64)
        self._setup_timings["rhs_build"] = time.perf_counter() - t0
        self._emit_debug_stage("rhs_build_ready")

        t0 = time.perf_counter()
        if self.local_hessian_mode == "sfd_local":
            self._setup_local_sfd()
            self._hvp_eval_mode = "sfd_local_batched"
        elif self.local_hessian_mode == "sfd_local_vmap":
            self._setup_local_sfd()
            self._hvp_eval_mode = "sfd_local_vmap_hvpjit"
        elif self.local_hessian_mode != "element":
            raise ValueError(
                f"Unsupported local_hessian_mode={self.local_hessian_mode!r}"
            )
        self._setup_timings["local_hessian_setup"] = time.perf_counter() - t0
        self._emit_debug_stage("local_hessian_setup_ready")

        t0 = time.perf_counter()
        self._gather_sizes = np.asarray(
            comm.allgather(self.layout.hi - self.layout.lo), dtype=np.int64
        )
        self._gather_displs = np.zeros_like(self._gather_sizes)
        if len(self._gather_displs) > 1:
            self._gather_displs[1:] = np.cumsum(self._gather_sizes[:-1])
        self._setup_timings["allgather_plan"] = time.perf_counter() - t0
        self._emit_debug_stage("allgather_plan_ready")

        t0 = time.perf_counter()
        with self._petsc_event("reordered:matrix_create"):
            self.A = self._create_matrix()
        self._setup_timings["matrix_create"] = time.perf_counter() - t0
        self._emit_debug_stage("matrix_create_ready")
        owned_nnz = int(self.layout.owned_rows.size)
        self._owned_hessian_values = np.zeros(owned_nnz, dtype=np.float64)
        if np.dtype(PETSc.ScalarType) == np.dtype(np.float64):
            self._owned_hessian_values_petsc = self._owned_hessian_values
        else:
            self._owned_hessian_values_petsc = np.zeros(
                owned_nnz, dtype=PETSc.ScalarType
            )
        self._memory_summary = self._build_memory_summary()
        self._nullspace = None
        t0 = time.perf_counter()
        with self._petsc_event("reordered:nullspace_build"):
            if self.use_near_nullspace and self.near_nullspace_key is not None:
                self._nullspace = build_near_nullspace(
                    self.layout,
                    params,
                    comm,
                    kernel_key=self.near_nullspace_key,
                )
                self.A.setNearNullSpace(self._nullspace)
        self._setup_timings["nullspace_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        if pc_options:
            opts = PETSc.Options()
            for key, value in pc_options.items():
                opts[key] = value
        self.ksp.setTolerances(rtol=float(ksp_rtol), max_it=int(ksp_max_it))
        self.ksp.setFromOptions()
        self._setup_timings["ksp_create"] = time.perf_counter() - t0
        self._emit_debug_stage("ksp_create_ready")

        skip_warmup = str(os.environ.get("FNE_SKIP_REORDERED_WARMUP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        t0 = time.perf_counter()
        if skip_warmup:
            self._setup_timings["warmup"] = 0.0
        else:
            with self._petsc_event("reordered:warmup"):
                self._warmup()
            self._setup_timings["warmup"] = time.perf_counter() - t0
        self._setup_timings["total"] = time.perf_counter() - t_setup_total
        self._emit_debug_stage("init_ready")

    def _emit_debug_stage(self, stage: str) -> None:
        target_dir = str(getattr(self, "_debug_stage_log_dir", "") or "").strip()
        if not target_dir:
            return
        try:
            path = Path(target_dir) / f"rank{int(self.rank):03d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stage": str(stage),
                "rank": int(self.rank),
                "rss_gib": float(self._current_rss_gib()),
                "rss_hwm_gib": float(self._rss_gib()),
                "time_unix": float(time.time()),
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass

    def _make_local_element_kernels(self):
        raise NotImplementedError

    def _resolve_assembly_backend(self, backend: str) -> str:
        backend_name = str(backend or "coo")
        if backend_name not in {"coo", "coo_local", "blocked_local"}:
            raise ValueError(f"Unsupported assembly backend {backend_name!r}")
        return backend_name

    def _petsc_event(self, name: str):
        if not self._petsc_log_events_enabled:
            return nullcontext()
        event = self._petsc_event_cache.get(str(name))
        if event is None:
            event = PETSc.Log.Event(str(name))
            self._petsc_event_cache[str(name)] = event
        return event

    @staticmethod
    def _current_rss_gib() -> float:
        try:
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(int(parts[1]) * 1024) / float(1024**3)
        except OSError:
            pass
        return 0.0

    def _check_memory_guard(self, *, extra_local_gib: float, reason: str) -> None:
        guard_total = self.memory_guard_total_gib
        if guard_total is None or not np.isfinite(float(guard_total)) or float(guard_total) <= 0.0:
            return
        current_local = float(self._current_rss_gib())
        peak_local = current_local + float(max(0.0, extra_local_gib))
        total_peak = float(self.comm.allreduce(peak_local, op=MPI.SUM))
        max_peak = float(self.comm.allreduce(peak_local, op=MPI.MAX))
        if total_peak <= float(guard_total):
            return
        raise MemoryError(
            f"Memory guard exceeded before {reason}: "
            f"estimated_total_peak_gib={total_peak:.2f} > guard_total_gib={float(guard_total):.2f} "
            f"(max_rank_peak_gib={max_peak:.2f}, local_current_gib={current_local:.2f}, "
            f"local_extra_gib={float(max(0.0, extra_local_gib)):.2f})"
        )

    def _needs_prebuilt_hessian_scatter(self) -> bool:
        return self.local_hessian_mode == "element"

    def _setup_matrix_backend_state(self) -> None:
        index_dtype = (
            np.int32
            if int(self.layout.n_free) <= int(np.iinfo(np.int32).max)
            else np.int64
        )
        self._matrix_lgmap: PETSc.LGMap | None = None
        self._local_free_index_by_total = np.zeros(0, dtype=index_dtype)
        self._local_elems_free = np.zeros((0, 0), dtype=index_dtype)
        self._local_free_global_indices = np.zeros(0, dtype=index_dtype)
        self._local_owned_free_mask = np.zeros(0, dtype=bool)
        self._local_coo_rows = np.zeros(0, dtype=index_dtype)
        self._local_coo_cols = np.zeros(0, dtype=index_dtype)
        self._local_owned_keys_sorted = np.zeros(0, dtype=np.int64)
        self._local_owned_pos_sorted = np.zeros(0, dtype=index_dtype)
        if self.assembly_backend not in {"coo_local", "blocked_local"}:
            return
        self._local_free_index_by_total = np.full(
            len(self.local_data.local_total_nodes), -1, dtype=index_dtype
        )
        local_free_positions = np.asarray(self._dist_free_local_indices, dtype=index_dtype)
        local_free_global = np.asarray(self._dist_free_global_indices, dtype=index_dtype)
        self._local_free_global_indices = local_free_global
        self._local_owned_free_mask = (
            (local_free_global >= int(self.layout.lo))
            & (local_free_global < int(self.layout.hi))
        )
        self._local_free_index_by_total[local_free_positions] = np.arange(
            local_free_positions.size, dtype=index_dtype
        )
        self._local_elems_free = np.asarray(
            self._local_free_index_by_total[self.local_data.elems_local_np],
            dtype=index_dtype,
        )
        if self.assembly_backend == "coo_local":
            local_scalar_elems = self.params.get("_distributed_local_elems_scalar_np")
            if local_scalar_elems is not None and len(self.local_data.local_total_nodes) % int(self.block_size) == 0:
                local_node_to_free = self._local_free_index_by_total.reshape(
                    (-1, int(self.block_size))
                )
                local_rows, local_cols = _owned_local_pattern_from_local_scalar_elems(
                    np.asarray(local_scalar_elems, dtype=np.int64),
                    free_local_by_node=local_node_to_free,
                    owned_free_mask=self._local_owned_free_mask,
                    n_local_free=int(local_free_global.size),
                )
            else:
                local_rows, local_cols = _owned_local_pattern_from_local_elems(
                    self._local_elems_free,
                    owned_free_mask=self._local_owned_free_mask,
                    n_local_free=int(local_free_global.size),
                )
            self._local_coo_rows = np.asarray(local_rows, dtype=index_dtype)
            self._local_coo_cols = np.asarray(local_cols, dtype=index_dtype)
            local_keys = (
                np.asarray(self._local_coo_rows, dtype=np.int64) * np.int64(local_free_global.size)
                + np.asarray(self._local_coo_cols, dtype=np.int64)
            )
            local_sort = np.argsort(local_keys, kind="mergesort")
            self._local_owned_keys_sorted = np.asarray(local_keys[local_sort], dtype=np.int64)
            self._local_owned_pos_sorted = np.asarray(local_sort, dtype=index_dtype)
            self.layout.owned_rows = np.asarray(
                local_free_global[self._local_coo_rows], dtype=index_dtype
            )
            self.layout.owned_cols = np.asarray(
                local_free_global[self._local_coo_cols], dtype=index_dtype
            )
            self.layout.coo_rows = self.layout.owned_rows
            self.layout.coo_cols = self.layout.owned_cols
            self.layout.owned_mask = np.ones(self.layout.owned_rows.size, dtype=bool)
            owned_keys = (
                np.asarray(self.layout.owned_rows, dtype=np.int64) * np.int64(self.layout.n_free)
                + np.asarray(self.layout.owned_cols, dtype=np.int64)
            )
            owned_sort = np.argsort(owned_keys, kind="mergesort")
            self.layout.owned_keys_sorted = np.asarray(owned_keys[owned_sort], dtype=np.int64)
            self.layout.owned_pos_sorted = np.asarray(owned_sort, dtype=index_dtype)

    def _create_matrix(self) -> PETSc.Mat:
        mat = PETSc.Mat().create(comm=self.comm)
        mat.setType(PETSc.Mat.Type.MPIAIJ)
        mat.setSizes(((self.layout.hi - self.layout.lo, self.layout.n_free),) * 2)
        if self.assembly_backend == "coo_local":
            self._matrix_lgmap = PETSc.LGMap().create(
                self._local_free_global_indices.astype(PETSc.IntType, copy=False),
                comm=self.comm,
            )
            mat.setLGMap(self._matrix_lgmap, self._matrix_lgmap)
            mat.setPreallocationCOOLocal(
                self._local_coo_rows.astype(PETSc.IntType, copy=False),
                self._local_coo_cols.astype(PETSc.IntType, copy=False),
            )
        else:
            mat.setPreallocationCOO(
                self.layout.owned_rows.astype(PETSc.IntType, copy=False),
                self.layout.owned_cols.astype(PETSc.IntType, copy=False),
            )
        if int(self.block_size) > 1:
            mat.setBlockSize(int(self.block_size))
        return mat

    def _insert_owned_hessian_values(self, owned_values: np.ndarray) -> None:
        self.A.setValuesCOO(
            self._owned_hessian_values_for_petsc(owned_values),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )

    def _build_memory_summary(self) -> dict[str, float | int]:
        local_elem_bytes = 0
        for value in self.local_data.local_elem_data.values():
            local_elem_bytes += _array_nbytes(value)
        summary: dict[str, float | int] = {
            "layout_bytes": int(
                _array_nbytes(self.layout.perm)
                + _array_nbytes(self.layout.iperm)
                + _array_nbytes(self.layout.total_to_free_reord)
                + _array_nbytes(self.layout.coo_rows)
                + _array_nbytes(self.layout.coo_cols)
                + _array_nbytes(self.layout.owned_mask)
                + _array_nbytes(self.layout.owned_rows)
                + _array_nbytes(self.layout.owned_cols)
                + _array_nbytes(self.layout.owned_keys_sorted)
                + _array_nbytes(self.layout.owned_pos_sorted)
                + _array_nbytes(self.layout.elem_owner)
            ),
            "local_overlap_bytes": int(
                _array_nbytes(self.local_data.local_elem_idx)
                + _array_nbytes(self.local_data.local_total_nodes)
                + _array_nbytes(self.local_data.elems_local_np)
                + _array_nbytes(self.local_data.elems_reordered)
                + _array_nbytes(self.local_data.energy_weights)
                + int(local_elem_bytes)
            ),
            "scatter_bytes": int(
                _array_nbytes(self._scatter.owned_local_pos)
                + _array_nbytes(self._scatter.vec_e)
                + _array_nbytes(self._scatter.vec_i)
                + _array_nbytes(self._scatter.vec_positions)
                + _array_nbytes(self._scatter.hess_e)
                + _array_nbytes(self._scatter.hess_i)
                + _array_nbytes(self._scatter.hess_j)
                + _array_nbytes(self._scatter.hess_positions)
            ),
            "owned_hessian_values_bytes": int(_array_nbytes(self._owned_hessian_values)),
            "local_backend_bytes": int(
                _array_nbytes(self._local_free_index_by_total)
                + _array_nbytes(self._local_elems_free)
                + _array_nbytes(self._local_free_global_indices)
                + _array_nbytes(self._local_owned_free_mask)
                + _array_nbytes(self._local_coo_rows)
                + _array_nbytes(self._local_coo_cols)
                + _array_nbytes(self._local_owned_keys_sorted)
                + _array_nbytes(self._local_owned_pos_sorted)
            ),
            "owned_nnz": int(self.layout.owned_rows.size),
            "local_elements": int(self.local_data.local_elem_idx.size),
            "local_overlap_dofs": int(self.local_data.local_total_nodes.size),
            "assembly_backend": str(self.assembly_backend),
            "matrix_type": str(self.A.getType()),
        }
        summary["layout_gib"] = float(summary["layout_bytes"]) / (1024.0**3)
        summary["local_overlap_gib"] = float(summary["local_overlap_bytes"]) / (1024.0**3)
        summary["scatter_gib"] = float(summary["scatter_bytes"]) / (1024.0**3)
        summary["owned_hessian_values_gib"] = float(summary["owned_hessian_values_bytes"]) / (
            1024.0**3
        )
        summary["petsc_owned_values_gib"] = float(summary["owned_hessian_values_gib"])
        summary["local_backend_gib"] = float(summary["local_backend_bytes"]) / (1024.0**3)
        summary["tracked_total_gib"] = (
            float(summary["layout_gib"])
            + float(summary["local_overlap_gib"])
            + float(summary["scatter_gib"])
            + float(summary["owned_hessian_values_gib"])
            + float(summary["local_backend_gib"])
        )
        return summary

    def _build_rhs_owned(self) -> np.ndarray:
        return np.zeros(self.layout.hi - self.layout.lo, dtype=np.float64)

    def setup_summary(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self._setup_timings.items()}

    def memory_summary(self) -> dict[str, float | int]:
        return dict(self._memory_summary)

    def _reset_owned_hessian_values(self) -> np.ndarray:
        if not self.reuse_hessian_value_buffers:
            return np.zeros(int(self.layout.owned_rows.size), dtype=np.float64)
        self._owned_hessian_values.fill(0.0)
        return self._owned_hessian_values

    def _owned_hessian_values_for_petsc(self, owned_values: np.ndarray | None = None) -> np.ndarray:
        values = self._owned_hessian_values if owned_values is None else np.asarray(
            owned_values, dtype=np.float64
        )
        if not self.reuse_hessian_value_buffers:
            return np.asarray(values, dtype=PETSc.ScalarType)
        if self._owned_hessian_values_petsc is self._owned_hessian_values:
            return self._owned_hessian_values_petsc
        np.copyto(
            self._owned_hessian_values_petsc,
            values,
            casting="unsafe",
        )
        return self._owned_hessian_values_petsc

    def callback_summary(self) -> dict[str, dict[str, float | int]]:
        summary = {}
        for phase, stats in self._callback_stats.items():
            summary[phase] = {}
            for key, value in stats.items():
                if key == "calls":
                    summary[phase][key] = int(value)
                else:
                    summary[phase][key] = float(value)
        return summary

    def _record_callback(self, phase: str, **timings) -> None:
        stats = self._callback_stats[str(phase)]
        stats["calls"] = int(stats.get("calls", 0)) + 1
        for key, value in timings.items():
            stats[key] = float(stats.get(key, 0.0)) + float(value)

    def _record_hessian_iteration(self, timings: dict[str, object]) -> None:
        self._record_callback(
            "hessian",
            allgatherv=float(timings.get("allgatherv", 0.0)),
            ghost_exchange=float(timings.get("ghost_exchange", 0.0)),
            build_v_local=float(timings.get("build_v_local", 0.0)),
            hvp_compute=float(timings.get("hvp_compute", 0.0)),
            pattern_lookup=float(timings.get("pattern_lookup", 0.0)),
            accumulate=float(timings.get("accumulate", 0.0)),
            extraction=float(timings.get("extraction", 0.0)),
            coo_assembly=float(timings.get("coo_assembly", 0.0)),
            total=float(timings.get("total", 0.0)),
        )

    def _warmup(self):
        if self._dirichlet_local_template is not None:
            v_local = np.asarray(self._dirichlet_local_template, dtype=np.float64)
        else:
            v_local = np.asarray(
                self.dirichlet_full[self.local_data.local_total_nodes], dtype=np.float64
            )
        self._energy_jit(jnp.asarray(v_local)).block_until_ready()
        self._grad_jit(jnp.asarray(v_local)).block_until_ready()
        self._warmup_hessian(v_local)

    def _warmup_hessian(self, v_local: np.ndarray) -> None:
        self._elem_hess_jit(jnp.asarray(v_local)).block_until_ready()

    def _setup_local_sfd(self):
        import igraph

        local_reord = self.layout.total_to_free_reord[self.local_data.local_total_nodes]
        rows = self.layout.coo_rows
        cols = self.layout.coo_cols
        mask = np.isin(rows, local_reord) & np.isin(cols, local_reord)
        row_arr = rows[mask]
        col_arr = cols[mask]
        valid = (row_arr >= 0) & (col_arr >= 0)
        row_arr = row_arr[valid]
        col_arr = col_arr[valid]

        J_arr = np.unique(local_reord[local_reord >= 0]).astype(np.int64)
        n_J = len(J_arr)
        J_to_idx = np.full(self.layout.n_free, -1, dtype=np.int64)
        J_to_idx[J_arr] = np.arange(n_J, dtype=np.int64)

        A_J = sparse.csr_matrix(
            (
                np.ones(len(row_arr), dtype=np.float64),
                (J_to_idx[row_arr], J_to_idx[col_arr]),
            ),
            shape=(n_J, n_J),
        )
        A_J.data[:] = 1.0
        A_J.eliminate_zeros()

        A2_J = sparse.csr_matrix(A_J @ A_J)
        A2_J.data[:] = 1.0
        A2_J.eliminate_zeros()

        A2_J_coo = A2_J.tocoo()
        lo_tri = A2_J_coo.row > A2_J_coo.col
        edges = np.column_stack((A2_J_coo.row[lo_tri], A2_J_coo.col[lo_tri]))
        graph = igraph.Graph(
            n_J, edges.tolist() if len(edges) > 0 else [], directed=False
        )
        coloring_raw = graph.vertex_coloring_greedy()
        self._sfd_local_coloring = np.array(coloring_raw, dtype=np.int32).ravel()
        self._sfd_n_colors = (
            int(self._sfd_local_coloring.max() + 1) if n_J > 0 else 0
        )
        self._sfd_J_dofs = J_arr
        self._sfd_J_to_idx = J_to_idx

        # SFD on high-order vector problems can require very large
        # `(n_colors, n_local_overlap)` indicator and HVP work arrays.
        # Guard before materializing them so benchmarks fail cleanly.
        indicator_shape_bytes = (
            float(self._sfd_n_colors)
            * float(len(local_reord))
            * float(np.dtype(np.float64).itemsize)
        )
        indicator_gib = indicator_shape_bytes / float(1024**3)
        # Lower-bound estimate:
        # 1x individual indicator storage + 1x stacked indicator array + 1x HVP output.
        self._check_memory_guard(
            extra_local_gib=3.0 * float(indicator_gib),
            reason=f"SFD local setup ({self.local_hessian_mode})",
        )

        reord_to_local = np.full(self.layout.n_free, -1, dtype=np.int64)
        free_mask = local_reord >= 0
        reord_to_local[local_reord[free_mask]] = np.nonzero(free_mask)[0]

        owned_local_rows = reord_to_local[self.layout.owned_rows]
        if np.any(owned_local_rows < 0):
            raise RuntimeError("Owned reordered rows are missing from the overlap domain")
        owned_col_J_idx = J_to_idx[self.layout.owned_cols]
        if np.any(owned_col_J_idx < 0):
            raise RuntimeError("Owned reordered columns are missing from the local SFD set")
        owned_col_colors = self._sfd_local_coloring[owned_col_J_idx]

        self._sfd_color_nz = {}
        for c in range(self._sfd_n_colors):
            mask_c = owned_col_colors == c
            positions = np.where(mask_c)[0].astype(np.int64)
            local_rows = owned_local_rows[positions].astype(np.int64)
            self._sfd_color_nz[c] = (positions, local_rows)

        indicators_local = []
        for c in range(self._sfd_n_colors):
            indicator = np.zeros(len(local_reord), dtype=np.float64)
            J_dofs_c = self._sfd_J_dofs[self._sfd_local_coloring == c]
            local_idx = reord_to_local[J_dofs_c]
            indicator[local_idx] = 1.0
            indicators_local.append(jnp.array(indicator))
        self._sfd_indicators_local = indicators_local
        self._sfd_indicators_stacked = (
            jnp.stack(indicators_local)
            if len(indicators_local) > 0
            else jnp.zeros((0, len(local_reord)), dtype=jnp.float64)
        )

        def hvp_fn(v_local, tangent):
            return jax.jvp(self._local_grad_raw, (v_local,), (tangent,))[1]

        self._sfd_hvp_jit = jax.jit(hvp_fn)

        def hvp_batched(v_local, tangents):
            return jax.vmap(lambda t: hvp_fn(v_local, t))(tangents)

        self._sfd_hvp_batched_jit = jax.jit(hvp_batched)

        def hvp_vmap(v_local, tangents):
            return jax.vmap(lambda t: self._sfd_hvp_jit(v_local, t))(tangents)

        self._sfd_hvp_vmap = hvp_vmap

    def _setup_distribution_exchange(self) -> None:
        if "_distributed_local_total_to_free_reord" in self.params:
            local_reord = np.asarray(
                self.params["_distributed_local_total_to_free_reord"], dtype=np.int64
            )
        else:
            local_reord = np.asarray(
                self.layout.total_to_free_reord[self.local_data.local_total_nodes],
                dtype=np.int64,
            )
        free_mask = local_reord >= 0
        self._dist_local_reord = local_reord
        self._dist_free_local_indices = np.where(free_mask)[0].astype(np.int64)
        self._dist_free_global_indices = np.asarray(
            local_reord[self._dist_free_local_indices],
            dtype=np.int64,
        )
        if self._dirichlet_local_template is not None:
            self._dist_dirichlet_template = np.asarray(
                self._dirichlet_local_template, dtype=np.float64
            ).copy()
        else:
            self._dist_dirichlet_template = np.zeros(len(local_reord), dtype=np.float64)
            dirichlet_mask = ~free_mask
            if np.any(dirichlet_mask):
                self._dist_dirichlet_template[dirichlet_mask] = self.dirichlet_full[
                    self.local_data.local_total_nodes[dirichlet_mask]
                ]
        self._p2p_owned_local = np.zeros(0, dtype=np.int64)
        self._p2p_owned_offset = np.zeros(0, dtype=np.int64)
        self._ghost_recv: dict[int, np.ndarray] = {}
        self._ghost_send_offsets: dict[int, np.ndarray] = {}
        self._ghost_send_bufs: dict[int, np.ndarray] = {}
        self._ghost_recv_bufs: dict[int, np.ndarray] = {}
        if self.distribution_strategy != "overlap_p2p":
            return

        lo, hi = self.layout.lo, self.layout.hi
        free_global = self._dist_free_global_indices
        owned_mask = (free_global >= lo) & (free_global < hi)
        ghost_mask = ~owned_mask
        self._p2p_owned_local = self._dist_free_local_indices[owned_mask]
        self._p2p_owned_offset = (free_global[owned_mask] - lo).astype(np.int64)

        ghost_local = self._dist_free_local_indices[ghost_mask]
        ghost_global = free_global[ghost_mask]
        send_requests: dict[int, np.ndarray] = {}
        if len(ghost_global) > 0:
            ghost_owners = _rank_of_dof_vec(
                ghost_global,
                self.layout.n_free,
                self.size,
                block_size=int(self.block_size),
            )
            for owner in np.unique(ghost_owners):
                owner = int(owner)
                if owner == self.rank:
                    continue
                mask_owner = ghost_owners == owner
                owner_lo, _ = petsc_ownership_range(
                    self.layout.n_free,
                    owner,
                    self.size,
                    block_size=int(self.block_size),
                )
                self._ghost_recv[owner] = ghost_local[mask_owner]
                send_requests[owner] = (ghost_global[mask_owner] - owner_lo).astype(
                    np.int64
                )

        n_we_need = np.zeros(self.size, dtype=np.int64)
        for owner, offsets in send_requests.items():
            n_we_need[int(owner)] = len(offsets)
        n_others_need = np.zeros(self.size, dtype=np.int64)
        self.comm.Alltoall(n_we_need, n_others_need)

        recv_reqs = []
        for owner in range(self.size):
            if owner == self.rank or int(n_others_need[owner]) == 0:
                continue
            buf = np.empty(int(n_others_need[owner]), dtype=np.int64)
            recv_reqs.append((self.comm.Irecv(buf, source=owner, tag=4200), owner, buf))

        send_reqs = []
        for owner, offsets in send_requests.items():
            send_reqs.append(
                self.comm.Isend(np.ascontiguousarray(offsets), dest=int(owner), tag=4200)
            )

        for req, owner, buf in recv_reqs:
            req.Wait()
            self._ghost_send_offsets[int(owner)] = buf
        for req in send_reqs:
            req.Wait()

        self._ghost_send_bufs = {
            int(owner): np.empty(len(offsets), dtype=np.float64)
            for owner, offsets in self._ghost_send_offsets.items()
        }
        self._ghost_recv_bufs = {
            int(owner): np.empty(len(local_idx), dtype=np.float64)
            for owner, local_idx in self._ghost_recv.items()
        }

    def _p2p_fill_local(
        self,
        owned_values: np.ndarray,
        *,
        zero_dirichlet: bool,
        tag: int,
    ) -> np.ndarray:
        owned_values = np.ascontiguousarray(owned_values, dtype=np.float64)
        if zero_dirichlet:
            v_local = np.zeros(len(self._dist_local_reord), dtype=np.float64)
        else:
            v_local = self._dist_dirichlet_template.copy()
        if len(self._p2p_owned_local) > 0:
            v_local[self._p2p_owned_local] = owned_values[self._p2p_owned_offset]
        if not self._ghost_recv and not self._ghost_send_offsets:
            return v_local

        recv_reqs = []
        for owner, buf in self._ghost_recv_bufs.items():
            recv_reqs.append((self.comm.Irecv(buf, source=owner, tag=tag), owner))

        send_reqs = []
        for owner, offsets in self._ghost_send_offsets.items():
            self._ghost_send_bufs[owner][:] = owned_values[offsets]
            send_reqs.append(self.comm.Isend(self._ghost_send_bufs[owner], dest=owner, tag=tag))

        for req, owner in recv_reqs:
            req.Wait()
            v_local[self._ghost_recv[owner]] = self._ghost_recv_bufs[owner]
        for req in send_reqs:
            req.Wait()
        return v_local

    def _owned_to_local(
        self,
        owned_values: np.ndarray,
        *,
        zero_dirichlet: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.distribution_strategy == "overlap_p2p":
            t0 = time.perf_counter()
            v_local = self._p2p_fill_local(
                owned_values,
                zero_dirichlet=bool(zero_dirichlet),
                tag=4202 if zero_dirichlet else 4201,
            )
            t_exchange = time.perf_counter() - t0
            return v_local, {
                "allgatherv": 0.0,
                "ghost_exchange": float(t_exchange),
                "build_v_local": 0.0,
                "exchange_total": float(t_exchange),
            }
        if self.layout.total_to_free_reord.size == 0:
            raise RuntimeError(
                "overlap_allgather requires a global total_to_free map; "
                "use distribution_strategy='overlap_p2p' for rank-local data"
            )
        full_reordered, t_comm = self._allgather_full_owned(
            np.asarray(owned_values, dtype=np.float64)
        )
        v_local, t_build = self._build_v_local(
            full_reordered,
            zero_dirichlet=bool(zero_dirichlet),
        )
        return v_local, {
            "allgatherv": float(t_comm),
            "ghost_exchange": 0.0,
            "build_v_local": float(t_build),
            "exchange_total": float(t_comm + t_build),
        }

    def _build_scatter_data(self) -> ScatterData:
        elems_reordered = self.local_data.elems_reordered
        if hasattr(self, "_dist_local_reord"):
            local_reord = np.asarray(self._dist_local_reord, dtype=np.int64)
        else:
            local_reord = np.asarray(
                self.layout.total_to_free_reord[self.local_data.local_total_nodes],
                dtype=np.int64,
            )
        owned_mask_local = (local_reord >= self.layout.lo) & (local_reord < self.layout.hi)
        owned_rows = local_reord[owned_mask_local] - self.layout.lo
        owned_local_pos = np.full(self.layout.hi - self.layout.lo, -1, dtype=np.int64)
        owned_local_pos[owned_rows] = np.where(owned_mask_local)[0].astype(np.int64)
        if np.any(owned_local_pos < 0):
            raise RuntimeError(
                "Failed to map all owned reordered DOFs to overlap-local indices"
            )

        vec_valid = (elems_reordered >= self.layout.lo) & (elems_reordered < self.layout.hi)
        vec_vi = np.where(vec_valid)
        vec_positions = elems_reordered[vec_vi] - self.layout.lo

        if not self._needs_prebuilt_hessian_scatter():
            return ScatterData(
                owned_local_pos=owned_local_pos,
                vec_e=np.asarray(vec_vi[0], dtype=np.int64),
                vec_i=np.asarray(vec_vi[1], dtype=np.int64),
                vec_positions=np.asarray(vec_positions, dtype=np.int64),
                hess_e=None,
                hess_i=None,
                hess_j=None,
                hess_positions=None,
            )

        if self.assembly_backend == "coo_local":
            elems_lookup = np.asarray(self._local_elems_free, dtype=np.int64)
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= 0) & (cols >= 0) & self._local_owned_free_mask[rows]
        else:
            elems_lookup = elems_reordered
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= self.layout.lo) & (rows < self.layout.hi) & (cols >= 0)
        vi = np.where(valid)
        row_vals = elems_lookup[vi[0], vi[1]]
        col_vals = elems_lookup[vi[0], vi[2]]
        if self.assembly_backend == "coo_local":
            keys = row_vals.astype(np.int64) * np.int64(self._local_free_global_indices.size) + col_vals.astype(
                np.int64
            )
            key_pos = np.searchsorted(self._local_owned_keys_sorted, keys)
            key_table = self._local_owned_keys_sorted
            pos_table = self._local_owned_pos_sorted
        else:
            keys = row_vals.astype(np.int64) * np.int64(self.layout.n_free) + col_vals.astype(
                np.int64
            )
            key_pos = np.searchsorted(self.layout.owned_keys_sorted, keys)
            key_table = self.layout.owned_keys_sorted
            pos_table = self.layout.owned_pos_sorted
        if np.any(key_pos >= key_table.size):
            raise RuntimeError("Scatter lookup exceeded owned COO pattern size")
        matched = key_table[key_pos]
        if not np.array_equal(matched, keys):
            raise RuntimeError("Scatter lookup found mismatched owned COO entries")
        positions = np.asarray(pos_table[key_pos], dtype=np.int64)
        return ScatterData(
            owned_local_pos=owned_local_pos,
            vec_e=np.asarray(vec_vi[0], dtype=np.int64),
            vec_i=np.asarray(vec_vi[1], dtype=np.int64),
            vec_positions=np.asarray(vec_positions, dtype=np.int64),
            hess_e=np.asarray(vi[0], dtype=np.int64),
            hess_i=np.asarray(vi[1], dtype=np.int64),
            hess_j=np.asarray(vi[2], dtype=np.int64),
            hess_positions=positions,
        )

    def _allgather_full_owned(self, owned_values: np.ndarray) -> tuple[np.ndarray, float]:
        full = np.empty(self.layout.n_free, dtype=np.float64)
        t0 = time.perf_counter()
        self.comm.Allgatherv(
            np.asarray(owned_values, dtype=np.float64),
            [full, self._gather_sizes, self._gather_displs, MPI.DOUBLE],
        )
        return full, time.perf_counter() - t0

    def _build_v_local(
        self,
        full_reordered: np.ndarray,
        *,
        zero_dirichlet: bool = False,
    ) -> tuple[np.ndarray, float]:
        if self.layout.total_to_free_reord.size == 0:
            raise RuntimeError("Cannot build local vector from full map-free layout")
        t0 = time.perf_counter()
        v_local = local_vec_from_full(
            full_reordered,
            self.layout.total_to_free_reord,
            self.local_data.local_total_nodes,
            np.zeros_like(self.dirichlet_full) if zero_dirichlet else self.dirichlet_full,
        )
        return v_local, time.perf_counter() - t0

    def update_dirichlet(self, u_0_new):
        incoming = np.asarray(u_0_new, dtype=np.float64)
        if self._dirichlet_local_template is not None:
            if incoming.size == self._dist_local_reord.size:
                local_values = incoming.reshape(self._dist_local_reord.shape)
            else:
                local_values = incoming[self.local_data.local_total_nodes]
            self._dirichlet_local_template = np.asarray(local_values, dtype=np.float64).copy()
            dirichlet_mask = self._dist_local_reord < 0
            if np.any(dirichlet_mask):
                self._dist_dirichlet_template[dirichlet_mask] = self._dirichlet_local_template[
                    dirichlet_mask
                ]
            return
        self.dirichlet_full = incoming
        dirichlet_mask = self._dist_local_reord < 0
        if np.any(dirichlet_mask):
            self._dist_dirichlet_template[dirichlet_mask] = self.dirichlet_full[
                self.local_data.local_total_nodes[dirichlet_mask]
            ]

    def create_vec(self, full_array_reordered=None):
        vec = PETSc.Vec().createMPI(
            (self.layout.hi - self.layout.lo, self.layout.n_free),
            comm=self.comm,
        )
        if full_array_reordered is not None:
            arr = np.asarray(full_array_reordered, dtype=np.float64)
            if arr.size == self.layout.hi - self.layout.lo:
                vec.array[:] = arr
            elif arr.size == self.layout.n_free:
                vec.array[:] = arr[self.layout.lo : self.layout.hi]
            else:
                raise ValueError(
                    f"Initial vector has length {arr.size}; expected owned "
                    f"{self.layout.hi - self.layout.lo} or global {self.layout.n_free}"
                )
            vec.assemble()
        return vec

    def energy_fn(self, vec):
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(vec.array[:], dtype=np.float64),
            zero_dirichlet=False,
        )
        t_kernel0 = time.perf_counter()
        val_local = float(self._energy_jit(jnp.asarray(v_local)).block_until_ready())
        t_kernel = time.perf_counter() - t_kernel0
        t_red0 = time.perf_counter()
        energy = float(self.comm.allreduce(val_local, op=MPI.SUM))
        t_allreduce = time.perf_counter() - t_red0
        t_load = 0.0
        if self._f_owned.size == 0:
            result = energy
        else:
            t_load0 = time.perf_counter()
            load = float(
                self.comm.allreduce(np.dot(self._f_owned, vec.array[:]), op=MPI.SUM)
            )
            t_load = time.perf_counter() - t_load0
            result = energy - load
        self._record_callback(
            "energy",
            allgatherv=float(exchange["allgatherv"]),
            ghost_exchange=float(exchange["ghost_exchange"]),
            build_v_local=float(exchange["build_v_local"]),
            kernel=float(t_kernel),
            allreduce=float(t_allreduce),
            load=float(t_load),
            total=float(time.perf_counter() - t_total),
        )
        return result

    def gradient_fn(self, vec, g):
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(vec.array[:], dtype=np.float64),
            zero_dirichlet=False,
        )
        t_kernel0 = time.perf_counter()
        grad_local = np.asarray(self._grad_jit(jnp.asarray(v_local)).block_until_ready())
        t_kernel = time.perf_counter() - t_kernel0
        grad_owned = grad_local[self._scatter.owned_local_pos]
        if self._f_owned.size:
            grad_owned = grad_owned - self._f_owned
        g.array[:] = grad_owned
        self._record_callback(
            "gradient",
            allgatherv=float(exchange["allgatherv"]),
            ghost_exchange=float(exchange["ghost_exchange"]),
            build_v_local=float(exchange["build_v_local"]),
            kernel=float(t_kernel),
            total=float(time.perf_counter() - t_total),
        )

    def assemble_hessian(self, u_owned, variant=2):
        del variant
        if self.local_hessian_mode == "sfd_local":
            return self._assemble_hessian_sfd_local(u_owned)
        if self.local_hessian_mode == "sfd_local_vmap":
            return self._assemble_hessian_sfd_local_vmap(u_owned)
        return self.assemble_hessian_element(u_owned)

    def _finalize_sfd_local_hessian(self, all_hvps_np, t_comm, t_build, t_total):
        timings = {}

        t0 = time.perf_counter()
        owned_vals = self._reset_owned_hessian_values()
        for c in range(self._sfd_n_colors):
            positions, local_rows = self._sfd_color_nz[c]
            if len(positions) > 0:
                owned_vals[positions] = all_hvps_np[c, local_rows]
        timings["extraction"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self._petsc_event("reordered:hessian_matrix_insert"):
            self._insert_owned_hessian_values(owned_vals)
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(t_comm)
        timings["ghost_exchange"] = 0.0
        timings["build_v_local"] = float(t_build)
        timings["p2p_exchange"] = float(t_comm + t_build)
        timings["n_hvps"] = int(self._sfd_n_colors)
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings

    def _assemble_hessian_sfd_local(self, u_owned):
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

        timings = {}
        if self._sfd_n_colors > 0:
            t0 = time.perf_counter()
            all_hvps = self._sfd_hvp_batched_jit(
                jnp.asarray(v_local), self._sfd_indicators_stacked
            ).block_until_ready()
            all_hvps_np = np.asarray(all_hvps)
            timings["hvp_compute"] = time.perf_counter() - t0
        else:
            all_hvps_np = np.zeros((0, len(v_local)), dtype=np.float64)
            timings["hvp_compute"] = 0.0

        timings["assembly_mode"] = "sfd_overlap_local"
        finalize = self._finalize_sfd_local_hessian(
            all_hvps_np,
            float(exchange["allgatherv"] + exchange["ghost_exchange"]),
            float(exchange["build_v_local"]),
            t_total,
        )
        finalize["allgatherv"] = float(exchange["allgatherv"])
        finalize["ghost_exchange"] = float(exchange["ghost_exchange"])
        finalize.update(timings)
        return finalize

    def _assemble_hessian_sfd_local_vmap(self, u_owned):
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

        timings = {}
        if self._sfd_n_colors > 0:
            t0 = time.perf_counter()
            all_hvps = self._sfd_hvp_vmap(
                jnp.asarray(v_local), self._sfd_indicators_stacked
            )
            all_hvps_np = np.asarray(all_hvps.block_until_ready())
            timings["hvp_compute"] = time.perf_counter() - t0
        else:
            all_hvps_np = np.zeros((0, len(v_local)), dtype=np.float64)
            timings["hvp_compute"] = 0.0

        timings["assembly_mode"] = "sfd_overlap_local_vmap_hvpjit"
        finalize = self._finalize_sfd_local_hessian(
            all_hvps_np,
            float(exchange["allgatherv"] + exchange["ghost_exchange"]),
            float(exchange["build_v_local"]),
            t_total,
        )
        finalize["allgatherv"] = float(exchange["allgatherv"])
        finalize["ghost_exchange"] = float(exchange["ghost_exchange"])
        finalize.update(timings)
        return finalize

    def assemble_hessian_element(self, u_owned):
        if (
            self._scatter.hess_e is None
            or self._scatter.hess_i is None
            or self._scatter.hess_j is None
            or self._scatter.hess_positions is None
        ):
            raise RuntimeError("Prebuilt Hessian scatter data is unavailable for this assembler")
        timings = {}
        t_total = time.perf_counter()

        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )

        t0 = time.perf_counter()
        elem_hess = np.asarray(self._elem_hess_jit(jnp.asarray(v_local)).block_until_ready())
        timings["elem_hessian_compute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        contrib = elem_hess[self._scatter.hess_e, self._scatter.hess_i, self._scatter.hess_j]
        owned_vals = self._reset_owned_hessian_values()
        np.add.at(owned_vals, self._scatter.hess_positions, contrib)
        timings["scatter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self._petsc_event("reordered:hessian_matrix_insert"):
            self._insert_owned_hessian_values(owned_vals)
        timings["coo_assembly"] = time.perf_counter() - t0

        timings["allgatherv"] = float(exchange["allgatherv"])
        timings["ghost_exchange"] = float(exchange["ghost_exchange"])
        timings["build_v_local"] = float(exchange["build_v_local"])
        timings["p2p_exchange"] = float(exchange["exchange_total"])
        timings["hvp_compute"] = float(timings["elem_hessian_compute"])
        timings["extraction"] = float(timings["scatter"])
        timings["n_hvps"] = 0
        timings["assembly_mode"] = "element_overlap"
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings

    def cleanup(self):
        self.ksp.destroy()
        self.A.destroy()
        if self._matrix_lgmap is not None:
            self._matrix_lgmap.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()
