"""Production HE element assembler using reordered PETSc ownership + overlap domains."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase
from src.problems.hyperelasticity.jax_petsc.parallel_hessian_dof import _he_energy_density


config.update("jax_enable_x64", True)


class HEReorderedElementAssembler(ReorderedElementAssemblerBase):
    """Production HE element assembler using overlap domains and reordered ownership."""

    block_size = 3
    coordinate_key = "nodes2coord"
    dirichlet_key = "u_0_ref"
    local_elem_data_keys = ("dphix", "dphiy", "dphiz", "vol")
    near_nullspace_key = "elastic_kernel"

    def __init__(
        self,
        params,
        comm,
        adjacency,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="gamg",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder_mode="block_xyz",
        use_abs_det=False,
        local_hessian_mode="element",
        distribution_strategy=None,
        assembly_backend="coo",
    ):
        self.use_abs_det = bool(use_abs_det)
        self._kernel_cache: dict[str, object] | None = None
        self.p4_hessian_chunk_size = max(
            1,
            int(params.get("p4_hessian_chunk_size", 16)),
        )
        super().__init__(
            params,
            comm,
            adjacency,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            use_near_nullspace=use_near_nullspace,
            pc_options=pc_options,
            reorder_mode=reorder_mode,
            local_hessian_mode=local_hessian_mode,
            distribution_strategy=distribution_strategy,
            assembly_backend=assembly_backend,
        )

    def _build_local_element_kernels(self) -> dict[str, object]:
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dphix = jnp.asarray(self.local_data.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy = jnp.asarray(self.local_data.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz = jnp.asarray(self.local_data.local_elem_data["dphiz"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)
        degree = int(self.params.get("element_degree", 1))

        C1 = float(self.params["C1"])
        D1 = float(self.params["D1"])
        use_abs_det = self.use_abs_det

        def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
            W = _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, use_abs_det)
            return jnp.sum(W * vol_e)

        hess_elem = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0, 0))

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, vol
            )
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(element_energy, in_axes=(0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, vol
            )
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return hess_elem(v_e, dphix, dphiy, dphiz, vol)

        @jax.jit
        def elem_hess_batch(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
            return hess_elem(v_e, dphix_e, dphiy_e, dphiz_e, vol_e)

        def elem_hess_chunk_fn(v_local, start: int, stop: int):
            chunk = slice(int(start), int(stop))
            v_e = v_local[elems[chunk]]
            return elem_hess_batch(
                v_e,
                dphix[chunk],
                dphiy[chunk],
                dphiz[chunk],
                vol[chunk],
            )

        return {
            "degree": int(degree),
            "energy_fn": energy_fn,
            "local_grad_fn": local_grad_fn,
            "elem_hess_fn": elem_hess_fn,
            "elem_hess_chunk_fn": elem_hess_chunk_fn,
            "grad_local": grad_local,
        }

    def _get_local_element_kernels(self) -> dict[str, object]:
        if self._kernel_cache is None:
            self._kernel_cache = self._build_local_element_kernels()
        return self._kernel_cache

    def _make_local_element_kernels(self):
        kernels = self._get_local_element_kernels()
        return (
            kernels["energy_fn"],
            kernels["local_grad_fn"],
            kernels["elem_hess_fn"],
            kernels["grad_local"],
        )

    def _needs_prebuilt_hessian_scatter(self) -> bool:
        return int(self.params.get("element_degree", 1)) != 4

    def _warmup_hessian(self, v_local: np.ndarray) -> None:
        kernels = self._get_local_element_kernels()
        n_local_elem = int(self.local_data.elems_local_np.shape[0])
        if int(kernels["degree"]) == 4 and n_local_elem > 0:
            stop = min(int(self.p4_hessian_chunk_size), n_local_elem)
            kernels["elem_hess_chunk_fn"](jnp.asarray(v_local), 0, stop).block_until_ready()
            return
        kernels["elem_hess_fn"](jnp.asarray(v_local)).block_until_ready()

    def _chunk_scatter_pattern(
        self,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.assembly_backend == "coo_local":
            elems_lookup = np.asarray(
                self._local_elems_free[int(start) : int(stop)],
                dtype=np.int64,
            )
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= 0) & (cols >= 0) & self._local_owned_free_mask[rows]
            key_base = np.int64(max(1, int(self._local_free_global_indices.size)))
            key_table = self._local_owned_keys_sorted
            pos_table = self._local_owned_pos_sorted
        else:
            elems_lookup = np.asarray(
                self.local_data.elems_reordered[int(start) : int(stop)], dtype=np.int64
            )
            rows = elems_lookup[:, :, None]
            cols = elems_lookup[:, None, :]
            valid = (rows >= self.layout.lo) & (rows < self.layout.hi) & (cols >= 0)
            key_base = np.int64(self.layout.n_free)
            key_table = self.layout.owned_keys_sorted
            pos_table = self.layout.owned_pos_sorted

        vi = np.where(valid)
        if vi[0].size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty
        row_vals = elems_lookup[vi[0], vi[1]]
        col_vals = elems_lookup[vi[0], vi[2]]
        keys = row_vals.astype(np.int64) * key_base + col_vals.astype(np.int64)
        key_pos = np.searchsorted(key_table, keys)
        if np.any(key_pos >= key_table.size):
            raise RuntimeError("Chunk scatter lookup exceeded owned COO pattern size")
        matched = key_table[key_pos]
        if not np.array_equal(matched, keys):
            raise RuntimeError("Chunk scatter lookup found mismatched owned COO entries")
        return (
            np.asarray(vi[0], dtype=np.int64),
            np.asarray(vi[1], dtype=np.int64),
            np.asarray(vi[2], dtype=np.int64),
            np.asarray(pos_table[key_pos], dtype=np.int64),
        )

    @staticmethod
    def _accumulate_owned_contrib(
        owned_vals: np.ndarray,
        positions: np.ndarray,
        contrib: np.ndarray,
    ) -> None:
        pos = np.asarray(positions).ravel()
        if pos.size == 0:
            return
        np.add.at(owned_vals, pos, np.asarray(contrib, dtype=np.float64).ravel())

    def assemble_hessian_element(self, u_owned):
        if int(self.params.get("element_degree", 1)) != 4:
            return super().assemble_hessian_element(u_owned)

        timings: dict[str, object] = {}
        t_total = time.perf_counter()
        v_local, exchange = self._owned_to_local(
            np.asarray(u_owned, dtype=np.float64),
            zero_dirichlet=False,
        )
        owned_vals = self._reset_owned_hessian_values()
        kernels = self._get_local_element_kernels()
        chunk_fn = kernels["elem_hess_chunk_fn"]
        chunk_elems = int(self.p4_hessian_chunk_size)
        n_local_elem = int(self.local_data.elems_local_np.shape[0])
        v_local_jax = jnp.asarray(v_local)

        timings["elem_hessian_compute"] = 0.0
        timings["scatter"] = 0.0
        timings["pattern_lookup"] = 0.0
        timings["accumulate"] = 0.0
        timings["chunk_count"] = int((n_local_elem + chunk_elems - 1) // chunk_elems)
        timings["chunk_size"] = int(chunk_elems)
        chunk_rows: list[int] = []
        for start in range(0, n_local_elem, chunk_elems):
            stop = min(start + chunk_elems, n_local_elem)
            t0 = time.perf_counter()
            elem_hess_chunk = np.asarray(
                chunk_fn(v_local_jax, start, stop).block_until_ready()
            )
            timings["elem_hessian_compute"] += float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            hess_e, hess_i, hess_j, hess_positions = self._chunk_scatter_pattern(
                start,
                stop,
            )
            timings["pattern_lookup"] += float(time.perf_counter() - t0)
            if hess_positions.size:
                t0 = time.perf_counter()
                contrib = elem_hess_chunk[hess_e, hess_i, hess_j]
                self._accumulate_owned_contrib(owned_vals, hess_positions, contrib)
                timings["accumulate"] += float(time.perf_counter() - t0)
            chunk_rows.append(int(hess_positions.size))

        timings["scatter"] = float(timings["pattern_lookup"]) + float(timings["accumulate"])
        timings["chunk_rows_max"] = int(max(chunk_rows) if chunk_rows else 0)

        t0 = time.perf_counter()
        self._insert_owned_hessian_values(owned_vals)
        timings["coo_assembly"] = time.perf_counter() - t0
        timings["allgatherv"] = float(exchange["allgatherv"])
        timings["ghost_exchange"] = float(exchange["ghost_exchange"])
        timings["build_v_local"] = float(exchange["build_v_local"])
        timings["p2p_exchange"] = float(exchange["exchange_total"])
        timings["hvp_compute"] = float(timings["elem_hessian_compute"])
        timings["extraction"] = float(timings["scatter"])
        timings["n_hvps"] = 0
        timings["assembly_mode"] = "element_overlap_p4_chunked"
        timings["total"] = time.perf_counter() - t_total
        self.iter_timings.append(timings)
        self._record_hessian_iteration(timings)
        return timings
