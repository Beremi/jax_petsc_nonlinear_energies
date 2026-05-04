"""Production HE element assembler using reordered PETSc ownership + overlap domains."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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

    def _make_local_element_kernels(self):
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)
        dphix = jnp.asarray(self.local_data.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy = jnp.asarray(self.local_data.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz = jnp.asarray(self.local_data.local_elem_data["dphiz"], dtype=jnp.float64)
        vol = jnp.asarray(self.local_data.local_elem_data["vol"], dtype=jnp.float64)
        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)

        C1 = float(self.params["C1"])
        D1 = float(self.params["D1"])
        use_abs_det = self.use_abs_det

        def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
            return (
                _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, use_abs_det)
                * vol_e
            )

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

        return energy_fn, local_grad_fn, elem_hess_fn, grad_local
