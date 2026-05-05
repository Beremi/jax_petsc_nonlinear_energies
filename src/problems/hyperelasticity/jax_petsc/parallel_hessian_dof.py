"""HyperElasticity-specific configuration for the generic JAX/PETSc assemblers."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.core.petsc.jax_tools.parallel_assembler import (  # noqa: E402
    JaxProblemSpec,
    ProblemDOFHessianAssembler,
    ProblemLocalColoringAssembler,
)


def _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, use_abs_det):
    """Neo-Hookean energy density for one or many elements."""
    vx = v_e[..., 0::3]
    vy = v_e[..., 1::3]
    vz = v_e[..., 2::3]

    F11 = jnp.sum(vx * dphix_e, axis=-1)
    F12 = jnp.sum(vx * dphiy_e, axis=-1)
    F13 = jnp.sum(vx * dphiz_e, axis=-1)
    F21 = jnp.sum(vy * dphix_e, axis=-1)
    F22 = jnp.sum(vy * dphiy_e, axis=-1)
    F23 = jnp.sum(vy * dphiz_e, axis=-1)
    F31 = jnp.sum(vz * dphix_e, axis=-1)
    F32 = jnp.sum(vz * dphiy_e, axis=-1)
    F33 = jnp.sum(vz * dphiz_e, axis=-1)

    I1 = (
        F11**2
        + F12**2
        + F13**2
        + F21**2
        + F22**2
        + F23**2
        + F31**2
        + F32**2
        + F33**2
    )
    detF = (
        F11 * (F22 * F33 - F23 * F32)
        - F12 * (F21 * F33 - F23 * F31)
        + F13 * (F21 * F32 - F22 * F31)
    )
    det_for_energy = jnp.abs(detF) if use_abs_det else detF
    return C1 * (I1 - 3.0 - 2.0 * jnp.log(det_for_energy)) + D1 * (
        det_for_energy - 1.0
    ) ** 2


def _build_he_state(params, options):
    """Extract hyperelastic material data and options used by the generic layer."""
    return {
        "C1": float(params["C1"]),
        "D1": float(params["D1"]),
        "use_abs_det": bool(options.get("use_abs_det", False)),
        "use_near_nullspace": bool(options.get("use_near_nullspace", True)),
    }


def _make_he_local_energy_fns(part, state):
    """Return weighted/full local energy closures for the generic assembler."""
    elems = jnp.array(part.elems_local_np, dtype=jnp.int32)
    dphix = jnp.array(part.local_elem_data["dphix"], dtype=jnp.float64)
    dphiy = jnp.array(part.local_elem_data["dphiy"], dtype=jnp.float64)
    dphiz = jnp.array(part.local_elem_data["dphiz"], dtype=jnp.float64)
    vol_np = np.asarray(part.local_elem_data["vol"], dtype=np.float64)
    weights_np = np.asarray(part.elem_weights, dtype=np.float64)
    vol = jnp.array(vol_np, dtype=jnp.float64)
    if vol_np.ndim == 1:
        vol_w_np = vol_np * weights_np
    else:
        vol_w_np = vol_np * weights_np[:, None]
    vol_w = jnp.array(vol_w_np, dtype=jnp.float64)

    C1 = state["C1"]
    D1 = state["D1"]
    use_abs_det = state["use_abs_det"]

    def _energy_with_volume(v_local, elem_vol):
        v_e = v_local[elems]
        W = _he_energy_density(v_e, dphix, dphiy, dphiz, C1, D1, use_abs_det)
        return jnp.sum(W * elem_vol)

    def energy_weighted(v_local):
        return _energy_with_volume(v_local, vol_w)

    def energy_full(v_local):
        return _energy_with_volume(v_local, vol)

    return energy_weighted, energy_full


def _make_he_element_hessian_jit(part, state):
    """Return a JIT-compiled exact per-element Hessian operator."""
    C1 = state["C1"]
    D1 = state["D1"]
    use_abs_det = state["use_abs_det"]

    elems_jnp = jnp.array(part.elems_local_np, dtype=jnp.int32)
    dphix_jnp = jnp.array(part.local_elem_data["dphix"], dtype=jnp.float64)
    dphiy_jnp = jnp.array(part.local_elem_data["dphiy"], dtype=jnp.float64)
    dphiz_jnp = jnp.array(part.local_elem_data["dphiz"], dtype=jnp.float64)
    vol_jnp = jnp.array(part.local_elem_data["vol"], dtype=jnp.float64)

    def element_energy(v_e, dphix_e, dphiy_e, dphiz_e, vol_e):
        W = _he_energy_density(v_e, dphix_e, dphiy_e, dphiz_e, C1, D1, use_abs_det)
        return jnp.sum(W * vol_e)

    vmapped_hess = jax.vmap(jax.hessian(element_energy), in_axes=(0, 0, 0, 0, 0))

    @jax.jit
    def compute_elem_hessians(v_local):
        v_e = v_local[elems_jnp]
        return vmapped_hess(v_e, dphix_jnp, dphiy_jnp, dphiz_jnp, vol_jnp)

    return compute_elem_hessians


def _make_he_near_nullspace(params, state):
    """Return rigid-body modes when requested by the problem options."""
    if not state["use_near_nullspace"]:
        return None
    kernel = np.asarray(params["elastic_kernel"], dtype=np.float64)
    return [kernel[:, i] for i in range(kernel.shape[1])]


HE_PROBLEM_SPEC = JaxProblemSpec(
    elem_data_keys=("dphix", "dphiy", "dphiz", "vol"),
    make_local_energy_fns=_make_he_local_energy_fns,
    make_element_hessian_jit=_make_he_element_hessian_jit,
    build_state=_build_he_state,
    make_near_nullspace=_make_he_near_nullspace,
    ownership_block_size=3,
    default_reorder=False,
)


class ParallelDOFHessianAssembler(ProblemDOFHessianAssembler):
    """Global-coloring HE assembler with a problem-specific energy spec."""

    problem_spec = HE_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="hypre",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder=False,
        use_abs_det=False,
    ):
        super().__init__(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            pc_options=pc_options,
            reorder=reorder,
            problem_options={
                "use_abs_det": use_abs_det,
                "use_near_nullspace": use_near_nullspace,
            },
        )


class LocalColoringAssembler(ProblemLocalColoringAssembler):
    """Per-rank local-coloring HE assembler with a problem-specific energy spec."""

    problem_spec = HE_PROBLEM_SPEC

    def __init__(
        self,
        params,
        comm,
        adjacency=None,
        coloring_trials_per_rank=10,
        ksp_rtol=1e-1,
        ksp_type="gmres",
        pc_type="hypre",
        ksp_max_it=30,
        use_near_nullspace=True,
        pc_options=None,
        reorder=False,
        use_abs_det=False,
        hvp_eval_mode="sequential",
    ):
        super().__init__(
            params=params,
            comm=comm,
            adjacency=adjacency,
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            pc_options=pc_options,
            reorder=reorder,
            hvp_eval_mode=hvp_eval_mode,
            problem_options={
                "use_abs_det": use_abs_det,
                "use_near_nullspace": use_near_nullspace,
            },
        )
