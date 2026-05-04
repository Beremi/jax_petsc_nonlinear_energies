from __future__ import annotations

import numpy as np
from mpi4py import MPI
from scipy import sparse

from src.core.petsc.reordered_element_base import (
    local_vec_from_full,
    select_permutation,
)
from src.problems.hyperelasticity.jax_petsc.reordered_element_assembler import (
    HEReorderedElementAssembler,
)
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D,
    load_rank_local_hyperelasticity,
    reordered_free_to_total_dofs,
    total_dofs_to_reordered_free,
)


def test_select_permutation_block_xyz_scalar_uses_coordinate_sort():
    adjacency = sparse.eye(4, format="csr")
    coords = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    freedofs = np.arange(4, dtype=np.int64)

    perm = select_permutation(
        "block_xyz",
        adjacency=adjacency,
        coords_all=coords,
        freedofs=freedofs,
        n_parts=2,
        block_size=1,
    )

    assert perm.tolist() == [2, 1, 0, 3]


def test_select_permutation_block_xyz_vector_preserves_triplets():
    adjacency = sparse.eye(6, format="csr")
    coords = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    freedofs = np.arange(6, dtype=np.int64)

    perm = select_permutation(
        "block_xyz",
        adjacency=adjacency,
        coords_all=coords,
        freedofs=freedofs,
        n_parts=2,
        block_size=3,
    )

    assert perm.tolist() == [3, 4, 5, 0, 1, 2]


def test_local_vec_from_full_merges_dirichlet_and_free_values():
    full_reordered = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    total_to_free_reord = np.array([-1, 0, 1, -1, 2], dtype=np.int64)
    local_total_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    dirichlet_full = np.array([5.0, -1.0, -1.0, 6.0, -1.0], dtype=np.float64)

    v_local = local_vec_from_full(
        full_reordered,
        total_to_free_reord,
        local_total_nodes,
        dirichlet_full,
    )

    np.testing.assert_allclose(v_local, np.array([5.0, 10.0, 20.0, 6.0, 30.0]))


def test_hyperelasticity_rank_local_block_xyz_formula_matches_permutation():
    mesh = MeshHyperElasticity3D(1)
    params, _, _ = load_rank_local_hyperelasticity(
        1,
        comm=MPI.COMM_SELF,
        reorder_mode="block_xyz",
    )
    grid = params["_he_grid"]

    perm = select_permutation(
        "block_xyz",
        adjacency=mesh.adjacency,
        coords_all=mesh.params["nodes2coord"],
        freedofs=mesh.params["freedofs"],
        n_parts=1,
        block_size=3,
    )
    reord = np.arange(int(params["_distributed_n_free"]), dtype=np.int64)
    total = reordered_free_to_total_dofs(reord, grid, "block_xyz")

    np.testing.assert_array_equal(total, mesh.params["freedofs"][perm])
    np.testing.assert_array_equal(
        total_dofs_to_reordered_free(total, grid, "block_xyz"),
        reord,
    )


def test_hyperelasticity_rank_local_comm_self_loads_all_elements_without_adjacency(monkeypatch):
    import src.problems.hyperelasticity.support.mesh as he_mesh

    mesh = MeshHyperElasticity3D(1)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("rank-local HyperElasticity loader must not call full HDF5 loader")

    monkeypatch.setattr(he_mesh, "load_problem_hdf5", _forbidden)

    params, adjacency, u_init = load_rank_local_hyperelasticity(
        1,
        comm=MPI.COMM_SELF,
        reorder_mode="block_xyz",
    )
    perm = select_permutation(
        "block_xyz",
        adjacency=mesh.adjacency,
        coords_all=mesh.params["nodes2coord"],
        freedofs=mesh.params["freedofs"],
        n_parts=1,
        block_size=3,
    )

    assert adjacency is None
    np.testing.assert_array_equal(
        params["_distributed_local_elem_idx"],
        np.arange(mesh.params["elems_scalar"].shape[0]),
    )
    np.testing.assert_array_equal(params["_distributed_local_elems_total"], mesh.params["elems"])
    np.testing.assert_allclose(params["_distributed_dphix"], mesh.params["dphix"])
    np.testing.assert_allclose(params["_distributed_dphiy"], mesh.params["dphiy"])
    np.testing.assert_allclose(params["_distributed_dphiz"], mesh.params["dphiz"])
    np.testing.assert_allclose(params["_distributed_vol"], mesh.params["vol"])
    np.testing.assert_allclose(u_init, mesh.u_init[perm])


def test_hyperelasticity_rank_local_matches_replicated_coo_on_level1():
    comm = MPI.COMM_SELF
    mesh = MeshHyperElasticity3D(1)
    replicated_params, replicated_adjacency, replicated_u = mesh.get_data()
    rank_local_params, rank_local_adjacency, _ = load_rank_local_hyperelasticity(
        1,
        comm=comm,
        reorder_mode="block_xyz",
    )

    common = dict(
        comm=comm,
        ksp_type="cg",
        pc_type="none",
        ksp_max_it=3,
        use_near_nullspace=False,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    )
    replicated = HEReorderedElementAssembler(
        replicated_params,
        adjacency=replicated_adjacency,
        distribution_strategy="overlap_allgather",
        assembly_backend="coo",
        **common,
    )
    rank_local = HEReorderedElementAssembler(
        rank_local_params,
        adjacency=rank_local_adjacency,
        distribution_strategy="overlap_p2p",
        assembly_backend="coo_local",
        **common,
    )

    x_rep = replicated.create_vec(
        np.asarray(replicated_u, dtype=np.float64)[replicated.part.perm]
    )
    x_dist = rank_local.create_vec(rank_local_params["_distributed_u_init_owned"])
    g_rep = x_rep.duplicate()
    g_dist = x_dist.duplicate()
    try:
        np.testing.assert_allclose(
            replicated.energy_fn(x_rep),
            rank_local.energy_fn(x_dist),
            rtol=1e-12,
            atol=1e-12,
        )
        replicated.gradient_fn(x_rep, g_rep)
        rank_local.gradient_fn(x_dist, g_dist)
        np.testing.assert_allclose(g_rep.array[:], g_dist.array[:], rtol=1e-10, atol=1e-9)

        replicated.assemble_hessian(x_rep.array[:])
        rank_local.assemble_hessian(x_dist.array[:])
        rep_indptr, rep_indices, rep_data = replicated.A.getValuesCSR()
        dist_indptr, dist_indices, dist_data = rank_local.A.getValuesCSR()
        np.testing.assert_array_equal(rep_indptr, dist_indptr)
        np.testing.assert_array_equal(rep_indices, dist_indices)
        np.testing.assert_allclose(rep_data, dist_data, rtol=1e-10, atol=1e-8)
    finally:
        g_rep.destroy()
        g_dist.destroy()
        x_rep.destroy()
        x_dist.destroy()
        replicated.cleanup()
        rank_local.cleanup()
