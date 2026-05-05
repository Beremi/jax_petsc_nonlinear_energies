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
from src.problems.hyperelasticity.jax_petsc.multigrid import (
    build_he_pmg_hierarchy,
    choose_he_pmg_coarsest_level,
)
from src.problems.hyperelasticity.support.mesh import (
    MeshHyperElasticity3D,
    _node_coordinates,
    _node_ijk,
    generate_structured_elements_for_indices,
    generate_structured_element_data_for_indices,
    grid_for_level,
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


def test_hyperelasticity_procedural_mesh_matches_hdf5_rows_on_level1():
    mesh = MeshHyperElasticity3D(1)
    grid = grid_for_level(1)
    elem_idx = np.arange(mesh.params["elems_scalar"].shape[0], dtype=np.int64)
    elems = generate_structured_elements_for_indices(elem_idx, grid)
    dphix, dphiy, dphiz, vol = generate_structured_element_data_for_indices(
        elem_idx,
        grid,
    )

    np.testing.assert_array_equal(elems, mesh.params["elems_scalar"])
    np.testing.assert_allclose(dphix, mesh.params["dphix"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dphiy, mesh.params["dphiy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dphiz, mesh.params["dphiz"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vol, mesh.params["vol"], rtol=1e-12, atol=1e-18)


def test_hyperelasticity_rank_local_procedural_matches_hdf5_source_level1():
    procedural, _, u_proc = load_rank_local_hyperelasticity(
        1,
        comm=MPI.COMM_SELF,
        reorder_mode="block_xyz",
        mesh_source="procedural",
    )
    hdf5, _, u_hdf5 = load_rank_local_hyperelasticity(
        1,
        comm=MPI.COMM_SELF,
        reorder_mode="block_xyz",
        mesh_source="hdf5",
    )

    assert procedural["_distributed_mesh_source"] == "procedural"
    assert hdf5["_distributed_mesh_source"] == "hdf5"
    for key in (
        "_distributed_local_elem_idx",
        "_distributed_local_elems_total",
        "_distributed_local_elems_reordered",
        "_distributed_local_total_nodes",
        "_distributed_elems_local_np",
        "_distributed_local_elems_scalar_np",
        "_distributed_local_total_to_free_reord",
    ):
        np.testing.assert_array_equal(procedural[key], hdf5[key])
    for key in (
        "_distributed_energy_weights",
        "_distributed_dphix",
        "_distributed_dphiy",
        "_distributed_dphiz",
        "_distributed_vol",
        "_distributed_dirichlet_ref_local",
        "_distributed_u_init_owned",
        "_distributed_owned_block_coordinates",
        "_distributed_owned_nullspace",
    ):
        np.testing.assert_allclose(procedural[key], hdf5[key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u_proc, u_hdf5, rtol=1e-12, atol=1e-12)


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


def test_hyperelasticity_pmg_auto_coarsest_level_scales_with_rank_count():
    assert choose_he_pmg_coarsest_level(
        finest_level=4,
        n_ranks=16,
        requested="auto",
        min_dofs_per_rank=128,
    ) == 1
    assert choose_he_pmg_coarsest_level(
        finest_level=4,
        n_ranks=512,
        requested="auto",
        min_dofs_per_rank=128,
    ) == 3


def test_hyperelasticity_pmg_transfer_preserves_partitioned_row_shape():
    hierarchy = build_he_pmg_hierarchy(
        finest_level=2,
        coarsest_level=1,
        reorder_mode="block_xyz",
        comm=MPI.COMM_SELF,
    )
    try:
        coarse = hierarchy.levels[0]
        fine = hierarchy.levels[1]
        prolong = hierarchy.prolongations[0]
        assert prolong.getSize() == (fine.n_free, coarse.n_free)

        indptr, _, data = prolong.getValuesCSR()
        row_sums = np.zeros(fine.n_free, dtype=np.float64)
        nonempty = indptr[1:] > indptr[:-1]
        row_sums[nonempty] = np.add.reduceat(data, indptr[:-1][nonempty])

        fine_total = reordered_free_to_total_dofs(
            np.arange(fine.n_free, dtype=np.int64),
            fine.grid,
            "block_xyz",
        )
        ix, _, _ = _node_ijk(fine_total // 3, fine.grid)
        fully_interior = (ix > 1) & (ix < int(fine.grid.nx) - 1)

        np.testing.assert_allclose(row_sums[fully_interior], 1.0, atol=1e-12)
        assert float(np.min(row_sums)) >= 0.0
        assert float(np.max(row_sums)) <= 1.0

        coarse_reord = np.arange(coarse.n_free, dtype=np.int64)
        fine_reord = np.arange(fine.n_free, dtype=np.int64)
        coarse_total = reordered_free_to_total_dofs(coarse_reord, coarse.grid, "block_xyz")
        fine_total = reordered_free_to_total_dofs(fine_reord, fine.grid, "block_xyz")
        coarse_coords = _node_coordinates(coarse_total // 3, coarse.grid)
        fine_coords = _node_coordinates(fine_total // 3, fine.grid)

        coarse_vec = prolong.createVecRight()
        fine_vec = prolong.createVecLeft()
        coarse_vec.array[:] = coarse_coords[:, 0] * coarse_coords[:, 1]
        prolong.mult(coarse_vec, fine_vec)
        ix, _, _ = _node_ijk(fine_total // 3, fine.grid)
        coarse_cell_x = np.minimum(ix // 2, int(coarse.grid.nx) - 1)
        interior_x = (coarse_cell_x > 0) & (coarse_cell_x < int(coarse.grid.nx) - 1)
        np.testing.assert_allclose(
            fine_vec.array[interior_x],
            fine_coords[interior_x, 0] * fine_coords[interior_x, 1],
            rtol=1e-12,
            atol=1e-12,
        )
        coarse_vec.destroy()
        fine_vec.destroy()
    finally:
        hierarchy.cleanup()
