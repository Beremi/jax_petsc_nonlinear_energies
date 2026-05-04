# HyperElasticity JAX+PETSc Implementation

This note summarises the current retained JAX+PETSc HyperElasticity
implementation after the repository cleanup. It focuses on the code that still
matters in the maintained path, not the older wrapper-era layout.

## Current Layout

- CLI: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- solver: `src/problems/hyperelasticity/jax_petsc/solver.py`
- problem-specific assembly glue:
  - `src/problems/hyperelasticity/jax_petsc/parallel_hessian_dof.py`
  - `src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py`
- shared PETSc scaffolding:
  - `src/core/petsc/load_step_driver.py`
  - `src/core/petsc/reordered_element_base.py`
  - `src/core/petsc/gamg.py`
  - `src/core/benchmark/repair.py`
- shared support data:
  - `src/problems/hyperelasticity/support/`

## Goal

The maintained solver is the MPI-parallel JAX+PETSc implementation used in the
current HyperElasticity benchmark and documentation. Its target is parity with
the maintained FEniCS custom trust-region path on the same load-step schedule,
linear solver policy, and stopping criteria.

## Data Model

The solver uses vector-valued first-order tetrahedral elements. Connectivity is
expanded from nodes to vector DOFs before it reaches the distributed assembly
layer, so partitioning and COO scatter always work in flat DOF space.

Important retained traits:

- block size `3`
- block-aware free-DOF ordering (`block_xyz` default)
- rank-local HDF5 reads for the production element path
- point-to-point overlap exchange (`overlap_p2p`)
- local-overlap PETSc COO preallocation (`coo_local`)
- optional GAMG coordinates
- rigid-body near-nullspace vectors

## Assembly Modes

### Production mode: exact element Hessians

The maintained production path is:

- `--assembly_mode element`
- `--element_reorder_mode block_xyz`
- `--local_hessian_mode element`
- `--problem_build_mode rank_local`
- `--distribution_strategy overlap_p2p`
- `--assembly_backend coo_local`
- `--local_coloring`

This uses exact per-element Hessians, vmapped JAX kernels, and an overlap-based
assembler that assigns PETSc ownership after the free DOFs have been reordered.
That ownership choice was the key fix that removed the old large performance gap
between the FEniCS and JAX+PETSc HE paths.

The legacy replicated mesh build remains available only as an explicit
regression baseline through `--problem_build_mode replicated`,
`--distribution_strategy overlap_allgather`, and `--assembly_backend coo`.

### Baseline mode: sparse finite differences

The SFD path remains available for comparison, but it is not the maintained
performance path. In 3D vector P1 elasticity the graph coloring is much more
expensive than in the scalar 2D problems, so exact element Hessians are the
preferred production mode.

## Nonlinear Policy

The current maintained benchmark default is:

- trust region enabled
- PETSc `stcg` trust-subproblem solve
- post trust-subproblem line search enabled
- `linesearch_tol = 1e-1`
- `trust_radius_init = 0.5`
- `trust_shrink = 0.5`, `trust_expand = 1.5`
- `trust_eta_shrink = 0.05`, `trust_eta_expand = 0.75`
- `trust_max_reject = 6`
- `tolf = 1e-4`
- `tolg = 1e-3`
- `tolg_rel = 1e-3`
- `tolx_rel = 1e-3`
- `tolx_abs = 1e-10`

The current maintained CLI also includes the repaired trust-region flag surface
used in the replication campaign.

## Linear Policy

The current maintained distributed HE runs use:

- `ksp_type = stcg`
- `pc_type = gamg`
- `ksp_rtol = 1e-1`
- `ksp_max_it = 30`
- `pc_setup_on_ksp_cap = False`
- `gamg_threshold = 0.05`
- `gamg_agg_nsmooths = 1`
- near-nullspace on
- GAMG coordinates on

This is the same linear policy documented in the current results page:
[docs/results/HyperElasticity.md](../results/HyperElasticity.md).

## Load Stepping And Repair

The maintained solver supports:

- `--steps`
- `--total_steps`
- `--start_step`

Per-step repair retries are handled through the shared load-step driver and
repair policy. When a step fails by non-finite state or nonlinear stall, the
solver can retry with tighter linear settings before the run is marked failed.

## Current Limits

- FEniCS SNES is still not part of the maintained parity table for the showcase
  case.
- The pure-JAX HE path remains serial-only and is kept as a reference, not as a
  distributed comparison path.
- Exact element assembly is the production mode; the SFD path remains useful for
  comparison and debugging but is not the recommended scalable configuration.

## Related Docs

- [HyperElasticity problem overview](../problems/HyperElasticity.md)
- [HyperElasticity results](../results/HyperElasticity.md)
- [GAMG setup for elastic-like systems](../reference/he_gamg_elasticity_setup.md)
- archived tuning trail:
  `archive/docs/tuning/trust_region_linesearch_tuning.md`
