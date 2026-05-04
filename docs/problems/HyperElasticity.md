# HyperElasticity

## Mathematical Formulation

The maintained HyperElasticity benchmark minimises the compressible
Neo-Hookean stored energy

$$
\Pi(y)=\int_\Omega C_1\bigl(\operatorname{tr}(F^T F)-3-2\ln J\bigr)
+ D_1(J-1)^2\,dx,
\qquad F=\nabla y,\quad J=\det F,
$$

with $C_1 = 38461538.461538464$ and $D_1 = 83333333.33333333$.
The unknown is the deformation map $y=X+u$, so the benchmark tracks a
three-component displacement field through a sequence of large-deformation
load steps.

## Geometry, Boundary Conditions, And Discretisation

- geometry: cantilever beam on `[0, 0.4] x [-0.005, 0.005]^2`
- left face: clamped
- right face: prescribed rotating displacement
- maintained load paths: `24` and `96` steps across the full rotation
- discretisation: first-order tetrahedral vector finite elements on
  `data/meshes/HyperElasticity/`

The maintained distributed solvers use block-aware `xyz` ordering, elasticity
near-nullspace enrichment, and GAMG coordinates so the large vector-valued
linear systems remain scalable on the finest current mesh.
The JAX+PETSc element path builds rank-local overlap data directly from HDF5 by
default; the older replicated mesh build is retained only for regression
comparisons.

## Maintained Implementations

| implementation | role |
| --- | --- |
| FEniCS custom trust-region Newton | maintained MPI benchmark path |
| FEniCS SNES | retained reference, excluded from current parity |
| JAX+PETSc element Hessian | maintained MPI benchmark path and sample render |
| pure JAX serial | maintained serial reference up to level `3` |

## Curated Sample Result

The sample render below comes from the maintained JAX+PETSc element path on
the finest current showcase mesh: level `4`, `24` load steps, `32` MPI ranks.
The view is a true 3D render of the deformed beam, coloured by elastic energy
density and shown with the beam length along the horizontal `x` axis.

![HyperElasticity sample result](../assets/hyperelasticity/hyperelasticity_sample_state.png)

PDF: [HyperElasticity sample result](../assets/hyperelasticity/hyperelasticity_sample_state.pdf)  
Vector render: [HyperElasticity vector sidecar](../assets/hyperelasticity/hyperelasticity_sample_state_render_vector.pdf)

![HyperElasticity energy vs level](../assets/hyperelasticity/hyperelasticity_energy_levels.png)

PDF: [HyperElasticity energy vs level](../assets/hyperelasticity/hyperelasticity_energy_levels.pdf)

On the shared level-`1`, `24`-step parity case, the converged maintained
implementations agree on the final total energy to within the expected
trust-region solver tolerance.

## Energy Table Across Levels

Maintained `24`-step reference values at `np=1`:

| level | FEniCS custom | JAX+PETSc element | pure JAX serial |
| --- | ---: | ---: | ---: |
| 1 | 197.775 | 197.755 | 197.750 |
| 2 | 116.338 | 116.324 | 116.324 |
| 3 | 93.705 | 93.705 | 93.704 |

The pure-JAX serial reference is intentionally not extended to level `4`; the
maintained distributed comparison at that scale is between the FEniCS custom
and JAX+PETSc element paths.

## Caveats

- FEniCS SNES is not part of the current maintained parity table because it
  fails on the shared showcase case.
- The current distributed benchmark default is a trust-region method with PETSc
  `stcg + gamg`, post trust-subproblem line search, near-nullspace enrichment,
  and GAMG coordinates.
- The JAX+PETSc CLI needed a flag-parity repair during the maintained
  replication refresh; that fix is already incorporated into the canonical
  solver path.

## Where To Go Next

- current maintained comparison and scaling: [HyperElasticity results](../results/HyperElasticity.md)
- setup and environment: [quickstart](../setup/quickstart.md)
- implementation details: [HyperElasticity JAX+PETSc implementation](../implementation/hyperelasticity_jax_petsc.md)

## Commands Used

Parity showcase commands:

```bash
mpiexec -n 1 ./.venv/bin/python -u experiments/runners/run_trust_region_case.py \
  --problem he --backend fenics --level 1 \
  --steps 24 --start-step 1 --total-steps 24 --profile performance \
  --ksp-type stcg --pc-type gamg --ksp-rtol 1e-1 --ksp-max-it 30 \
  --gamg-threshold 0.05 --gamg-agg-nsmooths 1 --gamg-set-coordinates \
  --use-near-nullspace --no-pc-setup-on-ksp-cap \
  --tolf 1e-4 --tolg 1e-3 --tolg-rel 1e-3 --tolx-rel 1e-3 --tolx-abs 1e-10 \
  --maxit 100 --linesearch-a -0.5 --linesearch-b 2.0 --linesearch-tol 1e-1 \
  --use-trust-region --trust-radius-init 0.5 \
  --trust-radius-min 1e-8 --trust-radius-max 1e6 \
  --trust-shrink 0.5 --trust-expand 1.5 \
  --trust-eta-shrink 0.05 --trust-eta-expand 0.75 --trust-max-reject 6 \
  --trust-subproblem-line-search --save-history --save-linear-timing --quiet \
  --out artifacts/raw_results/docs_showcase/hyperelasticity/fenics_custom/output.json
```

```bash
./.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py \
  --level 1 --steps 24 --total_steps 24 --quiet \
  --out artifacts/raw_results/docs_showcase/hyperelasticity/jax_serial/output.json \
  --state-out artifacts/raw_results/docs_showcase/hyperelasticity/jax_serial/state.npz
```

Finest current showcase render:

```bash
mpiexec -n 32 ./.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
  --level 4 --steps 24 --total_steps 24 --profile performance \
  --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates \
  --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz \
  --local_hessian_mode element --local_coloring --use_trust_region \
  --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 \
  --trust_radius_min 1e-8 --trust_radius_max 1e6 \
  --trust_shrink 0.5 --trust_expand 1.5 \
  --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --trust_max_reject 6 \
  --tolf 1e-4 --tolg 1e-3 --tolg_rel 1e-3 --tolx_rel 1e-3 --tolx_abs 1e-10 \
  --maxit 100 --quiet \
  --out artifacts/raw_results/docs_showcase/hyperelasticity/jax_petsc_element_l4_np32/output.json \
  --state-out artifacts/raw_results/docs_showcase/hyperelasticity/jax_petsc_element_l4_np32/state.npz
```
