# Paper Replications

This runbook is scoped to the paper experiments only. The docs-only
`pLaplace_u3_thesis` and `pLaplace_up_arctan` studies are intentionally not
primary replication entries here.

Run commands from the repository root unless a block explicitly changes
directory. The commands write new reproduction outputs under
`artifacts/reproduction/...`; tracked `docs/assets/...` and `paper/...`
generated assets are touched only by the explicit asset and paper build steps.

## Common Setup

Use one campaign directory per reproduction attempt:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p "$REPRO_ROOT/runs" "$REPRO_ROOT/smoke" "$REPRO_ROOT/paper"
```

Runtime classes:

- `short`: expected to finish within 300 seconds on the maintained local setup.
- `long`: full campaign is expected to exceed 300 seconds; run the smoke script
  for a health check.
- `external`: requires an external checkout, MPI launcher, or alternate virtual
  environment in addition to this repository.

## p-Laplace

Purpose: scalar p-Laplace baseline for nonlinear solve behavior, parallel
scaling, mesh timing, energy levels, and representative state figures.

Links: [paper results](paper/sections/results.tex),
[paper benchmark definitions](paper/sections/benchmarks.tex),
[docs results](docs/results/pLaplace.md),
[runner](experiments/runners/run_plaplace_final_suite.py),
[docs asset data](experiments/analysis/docs_assets/data/plaplace),
[docs asset builder](experiments/analysis/docs_assets/build_plaplace_figures.py),
[paper table](paper/tables/generated/plaplace_benchmark_summary.tex), and
[paper figures](paper/figures/generated).

Full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_plaplace_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/plaplace"

"$PY" -u experiments/runners/run_plaplace_final_suite.py \
  --out-dir "$REPRO_ROOT/runs/plaplace/final_suite"
```

Expected outputs: `summary.json`, `summary.md`, and one JSON/Markdown record per
case under `$REPRO_ROOT/runs/plaplace/final_suite`.

Expected runtime class: `long`; the full default suite spans levels 5-9 and MPI
ranks 1-32.

Smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_plaplace_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/smoke/plaplace"

"$PY" -u experiments/runners/run_plaplace_final_suite.py \
  --out-dir "$REPRO_ROOT/smoke/plaplace/final_suite" \
  --levels 5 \
  --nprocs 1 \
  --maxit 2 \
  --max-case-wall-s 300
```

Verification status: documented. Local smoke/full execution should import
`dolfinx`; see the verification log at the end of this file for the latest
environment result.

## Ginzburg-Landau

Purpose: scalar Ginzburg-Landau benchmark for nonlinear convergence, mesh
timing, parallel scaling, and representative complex-state assets.

Links: [paper results](paper/sections/results.tex),
[paper benchmark definitions](paper/sections/benchmarks.tex),
[docs results](docs/results/GinzburgLandau.md),
[runner](experiments/runners/run_gl_final_suite.py),
[docs asset data](experiments/analysis/docs_assets/data/ginzburg_landau),
[docs asset builder](experiments/analysis/docs_assets/build_ginzburg_landau_figures.py),
[paper table](paper/tables/generated/ginzburg_landau_benchmark_summary.tex), and
[paper figures](paper/figures/generated).

Full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_gl_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/ginzburg_landau"

"$PY" -u experiments/runners/run_gl_final_suite.py \
  --out-dir "$REPRO_ROOT/runs/ginzburg_landau/final_suite"
```

Expected outputs: `summary.json`, `summary.md`, and per-case JSON/Markdown
records under `$REPRO_ROOT/runs/ginzburg_landau/final_suite`.

Expected runtime class: `long`; the default suite spans levels 5-9 and MPI ranks
1-32.

Smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_gl_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/smoke/ginzburg_landau"

"$PY" -u experiments/runners/run_gl_final_suite.py \
  --out-dir "$REPRO_ROOT/smoke/ginzburg_landau/final_suite" \
  --levels 5 \
  --nprocs 1 \
  --maxit 2 \
  --max-case-wall-s 300
```

Verification status: documented. Local smoke/full execution should import
`dolfinx`; see the verification log at the end of this file for the latest
environment result.

## HyperElasticity

Purpose: finite-strain hyperelasticity benchmark, including the maintained
FEniCS/JAX-PETSc suite, pure-JAX companion, and JAX-FEM external baseline.

Links: [paper results](paper/sections/results.tex),
[paper benchmark definitions](paper/sections/benchmarks.tex),
[docs results](docs/results/HyperElasticity.md),
[implementation note](docs/implementation/hyperelasticity_jax_petsc.md),
[maintained runner](experiments/runners/run_he_final_suite_best.py),
[pure-JAX runner](experiments/runners/run_he_pure_jax_suite_best.py),
[JAX-FEM baseline runner](experiments/runners/run_jax_fem_hyperelastic_baseline.py),
[JAX-FEM asset generator](experiments/analysis/generate_jax_fem_hyperelastic_baseline_assets.py),
[docs asset data](experiments/analysis/docs_assets/data/hyperelasticity),
[paper table](paper/tables/generated/hyperelasticity_benchmark_summary.tex), and
[JAX-FEM paper table](paper/tables/generated/jax_fem_hyperelastic_baseline.tex).

Full maintained-suite run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_he_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/hyperelasticity"

"$PY" -u experiments/runners/run_he_final_suite_best.py \
  --out-dir "$REPRO_ROOT/runs/hyperelasticity/final_suite_best" \
  --no-seed-known-results
```

Expected outputs: `summary.json`, `summary.md`, and per-case result files under
`$REPRO_ROOT/runs/hyperelasticity/final_suite_best`.

Expected runtime class: `long`; the default suite includes multiple levels,
ranks, and 24/96-step schedules.

Maintained-suite smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_he_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/smoke/hyperelasticity"

"$PY" -u experiments/runners/run_he_final_suite_best.py \
  --out-dir "$REPRO_ROOT/smoke/hyperelasticity/final_suite_best" \
  --levels 1 \
  --nprocs 1 \
  --total-steps 24 \
  --maxit 2 \
  --max-case-wall-s 300 \
  --no-seed-known-results
```

Pure-JAX companion full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_he_pure_jax_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/hyperelasticity"

"$PY" -u experiments/runners/run_he_pure_jax_suite_best.py \
  --out-dir "$REPRO_ROOT/runs/hyperelasticity/pure_jax_suite_best"
```

Pure-JAX expected outputs: `summary.json`, `summary.md`, and per-case JSON files
under `$REPRO_ROOT/runs/hyperelasticity/pure_jax_suite_best`.

Pure-JAX runtime class: `long` for the complete suite.

Pure-JAX smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_he_pure_jax_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/smoke/hyperelasticity"

"$PY" -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py \
  --level 1 \
  --steps 2 \
  --total_steps 2 \
  --maxit 2 \
  --quiet \
  --out "$REPRO_ROOT/smoke/hyperelasticity/pure_jax_l1_steps2.json" \
  --state-out "$REPRO_ROOT/smoke/hyperelasticity/pure_jax_l1_steps2_state.npz"
```

JAX-FEM baseline full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_jax_fem_he_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/hyperelasticity/jax_fem_baseline"

"$PY" -u experiments/runners/run_jax_fem_hyperelastic_baseline.py \
  --out-dir "$REPRO_ROOT/runs/hyperelasticity/jax_fem_baseline"

"$PY" experiments/analysis/generate_jax_fem_hyperelastic_baseline_assets.py \
  --summary "$REPRO_ROOT/runs/hyperelasticity/jax_fem_baseline/comparison_summary.json" \
  --out-dir "$REPRO_ROOT/runs/hyperelasticity/jax_fem_baseline/assets"
```

Expected outputs: `comparison_summary.json`, `run_manifest.json`, per-run state
NPZ files, and generated baseline figures/reports under the selected output
directory.

Expected runtime class: `external`; the runner expects a JAX-FEM Python
environment, defaulting to `tmp_work/jax_fem_0_0_10_py312/bin/python`.

JAX-FEM smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_jax_fem_he_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/smoke/hyperelasticity/jax_fem_baseline"

"$PY" -u experiments/runners/run_jax_fem_hyperelastic_baseline.py \
  --out-dir "$REPRO_ROOT/smoke/hyperelasticity/jax_fem_baseline" \
  --schedule 0.0025 \
  --warmup-runs 0 \
  --timing-repeats 1 \
  --maintained-maxit 2
```

Verification status: the pure-JAX smoke command was run on 2026-04-28 and
reported `result: completed`. Maintained FEniCS/JAX-PETSc smoke execution
depends on `dolfinx`; the current local dependency status is recorded below.

## Plasticity2D

Purpose: 2D slope-stability benchmark family covering the paper showcase state,
P4/L5 visual assets, and L7 P4 deep-tail best/scaling results.

Links: [paper results](paper/sections/results.tex),
[paper appendix](paper/sections/appendix.tex),
[docs problem](docs/problems/Plasticity.md),
[docs results](docs/results/Plasticity.md),
[P4/L5 solver](src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py),
[P4/L5 asset generator](experiments/analysis/generate_mc_plasticity_p4_docs_assets.py),
[L7 best runner](experiments/runners/run_slope_stability_l7_p4_deep_p1_tail_best_np8_maxit20.py),
[L7 scaling runner](experiments/runners/run_slope_stability_l7_p4_deep_p1_tail_scaling_maxit20.py),
[L7 best report](experiments/analysis/generate_slope_stability_l7_p4_deep_p1_tail_best_report.py),
[L7 scaling report](experiments/analysis/generate_slope_stability_l7_p4_deep_p1_tail_scaling_maxit20_report.py),
[paper table](paper/tables/generated/plasticity2d_benchmark_summary.tex), and
[paper figures](paper/figures/generated).

P4/L5 showcase full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_plasticity2d_showcase_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/runs/plasticity2d/mc_p4_l5"
mkdir -p "$OUT/assets"

"$PY" -u src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py \
  --level 5 \
  --elem_degree 4 \
  --lambda-target 1.0 \
  --profile performance \
  --pc_type mg \
  --mg_strategy same_mesh_p4_p2_p1_lminus1_p1 \
  --mg_variant legacy_pmg \
  --ksp_type fgmres \
  --ksp_rtol 1e-2 \
  --ksp_max_it 100 \
  --save-linear-timing \
  --no-use_trust_region \
  --quiet \
  --out "$OUT/output.json" \
  --state-out "$OUT/state.npz"

"$PY" experiments/analysis/generate_mc_plasticity_p4_docs_assets.py \
  --state "$OUT/state.npz" \
  --result "$OUT/output.json" \
  --out-dir "$OUT/assets"
```

Expected outputs: `output.json`, `state.npz`, displacement/strain figures, and
`mc_plasticity_p4_l5_assets_summary.json` under `$OUT/assets`.

Expected runtime class: `long` on small machines; use the smoke script for a
sub-300-second command.

P4/L5 smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_plasticity2d_showcase_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/plasticity2d/mc_p4_l3"
mkdir -p "$OUT"

"$PY" -u src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py \
  --level 3 \
  --elem_degree 4 \
  --lambda-target 1.0 \
  --profile performance \
  --pc_type mg \
  --mg_strategy same_mesh_p4_p2_p1_lminus1_p1 \
  --mg_variant legacy_pmg \
  --ksp_type fgmres \
  --ksp_rtol 1e-2 \
  --ksp_max_it 10 \
  --maxit 2 \
  --save-linear-timing \
  --no-use_trust_region \
  --quiet \
  --out "$OUT/output.json" \
  --state-out "$OUT/state.npz"
```

L7 best/scaling full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"

"$PY" -u experiments/runners/run_slope_stability_l7_p4_deep_p1_tail_best_np8_maxit20.py
"$PY" -u experiments/runners/run_slope_stability_l7_p4_deep_p1_tail_scaling_maxit20.py

"$PY" experiments/analysis/generate_slope_stability_l7_p4_deep_p1_tail_best_report.py \
  --input artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_best_lambda1_np8_maxit20/summary.json \
  --output-dir artifacts/reports/slope_stability_l7_p4_deep_p1_tail_best_lambda1_np8_maxit20

"$PY" experiments/analysis/generate_slope_stability_l7_p4_deep_p1_tail_scaling_maxit20_report.py \
  --input artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json \
  --output-dir artifacts/reports/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20
```

Expected outputs: default raw summaries under `artifacts/raw_results/...` and
reports under `artifacts/reports/...`. These legacy L7 runners currently own
their output paths; the command preserves their CLI rather than adding a new
`--out-dir`.

Expected runtime class: `long`.

Verification status: documented. These JAX-PETSc runs require MPI/PETSc and are
smoke-checked by the P4/L3 solver command above.

## Plasticity3D

Purpose: 3D slope-stability validation, derivative-mode ablation,
lambda=1.55 degree/mesh energy study, lambda=1.0 local-PMG scaling, and
source-implementation comparisons used in the main paper and appendix.

Links: [paper results](paper/sections/results.tex),
[paper appendix](paper/sections/appendix.tex),
[docs problem](docs/problems/Plasticity3D.md),
[docs results](docs/results/Plasticity3D.md),
[autodiff modes note](docs/implementation/plasticity3d_autodiff_modes.md),
[3D maintained solver](src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py),
[validation runner](experiments/runners/run_plasticity3d_validation.py),
[validation assets](experiments/analysis/generate_plasticity3d_validation_assets.py),
[derivative ablation runner](experiments/runners/run_plasticity3d_derivative_ablation.py),
[derivative ablation assets](experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py),
[degree study runner](experiments/runners/run_plasticity3d_lambda1p55_degree_mesh_energy_study.py),
[degree study assets](experiments/analysis/generate_plasticity3d_lambda1p55_degree_mesh_energy_assets.py),
[local PMG scaling runner](experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling.py),
[mixed source comparison runner](experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_scaling_compare.py),
[all-PMG source comparison runner](experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_scaling_compare_all_pmg.py),
[implementation scaling assets](experiments/analysis/generate_plasticity3d_impl_scaling_assets.py),
[L1_2/lambda=1 assets](experiments/analysis/generate_plasticity3d_l1_2_lambda1_docs_assets.py), and
[paper tables](paper/tables/generated).

Validation full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_validation_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/runs/plasticity3d/validation"
mkdir -p "$OUT"

"$PY" -u experiments/runners/run_plasticity3d_validation.py \
  --out-dir "$OUT" \
  --source-root "$SOURCE_ROOT" \
  --force

"$PY" experiments/analysis/generate_plasticity3d_validation_assets.py \
  --manifest-json "$OUT/validation_manifest.json" \
  --out-dir "$OUT/assets"
```

Expected outputs: `validation_manifest.json`, `comparison_summary.json`, branch
summaries, per-branch states, and validation figures under `$OUT/assets`.

Expected runtime class: `external`; requires the source implementation checkout
identified by `$SOURCE_ROOT`.

Validation smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_validation_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/plasticity3d/validation"
mkdir -p "$OUT"

"$PY" -u experiments/runners/run_plasticity3d_validation.py \
  --out-dir "$OUT" \
  --source-root "$SOURCE_ROOT" \
  --maintained-ranks 1 \
  --source-reference-ranks 1 \
  --schedule 1.0 \
  --maxit 2 \
  --force
```

Derivative ablation full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_derivative_ablation_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/runs/plasticity3d/derivative_ablation"
mkdir -p "$OUT"

"$PY" -u experiments/runners/run_plasticity3d_derivative_ablation.py \
  --out-dir "$OUT" \
  --source-root "$SOURCE_ROOT" \
  --force

"$PY" experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py \
  --summary-json "$OUT/comparison_summary.json" \
  --out-dir "$OUT/assets"
```

Expected outputs: `comparison_summary.json`, per-mode run outputs, and
derivative ablation figures under `$OUT/assets`.

Expected runtime class: `external` and often `long`.

Derivative ablation smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_derivative_ablation_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/plasticity3d/derivative_ablation"
mkdir -p "$OUT"

"$PY" -u experiments/runners/run_plasticity3d_derivative_ablation.py \
  --out-dir "$OUT" \
  --source-root "$SOURCE_ROOT" \
  --ranks 1 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --maxit 2 \
  --force
```

Lambda=1.55 degree/mesh energy study full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_degree_study_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
STUDY="$REPRO_ROOT/runs/plasticity3d/lambda1p55_degree_mesh_energy_study"
mkdir -p "$STUDY"

"$PY" -u experiments/runners/run_plasticity3d_lambda1p55_degree_mesh_energy_study.py \
  --source-root "$SOURCE_ROOT" \
  --study-dir "$STUDY" \
  --no-resume

"$PY" experiments/analysis/generate_plasticity3d_lambda1p55_degree_mesh_energy_assets.py \
  --summary-json "$STUDY/comparison_summary.json" \
  --study-dir "$STUDY" \
  --docs-out-dir "$STUDY/assets"
```

Expected outputs: `comparison_summary.json`, case output trees, degree/mesh
figures, and asset summaries under `$STUDY/assets`.

Expected runtime class: `external` and `long`; the default study includes
large 32-rank cases.

Degree-study smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_degree_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/plasticity3d/lambda1p55_single_case"
mkdir -p "$OUT"

"$PY" -u src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py \
  --mesh_name mesh_l1 \
  --elem_degree 1 \
  --lambda-target 1.55 \
  --nproc 1 \
  --pc_type gamg \
  --ksp_type fgmres \
  --ksp_rtol 1e-2 \
  --ksp_max_it 10 \
  --maxit 2 \
  --quiet \
  --out "$OUT/output.json" \
  --state-out "$OUT/state.npz"
```

Lambda=1.0 local-PMG scaling and source comparison full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_l1p0_scaling_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
LOCAL="$REPRO_ROOT/runs/plasticity3d/l1_2_lambda1_grad1e2_local_pmg_scaling"
MIXED="$REPRO_ROOT/runs/plasticity3d/l1_2_lambda1_grad1e2_scaling"
SOURCEFIXED="$REPRO_ROOT/runs/plasticity3d/l1_2_lambda1_grad1e2_scaling_all_pmg"
mkdir -p "$LOCAL" "$MIXED" "$SOURCEFIXED"

"$PY" -u experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling.py \
  --source-root "$SOURCE_ROOT" \
  --out-dir "$LOCAL" \
  --ranks 1 2 \
  --seed-ranks 4 8 16 32 \
  --grad-stop-tol 1e-2 \
  --maxit 50

"$PY" -u experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_scaling_compare.py \
  --source-root "$SOURCE_ROOT" \
  --out-dir "$MIXED" \
  --ranks 4 8 16 32 \
  --grad-stop-tol 1e-2 \
  --maxit 50

"$PY" -u experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_scaling_compare_all_pmg.py \
  --source-root "$SOURCE_ROOT" \
  --out-dir "$SOURCEFIXED" \
  --ranks 4 8 16 32 \
  --grad-stop-tol 1e-2 \
  --maxit 50

"$PY" experiments/analysis/generate_plasticity3d_impl_scaling_assets.py \
  --summary-json "$LOCAL/comparison_summary.json" \
  --out-dir "$LOCAL/assets"

"$PY" experiments/analysis/generate_plasticity3d_l1_2_lambda1_docs_assets.py \
  --local-summary "$LOCAL/comparison_summary.json" \
  --mixed-summary "$MIXED/comparison_summary.json" \
  --sourcefixed-summary "$SOURCEFIXED/comparison_summary.json" \
  --out-dir "$REPRO_ROOT/runs/plasticity3d/l1_2_lambda1_assets"
```

Expected outputs: comparison summaries, per-rank result directories, local PMG
component/scaling figures, and local/source comparison assets under
`$REPRO_ROOT/runs/plasticity3d`.

Expected runtime class: `external` and `long`.

Lambda=1.0 scaling smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/tmp/source_compare/slope_stability_petsc4py}"
CAMPAIGN="${CAMPAIGN:-paper_plasticity3d_l1p0_scaling_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/plasticity3d/l1_2_lambda1_grad1e2_local_pmg_scaling"
mkdir -p "$OUT"

"$PY" -u experiments/runners/run_plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling.py \
  --source-root "$SOURCE_ROOT" \
  --out-dir "$OUT" \
  --ranks 1 \
  --seed-ranks 1 \
  --grad-stop-tol 1e-2 \
  --maxit 2 \
  --no-resume
```

Source-continuation appendix evidence:

- Primary paper inputs are existing raw/source-campaign artifacts consumed by
  [generate_paper_tables.py](paper/scripts/generate_paper_tables.py) and
  [generate_paper_figures.py](paper/scripts/generate_paper_figures.py), including
  `artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix`
  and
  `artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix`.
- Current repository scripts do not expose a clean single-command reproduction
  wrapper for those archived source-continuation campaigns. Treat those raw
  manifests as provenance and verify the paper-consumed outputs through the
  paper pipeline below.

Verification status: documented. Full and smoke commands require the 3D source
checkout and MPI/PETSc runtime; focused runner tests are listed in the
verification log below.

## Topology

Purpose: density-based topology optimization benchmark with serial/parallel
comparisons, final-density assets, objective history, mesh timing, and scaling
plots.

Links: [paper results](paper/sections/results.tex),
[paper benchmark definitions](paper/sections/benchmarks.tex),
[docs problem](docs/problems/Topology.md),
[docs results](docs/results/Topology.md),
[implementation note](docs/implementation/topology_jax_petsc.md),
[runner](experiments/runners/run_topology_docs_suite.py),
[serial solver](src/problems/topology/jax/solve_topopt_jax.py),
[parallel solver](src/problems/topology/jax/solve_topopt_parallel.py),
[docs asset data](experiments/analysis/docs_assets/data/topology),
[docs asset builder](experiments/analysis/docs_assets/build_topology_figures.py),
[paper table](paper/tables/generated/topology_benchmark_summary.tex), and
[paper figures](paper/figures/generated).

Full run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_topology_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
mkdir -p "$REPRO_ROOT/runs/topology"

"$PY" -u experiments/runners/run_topology_docs_suite.py \
  --out-dir "$REPRO_ROOT/runs/topology"
```

Expected outputs: topology sub-run summaries, `summary.json`, state NPZ files,
CSV tables, and generated figure-ready data under `$REPRO_ROOT/runs/topology`.

Expected runtime class: `long`; the full docs suite includes serial, parallel,
mesh-timing, and scaling campaigns.

Smoke script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="${CAMPAIGN:-paper_topology_smoke_$(date +%Y%m%d_%H%M%S)}"
REPRO_ROOT="$REPO_ROOT/artifacts/reproduction/$CAMPAIGN"
OUT="$REPRO_ROOT/smoke/topology/serial"
mkdir -p "$OUT"

"$PY" -u src/problems/topology/jax/solve_topopt_jax.py \
  --nx 32 \
  --ny 16 \
  --outer_maxit 1 \
  --mechanics_maxit 5 \
  --design_maxit 5 \
  --ksp_max_it 10 \
  --design_ksp_max_it 10 \
  --mechanics_solver_type petsc_gamg \
  --design_solver_type direct \
  --quiet \
  --json_out "$OUT/output.json" \
  --state_out "$OUT/state.npz"
```

Verification status: documented. This path requires PETSc but not the
source-comparison checkout.

## Docs Asset Snapshot For Paper Figures

Purpose: regenerate the docs asset snapshots that the paper figure/table
generators copy from for p-Laplace, Ginzburg-Landau, HyperElasticity, and
Topology.

Links: [asset orchestrator](experiments/analysis/docs_assets/build_all.py),
[p-Laplace builder](experiments/analysis/docs_assets/build_plaplace_figures.py),
[Ginzburg-Landau builder](experiments/analysis/docs_assets/build_ginzburg_landau_figures.py),
[HyperElasticity builder](experiments/analysis/docs_assets/build_hyperelasticity_figures.py),
[Topology builder](experiments/analysis/docs_assets/build_topology_figures.py), and
[generated docs assets](docs/assets).

Run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"

"$PY" experiments/analysis/docs_assets/build_all.py
```

Expected outputs: refreshed PNG/PDF figures under `docs/assets/...` and asset
source metadata under `experiments/analysis/docs_assets/data/...`.

Expected runtime class: `short` if the source CSV/NPZ data already exist. This
step writes tracked docs assets by design, so run it only when refreshing paper
inputs intentionally.

Verification status: documented; tracked docs assets were already dirty before
this runbook was added.

## Paper Build Pipeline

Purpose: regenerate paper layout measurements, tables, figures, reproducibility
note, and validate the generated assets consumed by `paper/main.tex`.

Links: [paper Makefile](paper/Makefile),
[layout script](paper/scripts/measure_layout.py),
[table generator](paper/scripts/generate_paper_tables.py),
[figure generator](paper/scripts/generate_paper_figures.py),
[reproducibility note generator](paper/scripts/generate_reproducibility_note.py),
[asset validator](paper/scripts/validate_paper_assets.py),
[generated tables](paper/tables/generated), and
[generated figures](paper/figures/generated).

Full paper pipeline run script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT/paper"
"$PY" scripts/measure_layout.py
"$PY" scripts/generate_paper_tables.py
"$PY" scripts/generate_paper_figures.py
"$PY" scripts/generate_reproducibility_note.py
"$PY" scripts/validate_paper_assets.py
```

Equivalent Makefile targets:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd paper
make layout tables figures reproducibility validate
```

Expected outputs: regenerated `paper/tables/generated/*`,
`paper/figures/generated/*`, `paper/build/reproducibility_note.md`, and a clean
validation report from `validate_paper_assets.py`.

Expected runtime class: `short` for generation/validation with existing raw
inputs; PDF compilation is separate.

Verification status: see the verification log below.

## Provenance Campaigns

These are useful for audit trails but are not the primary commands to run:

- [2026-03-17 local canonical resync](artifacts/reproduction/2026-03-17_local_canonical_resync/runs) contains replicated scalar, HyperElasticity, and Topology summaries.
- [2026-03-15 stage2b final archive](archive/results/reproduction_campaigns/2026-03-15_refactor_stage2b_final) contains archived full scalar/HyperElasticity summaries.
- [docs assets publication manifest](archive/results/docs_assets_publication_runs/runs/manifest.json) records a docs-publication asset refresh.
- Existing raw Plasticity2D/3D inputs under `artifacts/raw_results/...` and `artifacts/reports/...` are provenance for paper figures/tables when a runner still owns a legacy output path.

## Verification Log

Last updated: 2026-04-30 for campaign
`artifacts/reproduction/docs_paper_20260429_085344`.

- Latest docs + paper campaign:

  The isolated overnight campaign wrote outputs under
  `artifacts/reproduction/docs_paper_20260429_085344`, using
  `paper_worktree/` for mutating paper/docs generators. The final reconciled
  result is: 52 successful full/validation task executions, 10 successful smoke
  executions, 4 runtime-policy skips, 1 accepted documented capped-Newton
  diagnostic, and 0 unresolved blockers. The raw manifest keeps 25 failed,
  missing-output, or timeout attempts for audit; each was superseded by a
  later repair/follow-up pass, a smoke substitute, or an explicit runtime
  exclusion.

  Campaign logs and summaries:
  `artifacts/reproduction/docs_paper_20260429_085344/manifest.json`,
  `artifacts/reproduction/docs_paper_20260429_085344/REPORT.md`, and
  `artifacts/reproduction/docs_paper_20260429_085344/logs/`.

- Repairs made during the campaign:

  Minimal backward-compatible fixes were added for isolated output paths in
  topology report generators, cached topology-suite resumes, validation asset
  paths, HyperElasticity direct runner output directories, HyperElasticity
  pure-JAX smoke filters, and the Plasticity3D source/local-PMG constraint
  mismatch. Repair-only reruns also reconciled current asset filenames, source
  checkout links, derivative-ablation resume/assets, source-continuation
  validation, and the paper PDF path.

- Final focused pytest suite:

  ```bash
  ./.venv/bin/python -m pytest -q tests/test_docs_publication.py tests/test_runner_summary_contracts.py tests/test_plasticity3d_validation_runner.py tests/test_plasticity3d_derivative_ablation_runner.py tests/test_jax_fem_hyperelastic_baseline_runner.py tests/test_topology_report_generators.py tests/test_he_fenics_direct_cli.py tests/test_plaplace_snes_mpi.py
  ```

  Status: passed, `26 passed in 3.83s`.

- Final isolated paper validation:

  ```bash
  cd artifacts/reproduction/docs_paper_20260429_085344/paper_worktree/paper
  ../.venv/bin/python scripts/validate_paper_assets.py
  ```

  Status: passed, `Paper assets validated.`

Historical environment checks from the previous verification pass:

- Dependency probe:

  ```bash
  ./.venv/bin/python - <<'PY'
  mods = ['jax', 'petsc4py', 'mpi4py', 'dolfinx', 'h5py', 'pyamg']
  for mod in mods:
      try:
          imported = __import__(mod)
          version = getattr(imported, '__version__', 'unknown')
          print(f'{mod}: ok {version}')
      except Exception as exc:
          print(f'{mod}: FAIL {type(exc).__name__}: {exc}')
  PY
  ```

  Status: passed after repairing the local environment. `jax 0.9.1`,
  `petsc4py 3.24.2`, `mpi4py 4.1.1`, `dolfinx 0.10.0.post5`, `h5py 3.15.1`,
  and `pyamg 5.3.0` imported successfully. The repair installed the cached
  protobuf 33 / Abseil 2508 compatibility shared libraries into
  `local_env/prefix/lib`, matching the ADIOS2/DOLFINx libraries that were built
  before the system protobuf/Abseil upgrade.

- DOLFINx smoke:

  ```bash
  ./.venv/bin/python - <<'PY'
  from mpi4py import MPI
  from dolfinx import mesh
  m = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
  print('cells', m.topology.index_map(m.topology.dim).size_local)
  print('vertices', m.topology.index_map(0).size_local)
  PY
  ```

  Status: passed, producing `cells 8` and `vertices 9`.

- HyperElasticity pure-JAX smoke:

  ```bash
  OUT="artifacts/reproduction/paper_verification_20260428/smoke/hyperelasticity"
  mkdir -p "$OUT"
  ./.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py \
    --level 1 \
    --steps 2 \
    --total_steps 2 \
    --maxit 2 \
    --quiet \
    --out "$OUT/pure_jax_l1_steps2.json" \
    --state-out "$OUT/pure_jax_l1_steps2_state.npz"
  ```

  Status: passed after fixing the pure-JAX CLI status check to treat lowercase
  `converged` messages as successful. The smoke JSON reports
  `result: completed`, 2 load steps, 2 total Newton iterations, and 2 total
  linear iterations. JAX warned that no CUDA-enabled `jaxlib` is installed and
  fell back to CPU.

- Focused pytest suite:

  ```bash
  ./.venv/bin/python -m pytest -q tests/test_docs_publication.py tests/test_runner_summary_contracts.py tests/test_plasticity3d_validation_runner.py tests/test_plasticity3d_derivative_ablation_runner.py tests/test_jax_fem_hyperelastic_baseline_runner.py tests/test_topology_report_generators.py
  ```

  Status: passed, `23 passed in 1.92s`. Plain `pytest` is not on PATH in this
  shell, so the virtualenv module form was used.

- Paper asset validation:

  ```bash
  cd paper
  ../.venv/bin/python scripts/validate_paper_assets.py
  ```

  Status: passed, `Paper assets validated.`
