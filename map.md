# Repository Mental Map

This file is a compact index for navigating the repository without doing a full-tree search every time. It favors active implementation paths and maintained documentation over generated outputs, archived history, or local environment payloads.

Canonical truth order for day-to-day work:

1. [`src/`](src/)
2. [`docs/`](docs/)
3. [`experiments/`](experiments/)
4. [`tests/`](tests/)
5. [`data/meshes/`](data/meshes/)
6. [`artifacts/`](artifacts/) and [`archive/`](archive/) only for outputs, provenance, and superseded material

## Repo At A Glance

| Area | What it is | First stop |
| --- | --- | --- |
| [`src/`](src/) | Active implementation code | [`src/problems/`](src/problems/) for problem solvers, [`src/core/`](src/core/) for shared infrastructure |
| [`docs/`](docs/) | Current maintained docs | [`docs/README.md`](docs/README.md) |
| [`experiments/`](experiments/) | Benchmark launchers and post-processing | [`experiments/runners/`](experiments/runners/), [`experiments/analysis/`](experiments/analysis/) |
| [`tests/`](tests/) | Regression, contracts, smoke tests | `tests/test_*.py` |
| [`data/meshes/`](data/meshes/) | Checked-in problem inputs | family subfolders under [`data/meshes/`](data/meshes/) |
| [`artifacts/`](artifacts/) | Generated raw results, reports, repro outputs, temp probes | use only when tracing output lineage |
| [`archive/`](archive/) | Historical docs, old results, refactor/tuning trail | use when current docs point to historical provenance |

## Condensed Root Folder Map

Deliberately omitted from the map body: `.git/`, `.venv/`, `.pytest_cache/`, all `__pycache__/`, and compiled cache/build internals.

| Path | Role | Notes |
| --- | --- | --- |
| [`.claude/`](.claude/) | Local agent config | Small, tooling-specific settings |
| [`.devcontainer/`](.devcontainer/) | Devcontainer config | Dockerfile and devcontainer metadata |
| [`.vscode/`](.vscode/) | Editor config | Workspace settings |
| [`.gitignore`](.gitignore) | Ignore policy | Repo housekeeping |
| [`README.md`](README.md) | Top-level overview | Broad orientation; points readers into `docs/` |
| [`Makefile`](Makefile) | Small native build helper | Builds the committed graph-coloring shared object |
| [`src/`](src/) | Active source tree | ~164 files, primary implementation surface |
| [`docs/`](docs/) | Canonical documentation surface | ~148 files, current setup/problem/result/reference docs |
| [`experiments/`](experiments/) | Runner and analysis scripts | ~193 files, bridges source code to reproduced results |
| [`tests/`](tests/) | Test suite | 39 focused regression/contract tests |
| [`data/`](data/) | Checked-in inputs | 114 files, mostly HDF5 and mesh assets under `data/meshes/` |
| [`notebooks/`](notebooks/) | Demo and benchmark notebooks | 8 notebooks in `demos/` and `benchmarks/` |
| [`paper/`](paper/) | Publication material | ~173 files including sections, scripts, figures, tables, and build outputs |
| [`artifacts/`](artifacts/) | Generated outputs | ~7.9k files and ~3k subdirs; raw results, reports, reproduction bundles, temp probes |
| [`archive/`](archive/) | Historical material | ~2.6k files; superseded docs, old results, scratch, refactor history |
| `external/HPC_cluster_config/` | Ignored external reference checkout/index | Private HPC cluster configuration bundle; start at `hpc/agent_manifest.yaml` and `FILE_INDEX.md` for IT4I/Slurm guidance |
| [`local_env/`](local_env/) | Local toolchain/build payload | ~68k files; installed Python/prefix trees and vendored source tarballs |
| [`tmp/`](tmp/) | Transient working area | ~1.1k files, mostly comparison scratch space |
| [`scripts/`](scripts/) | Placeholder script area | Currently empty |
| [`LOCAL_FAILURE_WRITEUP_TEMPLATE.md`](LOCAL_FAILURE_WRITEUP_TEMPLATE.md) | Incident/writeup template | Root-level process note |
| [`local_env_build.sh`](local_env_build.sh) | Local environment bootstrap helper | Pairs with `local_env/` |
| [`migrate_venv.sh`](migrate_venv.sh) | Environment migration helper | Local maintenance script |
| [`thesis_replications.md`](thesis_replications.md) | Root note for thesis-related work | Supplemental context |
| [`plaplace_u3_problematic_experiment_prompt.md`](plaplace_u3_problematic_experiment_prompt.md) | One-off prompt note | Root-level task artifact |
| [`plaplace_u3_timing_enrichment_agent_prompt.md`](plaplace_u3_timing_enrichment_agent_prompt.md) | One-off prompt note | Root-level task artifact |
| [`BAI0012_FEI_P1807_1103V036_2023.pdf`](BAI0012_FEI_P1807_1103V036_2023.pdf) | External/reference PDF | Not part of active implementation flow |

## Where To Look For...

| If you need... | Go here first | Then |
| --- | --- | --- |
| Active solver implementation for a problem family | [`src/problems/`](src/problems/) | open the family folder, then `solve_*.py` entrypoints and `solver*.py` logic |
| Shared backend infrastructure | [`src/core/`](src/core/) | especially `petsc/`, `fenics/`, `serial/`, `benchmark/`, `problem_data/` |
| Current run instructions | [`docs/setup/`](docs/setup/) | then the matching problem page in [`docs/problems/`](docs/problems/) |
| Current maintained results | [`docs/results/`](docs/results/) | then `docs/assets/` for curated published figures |
| Benchmark launchers | [`experiments/runners/`](experiments/runners/) | files are mostly named `run_*.py` |
| Figure/report generation | [`experiments/analysis/`](experiments/analysis/) | files are mostly named `generate_*.py` |
| Regression and contracts | [`tests/`](tests/) | match the relevant `test_<family>*.py` file |
| Raw meshes and checked-in inputs | [`data/meshes/`](data/meshes/) | choose the family subfolder |
| Historical provenance or superseded notes | [`archive/docs/`](archive/docs/) | use [`archive/results/`](archive/results/) for old raw/report outputs |
| Publication source | [`paper/`](paper/) | `sections/`, `scripts/`, `figures/`, `tables/` |

Fast lookup rule of thumb:

1. Start in `docs/` if you want the current story.
2. Start in `src/` if you want the current implementation.
3. Start in `experiments/` if you want to reproduce or regenerate outputs.
4. Drop into `artifacts/` or `archive/` only after you know which campaign or historical trail you are following.

## Source Code Map

### `src/core/`: Shared Infrastructure

| Subtree | What lives there |
| --- | --- |
| [`src/core/benchmark/`](src/core/benchmark/) | Result normalization, replication helpers, repair utilities, state export |
| [`src/core/cli/`](src/core/cli/) | Small CLI/runtime helpers such as threading setup |
| [`src/core/coloring/`](src/core/coloring/) | Graph coloring implementations plus the committed `custom_coloring.c` / `.so` |
| [`src/core/fenics/`](src/core/fenics/) | Shared FEniCS helpers such as nullspace and scalar custom-Newton support |
| [`src/core/petsc/`](src/core/petsc/) | Parallel drivers, minimizers, trust-region KSP support, GAMG helpers, DOF partitioning, reordered element base |
| [`src/core/petsc/fenics_tools/`](src/core/petsc/fenics_tools/) | PETSc-side helpers specific to FEniCS-backed flows |
| [`src/core/petsc/jax_tools/`](src/core/petsc/jax_tools/) | PETSc-side helpers specific to JAX-backed assembly, including `parallel_assembler.py` |
| [`src/core/problem_data/`](src/core/problem_data/) | Problem-data I/O, especially HDF5 loading/writing |
| [`src/core/serial/`](src/core/serial/) | Serial minimizers, sparse solvers, graph SFD utilities, JAX autodiff helpers |

### `src/problems/`: Problem Families

Common naming conventions used across the problem packages:

| Pattern | Meaning |
| --- | --- |
| `solve_*.py` | CLI-style entrypoints, benchmark entrypoints, or maintained run front doors |
| `solver*.py` | Core nonlinear/linear solve logic |
| `jax_energy.py`, `functionals.py` | Energy definitions and problem functionals |
| `support/` | Mesh/material/helper utilities tied to that family |
| `jax_petsc/reordered_element_assembler.py` | Backend-specific JAX+PETSc element assembly path where present |
| `jax_petsc/multigrid.py` | Backend-specific multigrid policy where present |
| `scripts/solve_case.py` | Specialized per-case front door used by thesis or parameter-study style families |

| Family | Shape of the package | First places to look |
| --- | --- | --- |
| [`src/problems/plaplace/`](src/problems/plaplace/) | Classical split: `fenics/`, `jax/`, `jax_petsc/`, `support/` | `fenics/solve_pLaplace_*.py`, `jax/solve_pLaplace_jax_newton.py`, `jax_petsc/solve_pLaplace_dof.py` |
| [`src/problems/ginzburg_landau/`](src/problems/ginzburg_landau/) | Classical split: `fenics/`, `jax/`, `jax_petsc/` | `fenics/solve_GL_*.py`, `jax_petsc/solve_GL_dof.py`, then `solver.py` / `jax_energy.py` |
| [`src/problems/hyperelasticity/`](src/problems/hyperelasticity/) | `fenics/`, `jax/`, `jax_petsc/`, plus `support/` for shared mesh/boundary helpers | `fenics/solve_HE_*.py`, `jax/solve_HE_jax_newton.py`, `jax_petsc/solve_HE_dof.py` |
| [`src/problems/slope_stability/`](src/problems/slope_stability/) | 2D plasticity / slope-stability implementation: `jax/`, `jax_petsc/`, `support/` | `jax_petsc/solve_slope_stability_dof.py`, `solve_slope_stability_refined_p1_dof.py`, then `multigrid.py` and `solver.py` |
| [`src/problems/slope_stability_3d/`](src/problems/slope_stability_3d/) | 3D plasticity path with `jax/`, `jax_petsc/`, `support/` | `jax_petsc/solve_slope_stability_3d_dof.py`, `solve_slope_stability_3d_direct_continuation.py`, then `support/materials.py` / `mesh.py` |
| [`src/problems/topology/`](src/problems/topology/) | Mostly JAX-centric package: `jax/` plus `support/` | `jax/solve_topopt_jax.py`, `jax/solve_topopt_parallel.py`, then `parallel_support.py` and `support/policy.py` |
| [`src/problems/plaplace_u3/`](src/problems/plaplace_u3/) | Thesis-style specialization: root `common.py`, `support/`, and a dense [`thesis/`](src/problems/plaplace_u3/thesis/) subtree | `thesis/solve_plaplace_u3_thesis.py`, `thesis/scripts/solve_case.py`, then `solver_common.py`, `solver_mpa.py`, `solver_oa.py`, `solver_rmpa.py` |
| [`src/problems/plaplace_up_arctan/`](src/problems/plaplace_up_arctan/) | Mixed layout: important root-level math/solver modules plus `jax_petsc/`, `scripts/`, `support/` | root `workflow.py`, `solver_common.py`, `solver_mpa.py`, `solver_rmpa.py`, then `jax_petsc/solve_case.py` and `scripts/solve_case.py` |

Problem-family mental model:

1. The classical PDE families (`plaplace`, `ginzburg_landau`, `hyperelasticity`) use the cleanest backend split.
2. The plasticity families (`slope_stability`, `slope_stability_3d`) are more multigrid- and continuation-heavy.
3. The thesis/specialized families (`plaplace_u3`, `plaplace_up_arctan`) keep more math and workflow modules at package root or under `thesis/`.

## Documentation And Experiment Map

### `docs/`: Current Maintained Narrative

| Subtree | What it is |
| --- | --- |
| [`docs/README.md`](docs/README.md) | Canonical doc index; start here instead of `archive/docs/` |
| [`docs/setup/`](docs/setup/) | Build and run instructions |
| [`docs/problems/`](docs/problems/) | Problem overviews, maintained implementation pointers, curated sample results |
| [`docs/results/`](docs/results/) | Current maintained benchmark summaries and scaling reports |
| [`docs/implementation/`](docs/implementation/) | Active implementation notes that still describe current code |
| [`docs/reference/`](docs/reference/) | Cross-cutting reference notes and maintainer guidance |
| [`docs/assets/`](docs/assets/) | Curated figures used by the current docs |

### `experiments/`: Execution Pipeline

| Subtree | What it is |
| --- | --- |
| [`experiments/runners/`](experiments/runners/) | Benchmark/campaign launchers, mostly `run_*.py`; use this when you want to execute a maintained or targeted sweep |
| [`experiments/runners/barbora_he_first_step_scaling/`](experiments/runners/barbora_he_first_step_scaling/) | Barbora HyperElasticity first-step scaling bundle, including level-5 uniform mesh generation and Slurm matrix preparation |
| [`experiments/analysis/`](experiments/analysis/) | Post-processing and asset generation, mostly `generate_*.py`; use this when you want figures, tables, docs assets, or report material |

Useful naming pattern:

1. `run_<campaign>.py` usually creates or refreshes raw results under `artifacts/raw_results/`.
2. `generate_<campaign>_report.py` or `generate_<campaign>_assets.py` usually reads raw results and writes reports/assets into `artifacts/`, `docs/assets/`, or `paper/`.

### `paper/` And `notebooks/`

| Area | What it is |
| --- | --- |
| [`paper/sections/`](paper/sections/) | Paper source sections |
| [`paper/scripts/`](paper/scripts/) | Paper-specific figure/table generation and validation |
| [`paper/figures/`](paper/figures/) | Paper figures, including generated outputs |
| [`paper/tables/`](paper/tables/) | Paper tables, including generated outputs |
| [`paper/build/`](paper/build/) | Build/test outputs for publication assets |
| [`notebooks/demos/`](notebooks/demos/) | API-style demos |
| [`notebooks/benchmarks/`](notebooks/benchmarks/) | Notebook-based benchmark walkthroughs |

## Data, Outputs, And Provenance

### Inputs

| Path | What is checked in |
| --- | --- |
| [`data/meshes/GinzburgLandau/`](data/meshes/GinzburgLandau/) | HDF5 mesh hierarchy for Ginzburg-Landau |
| [`data/meshes/HyperElasticity/`](data/meshes/HyperElasticity/) | HDF5 mesh hierarchy for hyperelasticity |
| [`data/meshes/pLaplace/`](data/meshes/pLaplace/) | HDF5 mesh hierarchy for p-Laplace |
| [`data/meshes/SlopeStability/`](data/meshes/SlopeStability/) | 2D slope-stability meshes, including same-mesh variants |
| [`data/meshes/SlopeStability3D/`](data/meshes/SlopeStability3D/) | 3D source meshes and definitions |

### Generated And Historical Areas

| Path | Use it for | Do not treat it as |
| --- | --- | --- |
| [`artifacts/raw_results/`](artifacts/raw_results/) | Raw JSON/log/markdown outputs from experiment runs | source code |
| [`artifacts/reports/`](artifacts/reports/) | Generated report bundles and campaign summaries | canonical docs |
| [`artifacts/reproduction/`](artifacts/reproduction/) | Reproduction snapshots/resync bundles | primary implementation |
| [`artifacts/cache/`](artifacts/cache/) | Intermediate caches | maintained assets |
| `artifacts/tmp*` | Temporary probes, validation, diagnostics, scratch | stable interface or long-term truth |
| [`archive/docs/`](archive/docs/) | Superseded docs, refactor notes, tuning trail, historical benchmark pages | current maintained docs |
| [`archive/results/`](archive/results/) | Historical raw/report outputs and past campaigns | current maintained results |
| [`archive/scratch/`](archive/scratch/) | Scratch work | reproducible source-of-truth |
| [`local_env/`](local_env/) | Local build/install payload: `prefix/`, `python/`, `src/` | project source tree |
| [`tmp/`](tmp/) | Transient comparison workspace, especially `tmp/source_compare/` | maintained implementation |

## Practical Navigation Recipes

| Goal | Shortest path |
| --- | --- |
| Understand the current implementation for one problem | `docs/problems/<family>.md` -> `src/problems/<family>/solve_*.py` -> `solver*.py` -> shared pieces in `src/core/` |
| Reproduce a maintained benchmark or report | `docs/results/<family>.md` -> matching `experiments/runners/run_*.py` -> matching `experiments/analysis/generate_*.py` |
| Inspect backend mechanics | `src/core/petsc/` or `src/core/fenics/` -> family-specific `jax_petsc/` or `fenics/` subtree |
| Check whether behavior is already covered | `tests/test_<family>*.py` plus nearby contract-style tests |
| Trace where a published figure came from | `docs/assets/` or `paper/figures/` -> matching generator in `experiments/analysis/` or `paper/scripts/` -> raw inputs under `artifacts/raw_results/` |
| Find historical rationale for a current default | `docs/implementation/` or `docs/reference/` first -> `archive/docs/tuning/` or `archive/docs/refactor/` only if needed |
| Prepare HPC/IT4I jobs or Slurm scripts | `external/HPC_cluster_config/hpc/agent_manifest.yaml` and `external/HPC_cluster_config/FILE_INDEX.md` -> `hpc/providers/it4i/README.md` -> provider guide, resource summaries, and templates |

## Mental Model In One Paragraph

Treat the repository as a layered stack: `src/` holds the live solvers and shared infrastructure, `docs/` explains the current supported story, `experiments/` turns code into reproducible campaigns, `tests/` pin down behavior, `data/meshes/` provides checked-in inputs, `artifacts/` stores generated outputs, and `archive/` preserves how the project got here. If you are unsure where to start, begin with the relevant page in `docs/problems/`, then move into the corresponding `src/problems/<family>/` package, and only after that follow the trail into `experiments/`, `artifacts/`, or `archive/`.
