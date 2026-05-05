# Barbora HyperElasticity First-Step Scaling Prep

This directory prepares, but does not submit by itself, a Barbora CPU scaling
campaign for the first HyperElasticity load step.

Default case:

- solver backend: JAX+PETSc element (`--backend element`)
- mesh: HyperElasticity uniform-refinement level `5`
- load: first load step only (`--steps 1 --start-step 1 --total-steps 24`)
- nonlinear/linear settings: maintained STCG + GAMG trust-region profile from
  `docs/problems/HyperElasticity.md`
- nonlinear stabilization: Armijo subproblem line search, initial trust
  radius `1.0`, and stricter step convergence
  (`HE_LINE_SEARCH=armijo`, `HE_TRUST_RADIUS_INIT=1.0`, `HE_TOLX_REL=1e-4`)
- problem build: rank-local HDF5 reads with point-to-point overlap exchange
  (`HE_PROBLEM_BUILD_MODE=rank_local`,
  `HE_DISTRIBUTION_STRATEGY=overlap_p2p`,
  `HE_ASSEMBLY_BACKEND=coo_local`)
- cluster: Barbora CPU, account `fta-26-40`, QoS `3571_6324`, partition `qcpu`

## Local Sizing Evidence

Local sizing outputs were written under
`artifacts/raw_results/example_runs/he_first_step_local_sizing_20260503_0540/`.
The level-5 mesh generation manifest is under
`artifacts/raw_results/example_runs/he_level5_mesh_generation_20260503_085556/`.
The level-5 first-step run is under
`artifacts/raw_results/example_runs/he_level5_first_step_20260503_085556/`.
`data/meshes/HyperElasticity/HyperElasticity_level5.h5` is tracked with Git
LFS because the generated HDF5 is larger than GitHub's regular blob limit.

Observed on this workstation on 2026-05-03:

| level | MPI ranks | total DOFs | free DOFs | setup [s] | first-step solve [s] | shell wall [s] | Newton | linear | result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 2,187 | 2,133 | 0.514 | 0.291 | about 2.9 | 28 | 314 | converged |
| 2 | 16 | 12,075 | 11,925 | 0.563 | 0.478 | 4 | 23 | 336 | converged |
| 3 | 16 | 78,003 | 77,517 | 0.886 | 1.957 | 6 | 23 | 350 | converged |
| 4 | 16 | 555,747 | 554,013 | 2.836 | 21.898 | 28 | 21 | 407 | converged |
| 5 | 16 | 4,185,027 | 4,178,493 | 18.824 | 313.519 | 341 | 30 | 489 | converged |

The prepared Barbora default therefore uses level `5`: it is the largest
available uniform-refinement case and locally completed the first step with 16
MPI ranks in under the requested 10-minute guard.

`generate_he_uniform_mesh.py` validates the inferred level-1--4 pattern and can
regenerate missing uniform-refinement levels. The generated level-5 HDF5 uses
the same structured beam discretization, six-tet brick split, end-face
Dirichlet-free-DOF exclusion, derivative arrays, and vector-DOF adjacency
schema as the checked-in lower levels.

## Prepared Barbora Matrix

`submit_matrix.sh` prepares one Slurm submission per matrix point:

| variable | default |
| --- | --- |
| mesh level | `5` |
| nodes | `1 2 4 8 16` |
| MPI ranks per socket | `4 8 12 18` |
| MPI ranks per node | `8 16 24 36` |
| CPUs per task | `1` |
| time limit | `00:20:00` |
| estimated node-hours | `41.33` for the default one-backend matrix |

Barbora CPU nodes have two sockets and 18 cores per socket. The rank density is
controlled by `RPS_LIST`; the script passes `--ntasks-per-socket` to Slurm and
uses `srun --cpu-bind=cores --distribution=block:block` inside each job.

The scripts intentionally do not use `#SBATCH --exclusive` or `#SBATCH --mem=`.

The element scripts default to the distributed HyperElasticity path. To repeat
the old replicated baseline for a small diagnostic only, override:

```bash
HE_PROBLEM_BUILD_MODE=replicated \
HE_DISTRIBUTION_STRATEGY=overlap_allgather \
HE_ASSEMBLY_BACKEND=coo \
HE_LINE_SEARCH=golden_fixed \
HE_TRUST_RADIUS_INIT=0.5 \
HE_TOLX_REL=1e-3 \
bash experiments/runners/barbora_he_first_step_scaling/submit_matrix.sh
```

## Two-Node Full-Rank Smoke Test

`submit_two_node_full_rank_10min.sh` prepares exactly one capped level-5 test:

| variable | value |
| --- | --- |
| nodes | `2` |
| MPI ranks per socket | `18` |
| MPI ranks per node | `36` |
| total MPI ranks | `72` |
| CPUs per task | `1` |
| time limit | `00:10:00` |
| estimated node-hours | `0.3333` |

Preview the single job without submitting:

```bash
DRY_RUN=1 bash experiments/runners/barbora_he_first_step_scaling/submit_two_node_full_rank_10min.sh
```

Ask Slurm to validate the request without submitting a real job:

```bash
SBATCH_TEST_ONLY=1 bash experiments/runners/barbora_he_first_step_scaling/submit_two_node_full_rank_10min.sh
```

Submit the capped two-node test:

```bash
bash experiments/runners/barbora_he_first_step_scaling/submit_two_node_full_rank_10min.sh
```

## Build The Barbora PETSc 3.24 Environment

The HyperElasticity element path expects PETSc/petsc4py `3.24.2`; Barbora's
system `PETSc/3.21.2-foss-2022b` module is too old for the current COO
preallocation path. Build the experiment environment in the repository clone:

```bash
# Preview the build job without submitting it.
DRY_RUN=1 bash experiments/runners/barbora_he_first_step_scaling/submit_build_barbora_petsc_env.sh

# Optional Slurm admission check.
SBATCH_TEST_ONLY=1 bash experiments/runners/barbora_he_first_step_scaling/submit_build_barbora_petsc_env.sh

# Submit the build job.
bash experiments/runners/barbora_he_first_step_scaling/submit_build_barbora_petsc_env.sh
```

The build job creates:

```text
.venv/
local_env/prefix/
local_env/src/
experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh
```

It loads the Barbora `foss/2022b` toolchain, builds PETSc `3.24.2` into
`local_env/prefix`, installs `petsc4py 3.24.2`, and installs only the Python
packages needed by the current JAX+PETSc runner: `numpy`, `scipy`, `h5py`,
`mpi4py`, and `jax[cpu]`. PETSc downloads its own CMake during configure
because the prepared external packages require a newer CMake than Barbora's
`CMake/3.24.3` module. The default Python pins are `numpy 1.26.4`,
`scipy 1.11.4`, `h5py 3.10.0`, `mpi4py 4.1.1`, and `jax[cpu] 0.4.30`;
override `BARBORA_PYTHON_PACKAGES` or `BARBORA_MPI4PY_SPEC` if needed. The
PETSc configure uses `mpicc`, `mpicxx`, and `mpifort` by default; override
`BARBORA_PETSC_CC`, `BARBORA_PETSC_CXX`, or `BARBORA_PETSC_FC` only if the
cluster wrapper names change.

After the build finishes, verify the runtime stack:

```bash
export HE_ENV_SETUP="$PWD/experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh"
bash experiments/runners/barbora_he_first_step_scaling/check_barbora_env.sh
```

The check verifies the imports, PETSc/petsc4py version `3.24.2`, and the
presence of `PETSc.Mat.setPreallocationCOO`.

## One-Node qcpu_exp Smoke Test

After the PETSc `3.24.2` environment is built and checked, this wrapper submits
one small admission/runtime smoke case: level `4`, one `qcpu_exp` node, full CPU
rank population (`36` MPI ranks), and a one-minute wall cap.

```bash
DRY_RUN=1 bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_1min_qexp.sh
bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_1min_qexp.sh
```

The wrapper enables single-node OpenMPI shared-memory transport for the smoke
case only; the multi-node level-5 scripts use the normal Barbora transport.

## One-Node Socket Scaling

`submit_level4_one_node_socket_scaling.sh` submits one independent Slurm job per
valid level-4 socket-layout case on `qcpu_exp`. Layouts are written as
`socket0+socket1` MPI ranks. Barbora CPU sockets have 18 cores, so `36+0` is
recorded as invalid and is not submitted.

| layout | total ranks | active sockets | solver cap [s] | Slurm wall |
| --- | ---: | ---: | ---: | --- |
| `18+18` | 36 | 2 | 35 | `00:01:35` |
| `18+0` | 18 | 1 | 140 | `00:03:20` |
| `9+9` | 18 | 2 | 70 | `00:02:10` |
| `9+0` | 9 | 1 | 280 | `00:05:40` |

The solver cap is passed as `--step-time-limit-s`; the Slurm wall limit adds a
60-second startup/output allowance. The submitter still requests Slurm socket
placement, and the job step records an explicit `map_cpu` binding so sparse
balanced layouts such as `9+9` can be audited from `binding.txt`. For sparse
balanced layouts, the job step may allocate two CPUs per task to expose both
sockets to Slurm's step cpuset while keeping `OMP_NUM_THREADS=1`,
`--nproc-threads 1`, and one mapped CPU per MPI rank. Preview, validate, and
submit:

```bash
DRY_RUN=1 bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_socket_scaling.sh
SBATCH_TEST_ONLY=1 bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_socket_scaling.sh
bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_socket_scaling.sh
```

## Run Workflow On Barbora

From a full clone of this repository on Barbora:

```bash
ssh ber0061@barbora.it4i.cz
cd /path/to/fenics_nonlinear_energies

# If the earlier PETSc-3.21 smoke debugging left local edits, restore them
# before pulling the PETSc-3.24 environment scripts.
git restore src/core/petsc/reordered_element_base.py tests/test_reordered_element_base.py
rm -f experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh \
      experiments/runners/barbora_he_first_step_scaling/env_barbora.smoke_local.sh

# If the old two-node PETSc-3.21 job is still pending, cancel it before it runs.
# Example old job id from the first attempt:
# scancel 1970434

git pull --ff-only origin main
git lfs pull

# Build and check the local PETSc 3.24.2 environment first.
bash experiments/runners/barbora_he_first_step_scaling/submit_build_barbora_petsc_env.sh
# wait for the build job to complete, then:
export HE_ENV_SETUP="$PWD/experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh"
bash experiments/runners/barbora_he_first_step_scaling/check_barbora_env.sh

# Confirms the generated mesh recipe still matches checked-in levels 1--4.
"$PYTHON" experiments/runners/barbora_he_first_step_scaling/generate_he_uniform_mesh.py \
  --validate-only

# Quick post-build runtime smoke on qcpu_exp.
bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_1min_qexp.sh

# One-node level-4 socket-layout scaling on qcpu_exp.
bash experiments/runners/barbora_he_first_step_scaling/submit_level4_one_node_socket_scaling.sh

# Preview commands and write the campaign plan, without submitting.
DRY_RUN=1 bash experiments/runners/barbora_he_first_step_scaling/submit_matrix.sh

# Optional Slurm admission check without submitting real jobs.
SBATCH_TEST_ONLY=1 bash experiments/runners/barbora_he_first_step_scaling/submit_matrix.sh

# Submit the default level-5 first-step matrix.
bash experiments/runners/barbora_he_first_step_scaling/submit_matrix.sh

# Submit only the capped two-node full-rank level-5 test.
bash experiments/runners/barbora_he_first_step_scaling/submit_two_node_full_rank_10min.sh

exit
```

The submitter prints the campaign output root. After jobs finish:

```bash
./.venv/bin/python experiments/runners/barbora_he_first_step_scaling/summarize_results.py \
  artifacts/raw_results/barbora/he_first_step_scaling/<campaign>
```

Outputs are placed under:

```text
artifacts/raw_results/barbora/he_first_step_scaling/<campaign>/
  campaign_plan.csv
  submitted_jobs.txt
  slurm/
  cases/
  summary/results_summary.csv
  summary/results_summary.md
```

## Useful Overrides

All overrides are environment variables:

```bash
HE_LEVEL=4                         # choose a smaller uniform-refinement level
RPS_LIST="8 18"                    # subset ranks-per-socket cases
NODES_LIST="1 2 4"                 # subset node counts
TIME_LIMIT=00:10:00                # shorter first-step wall limit
MAX_NODE_HOURS=100                 # guard before submitting
BACKENDS="element"                 # optionally: "element fenics"
HE_LINE_SEARCH=golden_fixed        # reproduce the old golden-section setting
HE_TRUST_RADIUS_INIT=0.5           # reproduce the old conservative radius
HE_TOLX_REL=1e-3                   # reproduce the old step tolerance
PYTHON=./.venv/bin/python          # Python executable in the full clone
CAMPAIGN=my_he_first_step_campaign # stable output folder name
```
