# Barbora HyperElasticity First-Step Scaling Prep

This directory prepares, but does not submit by itself, a Barbora CPU scaling
campaign for the first HyperElasticity load step.

Default case:

- solver backend: JAX+PETSc element (`--backend element`)
- mesh: HyperElasticity uniform-refinement level `5`
- load: first load step only (`--steps 1 --start-step 1 --total-steps 24`)
- nonlinear/linear settings: maintained STCG + GAMG trust-region profile from
  `docs/problems/HyperElasticity.md`
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

## Run Workflow On Barbora

From a full clone of this repository on Barbora:

```bash
ssh ber0061@barbora.it4i.cz
cd /path/to/fenics_nonlinear_energies
git pull --ff-only origin main
git lfs pull

# Confirms the generated mesh recipe still matches checked-in levels 1--4.
./.venv/bin/python experiments/runners/barbora_he_first_step_scaling/generate_he_uniform_mesh.py \
  --validate-only

# Optional: define modules/environment in a separate file.
cp experiments/runners/barbora_he_first_step_scaling/env_barbora.example.sh \
  experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh
# edit env_barbora.local.sh if needed
export HE_ENV_SETUP="$PWD/experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh"

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
PYTHON=./.venv/bin/python          # Python executable in the full clone
CAMPAIGN=my_he_first_step_campaign # stable output folder name
```
