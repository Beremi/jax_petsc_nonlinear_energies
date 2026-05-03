#!/usr/bin/env bash
# Generic Barbora HyperElasticity first-step case.
#
# Resource shape is supplied by submit_matrix.sh through sbatch command-line
# options so each matrix point can request a different node count and rank
# density.

#SBATCH --job-name=he_first_step_case
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6324
#SBATCH --partition=qcpu
#SBATCH --time=00:20:00

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 OUT_ROOT HE_LEVEL RANKS_PER_SOCKET RANKS_PER_NODE TOTAL_STEPS BACKEND" >&2
  exit 2
fi

OUT_ROOT="$1"
HE_LEVEL="$2"
RANKS_PER_SOCKET="$3"
RANKS_PER_NODE="$4"
TOTAL_STEPS="$5"
BACKEND="$6"

case "$BACKEND" in
  element|fenics) ;;
  *)
    echo "Unsupported backend '$BACKEND'; expected 'element' or 'fenics'." >&2
    exit 2
    ;;
esac

if [[ "${SLURM_CPUS_PER_TASK:-1}" != "1" ]]; then
  echo "This first-step matrix expects --cpus-per-task=1, got ${SLURM_CPUS_PER_TASK}." >&2
  exit 2
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1}"

if [[ -n "${HE_ENV_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "$HE_ENV_SETUP"
elif [[ -f "experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh"
fi

PYTHON="${PYTHON:-./.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable '$PYTHON' is not available. Set PYTHON or create .venv in the full clone." >&2
  exit 2
fi

MESH_PATH="data/meshes/HyperElasticity/HyperElasticity_level${HE_LEVEL}.h5"
if [[ ! -f "$MESH_PATH" ]]; then
  echo "Required HyperElasticity mesh '$MESH_PATH' is missing." >&2
  echo "Generate it before submitting, for example:" >&2
  echo "  $PYTHON experiments/runners/barbora_he_first_step_scaling/generate_he_uniform_mesh.py --level $HE_LEVEL --validate-existing" >&2
  exit 2
fi

TOTAL_RANKS="${SLURM_NTASKS:?}"
NODES="${SLURM_JOB_NUM_NODES:?}"
CASE_ID="he_${BACKEND}_l${HE_LEVEL}_step1_n${NODES}_rps${RANKS_PER_SOCKET}_rpn${RANKS_PER_NODE}_np${TOTAL_RANKS}"
CASE_DIR="${OUT_ROOT}/cases/${CASE_ID}/job_${SLURM_JOB_ID}"
mkdir -p "$CASE_DIR"

cleanup() {
  set +e
  sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName,State,Elapsed,AllocNodes,AllocCPUS,MaxRSS,ExitCode -P \
    > "$CASE_DIR/sacct_${SLURM_JOB_ID}.txt" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "case_id=$CASE_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
  echo "pwd=$PWD"
  echo "nodes=$NODES"
  echo "nodelist=${SLURM_JOB_NODELIST:-}"
  echo "total_ranks=$TOTAL_RANKS"
  echo "ranks_per_node=$RANKS_PER_NODE"
  echo "ranks_per_socket=$RANKS_PER_SOCKET"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
  echo "he_level=$HE_LEVEL"
  echo "total_steps=$TOTAL_STEPS"
  echo "backend=$BACKEND"
  echo "python=$PYTHON"
  echo "timestamp=$(date --iso-8601=seconds)"
} | tee "$CASE_DIR/run_metadata.env"

cat > "$CASE_DIR/case_metadata.json" <<JSON
{
  "case_id": "${CASE_ID}",
  "job_id": "${SLURM_JOB_ID}",
  "backend": "${BACKEND}",
  "he_level": ${HE_LEVEL},
  "first_step": 1,
  "total_steps": ${TOTAL_STEPS},
  "nodes": ${NODES},
  "total_ranks": ${TOTAL_RANKS},
  "ranks_per_node": ${RANKS_PER_NODE},
  "ranks_per_socket": ${RANKS_PER_SOCKET},
  "cpus_per_task": ${SLURM_CPUS_PER_TASK:-1},
  "out_root": "${OUT_ROOT}"
}
JSON

{
  echo "=== git ==="
  git rev-parse HEAD || true
  git status --short || true
  echo "=== python ==="
  "$PYTHON" --version || true
  echo "=== modules ==="
  if type module >/dev/null 2>&1; then
    module list 2>&1 || true
  else
    echo "module command not available"
  fi
  echo "=== environment ==="
  env | sort
} > "$CASE_DIR/environment.txt" 2>&1

srun -n "$TOTAL_RANKS" --cpu-bind=cores --distribution=block:block hostname \
  | sort | uniq -c > "$CASE_DIR/rank_hosts.txt"

srun -n "$TOTAL_RANKS" --cpu-bind=verbose --distribution=block:block \
  bash -lc 'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} host=$(hostname) cpuset=$(taskset -pc $$)"' \
  > "$CASE_DIR/binding.txt" 2>&1 || true

RESULT_JSON="$CASE_DIR/output.json"
COMMAND=(
  "$PYTHON" -u experiments/runners/run_trust_region_case.py
  --problem he
  --backend "$BACKEND"
  --level "$HE_LEVEL"
  --steps 1
  --start-step 1
  --total-steps "$TOTAL_STEPS"
  --profile performance
  --ksp-type stcg
  --pc-type gamg
  --ksp-rtol 1e-1
  --ksp-max-it 30
  --gamg-threshold 0.05
  --gamg-agg-nsmooths 1
  --gamg-set-coordinates
  --use-near-nullspace
  --no-pc-setup-on-ksp-cap
  --tolf 1e-4
  --tolg 1e-3
  --tolg-rel 1e-3
  --tolx-rel 1e-3
  --tolx-abs 1e-10
  --maxit 100
  --linesearch-a -0.5
  --linesearch-b 2.0
  --linesearch-tol 1e-1
  --use-trust-region
  --trust-radius-init 0.5
  --trust-radius-min 1e-8
  --trust-radius-max 1e6
  --trust-shrink 0.5
  --trust-expand 1.5
  --trust-eta-shrink 0.05
  --trust-eta-expand 0.75
  --trust-max-reject 6
  --trust-subproblem-line-search
  --save-history
  --save-linear-timing
  --quiet
  --nproc-threads 1
  --element-reorder-mode block_xyz
  --local-hessian-mode element
  --local-coloring
  --out "$RESULT_JSON"
)

printf "%q " srun -n "$TOTAL_RANKS" --cpu-bind=cores --distribution=block:block "${COMMAND[@]}" \
  > "$CASE_DIR/command.txt"
printf "\n" >> "$CASE_DIR/command.txt"

srun -n "$TOTAL_RANKS" --cpu-bind=cores --distribution=block:block "${COMMAND[@]}"
