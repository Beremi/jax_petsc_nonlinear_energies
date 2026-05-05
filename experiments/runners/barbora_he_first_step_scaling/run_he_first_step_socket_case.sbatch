#!/usr/bin/env bash
# Barbora HyperElasticity first-step case for one-node socket-layout scaling.

#SBATCH --job-name=he_socket_case
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6324
#SBATCH --partition=qcpu_exp
#SBATCH --nodes=1
#SBATCH --time=00:05:40

set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: $0 OUT_ROOT HE_LEVEL LAYOUT SOCKET0_RANKS SOCKET1_RANKS ACTIVE_SOCKETS TOTAL_RANKS STEP_TIME_LIMIT_S TOTAL_STEPS BACKEND [SLURM_TIME_LIMIT] [CPU_MAP]" >&2
  exit 2
fi

OUT_ROOT="$1"
HE_LEVEL="$2"
SOCKET_LAYOUT="$3"
SOCKET0_RANKS="$4"
SOCKET1_RANKS="$5"
ACTIVE_SOCKETS="$6"
TOTAL_RANKS_EXPECTED="$7"
STEP_TIME_LIMIT_S="$8"
TOTAL_STEPS="$9"
BACKEND="${10}"
SLURM_TIME_LIMIT_REQUESTED="${11:-${SLURM_TIMELIMIT:-}}"
CPU_MAP="${12:-}"
HE_PROBLEM_BUILD_MODE="${HE_PROBLEM_BUILD_MODE:-rank_local}"
HE_DISTRIBUTION_STRATEGY="${HE_DISTRIBUTION_STRATEGY:-overlap_p2p}"
HE_ASSEMBLY_BACKEND="${HE_ASSEMBLY_BACKEND:-coo_local}"
HE_LINE_SEARCH="${HE_LINE_SEARCH:-armijo}"
HE_TRUST_RADIUS_INIT="${HE_TRUST_RADIUS_INIT:-1.0}"
HE_TOLX_REL="${HE_TOLX_REL:-1e-4}"

case "$BACKEND" in
  element|fenics) ;;
  *)
    echo "Unsupported backend '$BACKEND'; expected 'element' or 'fenics'." >&2
    exit 2
    ;;
esac

case "$HE_LINE_SEARCH" in
  golden_fixed|armijo) ;;
  *)
    echo "Unsupported HE_LINE_SEARCH '$HE_LINE_SEARCH'; expected 'golden_fixed' or 'armijo'." >&2
    exit 2
    ;;
esac

if [[ "${SLURM_JOB_NUM_NODES:?}" != "1" ]]; then
  echo "Socket scaling expects exactly one node, got ${SLURM_JOB_NUM_NODES}." >&2
  exit 2
fi

if [[ "${SLURM_CPUS_PER_TASK:-1}" != "1" ]]; then
  echo "Socket scaling expects --cpus-per-task=1, got ${SLURM_CPUS_PER_TASK}." >&2
  exit 2
fi

if [[ "${SLURM_NTASKS:?}" != "$TOTAL_RANKS_EXPECTED" ]]; then
  echo "Expected ${TOTAL_RANKS_EXPECTED} MPI ranks, got SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi

if [[ -z "$CPU_MAP" ]]; then
  echo "Socket scaling requires an explicit Slurm map_cpu binding list." >&2
  exit 2
fi
IFS=, read -r -a CPU_MAP_ENTRIES <<< "$CPU_MAP"
if (( ${#CPU_MAP_ENTRIES[@]} != TOTAL_RANKS_EXPECTED )); then
  echo "CPU map '$CPU_MAP' has ${#CPU_MAP_ENTRIES[@]} entries; expected ${TOTAL_RANKS_EXPECTED}." >&2
  exit 2
fi

SRUN_STEP_CPUS_PER_TASK=1
if (( ACTIVE_SOCKETS == 2 && TOTAL_RANKS_EXPECTED < 36 )); then
  # Barbora's Slurm step allocator otherwise gives an 18-rank sparse balanced
  # step only CPUs 0-17, even though the job allocation reserves both sockets.
  SRUN_STEP_CPUS_PER_TASK=2
fi

SRUN_BIND_ARGS=(
  --cpus-per-task="$SRUN_STEP_CPUS_PER_TASK"
  --cpu-bind="map_cpu:${CPU_MAP}"
  --distribution=block:block
)
SRUN_BIND_VERBOSE_ARGS=(
  --cpus-per-task="$SRUN_STEP_CPUS_PER_TASK"
  --cpu-bind="verbose,map_cpu:${CPU_MAP}"
  --distribution=block:block
)
SRUN_PLACEMENT_ARGS=(--sockets-per-node="$ACTIVE_SOCKETS")
if (( ACTIVE_SOCKETS == 2 )); then
  SRUN_PLACEMENT_ARGS+=(--ntasks-per-socket="$SOCKET0_RANKS")
else
  SRUN_PLACEMENT_ARGS+=(--ntasks-per-socket="$TOTAL_RANKS_EXPECTED")
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1}"
export HE_SINGLE_NODE_SMOKE_TRANSPORT="${HE_SINGLE_NODE_SMOKE_TRANSPORT:-1}"

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
  exit 2
fi

CASE_ID="he_${BACKEND}_l${HE_LEVEL}_socket_${SOCKET_LAYOUT}_np${TOTAL_RANKS_EXPECTED}"
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
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "nodelist=${SLURM_JOB_NODELIST:-}"
  echo "socket_layout=$SOCKET_LAYOUT"
  echo "socket0_ranks=$SOCKET0_RANKS"
  echo "socket1_ranks=$SOCKET1_RANKS"
  echo "active_sockets=$ACTIVE_SOCKETS"
  echo "total_ranks=$TOTAL_RANKS_EXPECTED"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
  echo "srun_step_cpus_per_task=$SRUN_STEP_CPUS_PER_TASK"
  echo "cpu_map=$CPU_MAP"
  echo "step_time_limit_s=$STEP_TIME_LIMIT_S"
  echo "slurm_time_limit=${SLURM_TIME_LIMIT_REQUESTED}"
  echo "he_level=$HE_LEVEL"
  echo "total_steps=$TOTAL_STEPS"
  echo "backend=$BACKEND"
  echo "problem_build_mode=$HE_PROBLEM_BUILD_MODE"
  echo "distribution_strategy=$HE_DISTRIBUTION_STRATEGY"
  echo "assembly_backend=$HE_ASSEMBLY_BACKEND"
  echo "line_search=$HE_LINE_SEARCH"
  echo "trust_radius_init=$HE_TRUST_RADIUS_INIT"
  echo "tolx_rel=$HE_TOLX_REL"
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
  "nodes": ${SLURM_JOB_NUM_NODES},
  "socket_layout": "${SOCKET_LAYOUT}",
  "socket0_ranks": ${SOCKET0_RANKS},
  "socket1_ranks": ${SOCKET1_RANKS},
  "active_sockets": ${ACTIVE_SOCKETS},
  "total_ranks": ${TOTAL_RANKS_EXPECTED},
  "cpus_per_task": ${SLURM_CPUS_PER_TASK:-1},
  "srun_step_cpus_per_task": ${SRUN_STEP_CPUS_PER_TASK},
  "cpu_map": "${CPU_MAP}",
  "step_time_limit_s": ${STEP_TIME_LIMIT_S},
  "slurm_time_limit": "${SLURM_TIME_LIMIT_REQUESTED}",
  "problem_build_mode": "${HE_PROBLEM_BUILD_MODE}",
  "distribution_strategy": "${HE_DISTRIBUTION_STRATEGY}",
  "assembly_backend": "${HE_ASSEMBLY_BACKEND}",
  "line_search": "${HE_LINE_SEARCH}",
  "trust_radius_init": "${HE_TRUST_RADIUS_INIT}",
  "tolx_rel": "${HE_TOLX_REL}",
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

srun -n "$TOTAL_RANKS_EXPECTED" "${SRUN_PLACEMENT_ARGS[@]}" "${SRUN_BIND_ARGS[@]}" hostname \
  | sort | uniq -c > "$CASE_DIR/rank_hosts.txt"

srun -n "$TOTAL_RANKS_EXPECTED" "${SRUN_PLACEMENT_ARGS[@]}" "${SRUN_BIND_VERBOSE_ARGS[@]}" \
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
  --tolx-rel "$HE_TOLX_REL"
  --tolx-abs 1e-10
  --maxit 100
  --step-time-limit-s "$STEP_TIME_LIMIT_S"
  --linesearch-a -0.5
  --linesearch-b 2.0
  --linesearch-tol 1e-1
  --line-search "$HE_LINE_SEARCH"
  --use-trust-region
  --trust-radius-init "$HE_TRUST_RADIUS_INIT"
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
  --problem-build-mode "$HE_PROBLEM_BUILD_MODE"
  --distribution-strategy "$HE_DISTRIBUTION_STRATEGY"
  --assembly-backend "$HE_ASSEMBLY_BACKEND"
  --local-coloring
  --out "$RESULT_JSON"
)

printf "%q " srun -n "$TOTAL_RANKS_EXPECTED" "${SRUN_PLACEMENT_ARGS[@]}" "${SRUN_BIND_ARGS[@]}" "${COMMAND[@]}" \
  > "$CASE_DIR/command.txt"
printf "\n" >> "$CASE_DIR/command.txt"

srun -n "$TOTAL_RANKS_EXPECTED" "${SRUN_PLACEMENT_ARGS[@]}" "${SRUN_BIND_ARGS[@]}" "${COMMAND[@]}"
