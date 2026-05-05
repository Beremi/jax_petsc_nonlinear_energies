#!/usr/bin/env bash
# Karolina HyperElasticity level-5 PMG coarse-solver candidate case.
#
# Resource shape and PMG settings are supplied by
# submit_karolina_level5_8node_pmg_candidates.sh.  The runner records and
# validates rank-to-host placement before starting the solver because the
# redundant coarse-solve candidates assume block rank groups by physical node.

#SBATCH --job-name=he5_pmg_candidate
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu
#SBATCH --nodes=8
#SBATCH --time=00:05:00

set -euo pipefail

if [[ $# -lt 14 ]]; then
  echo "usage: $0 OUT_ROOT CANDIDATE HE_LEVEL EXPECTED_NODES RANKS_PER_NODE TOTAL_RANKS TOTAL_STEPS STEP_TIME_LIMIT_S COARSE_LEVEL COARSE_PC REDUNDANT_NUMBER TELESCOPE_REDUCTION FACTOR_SOLVER WALL_TIME" >&2
  exit 2
fi

OUT_ROOT="$1"
CANDIDATE="$2"
HE_LEVEL="$3"
EXPECTED_NODES="$4"
RANKS_PER_NODE="$5"
TOTAL_RANKS_EXPECTED="$6"
TOTAL_STEPS="$7"
STEP_TIME_LIMIT_S="$8"
COARSE_LEVEL="$9"
COARSE_PC="${10}"
REDUNDANT_NUMBER="${11}"
TELESCOPE_REDUCTION="${12}"
FACTOR_SOLVER="${13}"
WALL_TIME="${14}"

HE_PROBLEM_BUILD_MODE="${HE_PROBLEM_BUILD_MODE:-rank_local}"
HE_MESH_SOURCE="${HE_MESH_SOURCE:-procedural}"
HE_DISTRIBUTION_STRATEGY="${HE_DISTRIBUTION_STRATEGY:-overlap_p2p}"
HE_ASSEMBLY_BACKEND="${HE_ASSEMBLY_BACKEND:-coo_local}"
HE_LINE_SEARCH="${HE_LINE_SEARCH:-armijo}"
HE_TRUST_RADIUS_INIT="${HE_TRUST_RADIUS_INIT:-1.0}"
HE_TOLX_REL="${HE_TOLX_REL:-1e-4}"
VALIDATE_NODE_CONTIGUITY="${VALIDATE_NODE_CONTIGUITY:-1}"

case "$COARSE_PC" in
  redundant|telescope|lu|hypre) ;;
  *)
    echo "Unsupported PMG coarse PC '$COARSE_PC'." >&2
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

if [[ "${SLURM_JOB_NUM_NODES:?}" != "$EXPECTED_NODES" ]]; then
  echo "Expected ${EXPECTED_NODES} nodes, got ${SLURM_JOB_NUM_NODES}." >&2
  exit 2
fi

if [[ "${SLURM_NTASKS:?}" != "$TOTAL_RANKS_EXPECTED" ]]; then
  echo "Expected ${TOTAL_RANKS_EXPECTED} MPI ranks, got SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi

if [[ "${SLURM_CPUS_PER_TASK:-1}" != "1" ]]; then
  echo "This PMG campaign expects --cpus-per-task=1, got ${SLURM_CPUS_PER_TASK}." >&2
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
if [[ "$HE_MESH_SOURCE" == "hdf5" && ! -f "$MESH_PATH" ]]; then
  echo "Required HyperElasticity mesh '$MESH_PATH' is missing." >&2
  exit 2
fi

CASE_ID="karolina_he_element_l${HE_LEVEL}_${CANDIDATE}_n${EXPECTED_NODES}_np${TOTAL_RANKS_EXPECTED}"
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
  echo "candidate=$CANDIDATE"
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
  echo "pwd=$PWD"
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "nodelist=${SLURM_JOB_NODELIST:-}"
  echo "total_ranks=$TOTAL_RANKS_EXPECTED"
  echo "ranks_per_node=$RANKS_PER_NODE"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
  echo "he_level=$HE_LEVEL"
  echo "total_steps=$TOTAL_STEPS"
  echo "step_time_limit_s=$STEP_TIME_LIMIT_S"
  echo "wall_time=$WALL_TIME"
  echo "backend=element"
  echo "problem_build_mode=$HE_PROBLEM_BUILD_MODE"
  echo "mesh_source=$HE_MESH_SOURCE"
  echo "distribution_strategy=$HE_DISTRIBUTION_STRATEGY"
  echo "assembly_backend=$HE_ASSEMBLY_BACKEND"
  echo "line_search=$HE_LINE_SEARCH"
  echo "trust_radius_init=$HE_TRUST_RADIUS_INIT"
  echo "tolx_rel=$HE_TOLX_REL"
  echo "pmg_coarsest_level=$COARSE_LEVEL"
  echo "pmg_coarse_pc_type=$COARSE_PC"
  echo "pmg_coarse_redundant_number=$REDUNDANT_NUMBER"
  echo "pmg_coarse_telescope_reduction_factor=$TELESCOPE_REDUCTION"
  echo "pmg_coarse_factor_solver_type=$FACTOR_SOLVER"
  echo "validate_node_contiguity=$VALIDATE_NODE_CONTIGUITY"
  echo "python=$PYTHON"
  echo "timestamp=$(date --iso-8601=seconds)"
} | tee "$CASE_DIR/run_metadata.env"

cat > "$CASE_DIR/case_metadata.json" <<JSON
{
  "case_id": "${CASE_ID}",
  "candidate": "${CANDIDATE}",
  "job_id": "${SLURM_JOB_ID}",
  "backend": "element",
  "he_level": ${HE_LEVEL},
  "first_step": 1,
  "total_steps": ${TOTAL_STEPS},
  "nodes": ${SLURM_JOB_NUM_NODES},
  "total_ranks": ${TOTAL_RANKS_EXPECTED},
  "ranks_per_node": ${RANKS_PER_NODE},
  "cpus_per_task": ${SLURM_CPUS_PER_TASK:-1},
  "step_time_limit_s": ${STEP_TIME_LIMIT_S},
  "wall_time": "${WALL_TIME}",
  "problem_build_mode": "${HE_PROBLEM_BUILD_MODE}",
  "mesh_source": "${HE_MESH_SOURCE}",
  "distribution_strategy": "${HE_DISTRIBUTION_STRATEGY}",
  "assembly_backend": "${HE_ASSEMBLY_BACKEND}",
  "line_search": "${HE_LINE_SEARCH}",
  "trust_radius_init": "${HE_TRUST_RADIUS_INIT}",
  "tolx_rel": "${HE_TOLX_REL}",
  "pmg_coarsest_level": ${COARSE_LEVEL},
  "pmg_coarse_pc_type": "${COARSE_PC}",
  "pmg_coarse_redundant_number": ${REDUNDANT_NUMBER},
  "pmg_coarse_telescope_reduction_factor": ${TELESCOPE_REDUCTION},
  "pmg_coarse_factor_solver_type": "${FACTOR_SOLVER}",
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

RANK_HOST_ORDER="$CASE_DIR/rank_host_order.csv"
srun -n "$TOTAL_RANKS_EXPECTED" --cpu-bind=cores --distribution=block:block \
  bash -lc 'printf "%06d,%06d,%s\n" "${SLURM_PROCID}" "${SLURM_LOCALID}" "$(hostname)"' \
  | sort > "$RANK_HOST_ORDER"
awk -F, '{print $3}' "$RANK_HOST_ORDER" | sort | uniq -c > "$CASE_DIR/rank_hosts.txt"

srun -n "$TOTAL_RANKS_EXPECTED" --cpu-bind=verbose --distribution=block:block \
  bash -lc 'echo "rank=${SLURM_PROCID} local=${SLURM_LOCALID} host=$(hostname) cpuset=$(taskset -pc $$)"' \
  > "$CASE_DIR/binding.txt" 2>&1 || true

if [[ "$VALIDATE_NODE_CONTIGUITY" == "1" ]]; then
  "$PYTHON" - "$RANK_HOST_ORDER" "$EXPECTED_NODES" "$RANKS_PER_NODE" "$TOTAL_RANKS_EXPECTED" "$CASE_DIR/rank_node_layout.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

rank_host_path = Path(sys.argv[1])
expected_nodes = int(sys.argv[2])
ranks_per_node = int(sys.argv[3])
total_ranks = int(sys.argv[4])
out_path = Path(sys.argv[5])

rows = []
for line in rank_host_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rank_s, local_s, host = line.split(",", 2)
    rows.append({"rank": int(rank_s), "local_rank": int(local_s), "host": host})
rows.sort(key=lambda row: row["rank"])

errors = []
if len(rows) != total_ranks:
    errors.append(f"expected {total_ranks} rank rows, got {len(rows)}")
if [row["rank"] for row in rows] != list(range(len(rows))):
    errors.append("rank_host_order.csv does not contain contiguous global ranks")

host_counts = Counter(row["host"] for row in rows)
if len(host_counts) != expected_nodes:
    errors.append(f"expected {expected_nodes} unique hosts, got {len(host_counts)}")
for host, count in sorted(host_counts.items()):
    if count != ranks_per_node:
        errors.append(f"host {host} has {count} ranks, expected {ranks_per_node}")

groups = []
for group_id in range(expected_nodes):
    start = group_id * ranks_per_node
    stop = start + ranks_per_node
    group = rows[start:stop]
    hosts = sorted({row["host"] for row in group})
    groups.append(
        {
            "group": group_id,
            "rank_start": start,
            "rank_stop_exclusive": stop,
            "hosts": hosts,
            "rank_count": len(group),
        }
    )
    if len(hosts) != 1:
        errors.append(
            f"group {group_id} ranks {start}:{stop} span hosts {','.join(hosts)}"
        )

payload = {
    "expected_nodes": expected_nodes,
    "ranks_per_node": ranks_per_node,
    "total_ranks": total_ranks,
    "unique_hosts": sorted(host_counts),
    "host_counts": dict(sorted(host_counts.items())),
    "groups": groups,
    "valid": not errors,
    "errors": errors,
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(2)
PY
fi

RESULT_JSON="$CASE_DIR/output.json"
COMMAND=(
  "$PYTHON" -u experiments/runners/run_trust_region_case.py
  --problem he
  --backend element
  --level "$HE_LEVEL"
  --steps 1
  --start-step 1
  --total-steps "$TOTAL_STEPS"
  --profile performance
  --ksp-type stcg
  --pc-type mg
  --ksp-rtol 1e-1
  --ksp-max-it 30
  --gamg-threshold 0.05
  --gamg-agg-nsmooths 1
  --gamg-set-coordinates
  --use-near-nullspace
  --no-pc-setup-on-ksp-cap
  --he-pmg-coarsest-level "$COARSE_LEVEL"
  --he-pmg-smoother-ksp-type chebyshev
  --he-pmg-smoother-pc-type jacobi
  --he-pmg-smoother-steps 2
  --he-pmg-coarse-pc-type "$COARSE_PC"
  --he-pmg-coarse-redundant-number "$REDUNDANT_NUMBER"
  --he-pmg-coarse-telescope-reduction-factor "$TELESCOPE_REDUCTION"
  --he-pmg-galerkin both
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
  --he-mesh-source "$HE_MESH_SOURCE"
  --distribution-strategy "$HE_DISTRIBUTION_STRATEGY"
  --assembly-backend "$HE_ASSEMBLY_BACKEND"
  --local-coloring
  --out "$RESULT_JSON"
)

if [[ "$FACTOR_SOLVER" != "none" && -n "$FACTOR_SOLVER" ]]; then
  COMMAND+=(--he-pmg-coarse-factor-solver-type "$FACTOR_SOLVER")
fi

printf "%q " srun -n "$TOTAL_RANKS_EXPECTED" --cpu-bind=cores --distribution=block:block "${COMMAND[@]}" \
  > "$CASE_DIR/command.txt"
printf "\n" >> "$CASE_DIR/command.txt"

srun -n "$TOTAL_RANKS_EXPECTED" --cpu-bind=cores --distribution=block:block "${COMMAND[@]}"
