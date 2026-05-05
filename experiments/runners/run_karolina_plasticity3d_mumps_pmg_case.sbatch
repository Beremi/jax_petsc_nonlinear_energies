#!/usr/bin/env bash
# Karolina Plasticity3D P4(L1_2), lambda=1.55, MUMPS-coarse PMG case.
#
# The submitter provides one full-node-population job shape.  This script keeps
# rank order contiguous by node and sets PCREDUNDANT groups to the node count.

#SBATCH --job-name=p3d_mumps_pmg
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --nodes=1
#SBATCH --time=00:12:00

set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "usage: $0 OUT_ROOT CASE_NAME EXPECTED_NODES RANKS_PER_NODE TOTAL_RANKS MAXIT REDUNDANT_NUMBER FACTOR_SOLVER WALL_TIME" >&2
  exit 2
fi

OUT_ROOT="$1"
CASE_NAME="$2"
EXPECTED_NODES="$3"
RANKS_PER_NODE="$4"
TOTAL_RANKS_EXPECTED="$5"
MAXIT="$6"
REDUNDANT_NUMBER="$7"
FACTOR_SOLVER="$8"
WALL_TIME="$9"

SOURCE_ROOT="${SOURCE_ROOT:-tmp/source_compare/slope_stability_petsc4py}"
MESH_NAME="${MESH_NAME:-hetero_ssr_L1_2}"
CONSTRAINT_VARIANT="${CONSTRAINT_VARIANT:-glued_bottom}"
LAMBDA_TARGET="${LAMBDA_TARGET:-1.55}"
KSP_RTOL="${KSP_RTOL:-1e-1}"
KSP_MAX_IT="${KSP_MAX_IT:-100}"
GRAD_STOP_TOL="${GRAD_STOP_TOL:-1e-2}"
VALIDATE_NODE_CONTIGUITY="${VALIDATE_NODE_CONTIGUITY:-1}"
KAROLINA_MEM_BIND="${KAROLINA_MEM_BIND:-local}"

build_sequential_cpu_map() {
  local count="$1"
  local map=""
  local cpu

  if (( count < 1 || count > 128 )); then
    echo "Karolina CPU map supports 1..128 ranks per node, got ${count}." >&2
    return 2
  fi

  for ((cpu = 0; cpu < count; cpu++)); do
    if [[ -n "$map" ]]; then
      map+=","
    fi
    map+="$cpu"
  done
  printf "%s\n" "$map"
}

if [[ "${SLURM_JOB_NUM_NODES:?}" != "$EXPECTED_NODES" ]]; then
  echo "Expected ${EXPECTED_NODES} nodes, got ${SLURM_JOB_NUM_NODES}." >&2
  exit 2
fi

if [[ "${SLURM_NTASKS:?}" != "$TOTAL_RANKS_EXPECTED" ]]; then
  echo "Expected ${TOTAL_RANKS_EXPECTED} MPI ranks, got SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi

if [[ "${SLURM_CPUS_PER_TASK:-1}" != "1" ]]; then
  echo "This campaign expects --cpus-per-task=1, got ${SLURM_CPUS_PER_TASK}." >&2
  exit 2
fi

KAROLINA_CPU_MAP="${KAROLINA_CPU_MAP:-$(build_sequential_cpu_map "$RANKS_PER_NODE")}"
IFS=, read -r -a KAROLINA_CPU_MAP_ENTRIES <<< "$KAROLINA_CPU_MAP"
if (( ${#KAROLINA_CPU_MAP_ENTRIES[@]} != RANKS_PER_NODE )); then
  echo "KAROLINA_CPU_MAP has ${#KAROLINA_CPU_MAP_ENTRIES[@]} entries, expected ${RANKS_PER_NODE}." >&2
  exit 2
fi

SRUN_PLACEMENT_ARGS=(
  -n "$TOTAL_RANKS_EXPECTED"
  --ntasks-per-node="$RANKS_PER_NODE"
  --cpu-bind="map_cpu:${KAROLINA_CPU_MAP}"
  --mem-bind="$KAROLINA_MEM_BIND"
  --distribution=block:block
)
SRUN_PLACEMENT_VERBOSE_ARGS=(
  -n "$TOTAL_RANKS_EXPECTED"
  --ntasks-per-node="$RANKS_PER_NODE"
  --cpu-bind="verbose,map_cpu:${KAROLINA_CPU_MAP}"
  --mem-bind="$KAROLINA_MEM_BIND"
  --distribution=block:block
)

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1}"
export MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER="$REDUNDANT_NUMBER"
export MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER="$FACTOR_SOLVER"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -n "${HE_ENV_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "$HE_ENV_SETUP"
elif [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_SETUP"
elif [[ -f "experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh"
fi

PYTHON="${PYTHON:-./.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable '$PYTHON' is not available. Set PYTHON or create .venv in the full clone." >&2
  exit 2
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source comparison root '$SOURCE_ROOT' is missing." >&2
  exit 2
fi

CASE_DIR="${OUT_ROOT}/cases/${CASE_NAME}/job_${SLURM_JOB_ID}"
mkdir -p "$CASE_DIR"

cleanup() {
  set +e
  sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName,State,Elapsed,AllocNodes,AllocCPUS,MaxRSS,ExitCode -P \
    > "$CASE_DIR/sacct_${SLURM_JOB_ID}.txt" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "case_name=$CASE_NAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
  echo "pwd=$PWD"
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "nodelist=${SLURM_JOB_NODELIST:-}"
  echo "total_ranks=$TOTAL_RANKS_EXPECTED"
  echo "ranks_per_node=$RANKS_PER_NODE"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
  echo "karolina_cpu_map=$KAROLINA_CPU_MAP"
  echo "karolina_mem_bind=$KAROLINA_MEM_BIND"
  echo "mesh_name=$MESH_NAME"
  echo "constraint_variant=$CONSTRAINT_VARIANT"
  echo "lambda_target=$LAMBDA_TARGET"
  echo "maxit=$MAXIT"
  echo "ksp_rtol=$KSP_RTOL"
  echo "ksp_max_it=$KSP_MAX_IT"
  echo "grad_stop_tol=$GRAD_STOP_TOL"
  echo "wall_time=$WALL_TIME"
  echo "assembly_backend=local_constitutiveAD"
  echo "solver_backend=local_pmg_mumps"
  echo "pmg_strategy=same_mesh_p4_p2_p1"
  echo "pmg_coarse_pc_type=redundant"
  echo "pmg_coarse_redundant_number=$REDUNDANT_NUMBER"
  echo "pmg_coarse_factor_solver_type=$FACTOR_SOLVER"
  echo "validate_node_contiguity=$VALIDATE_NODE_CONTIGUITY"
  echo "source_root=$SOURCE_ROOT"
  echo "python=$PYTHON"
  echo "timestamp=$(date --iso-8601=seconds)"
} | tee "$CASE_DIR/run_metadata.env"

cat > "$CASE_DIR/case_metadata.json" <<JSON
{
  "case_name": "${CASE_NAME}",
  "job_id": "${SLURM_JOB_ID}",
  "mesh_name": "${MESH_NAME}",
  "constraint_variant": "${CONSTRAINT_VARIANT}",
  "lambda_target": ${LAMBDA_TARGET},
  "maxit": ${MAXIT},
  "nodes": ${SLURM_JOB_NUM_NODES},
  "total_ranks": ${TOTAL_RANKS_EXPECTED},
  "ranks_per_node": ${RANKS_PER_NODE},
  "cpus_per_task": ${SLURM_CPUS_PER_TASK:-1},
  "karolina_cpu_map": "${KAROLINA_CPU_MAP}",
  "karolina_mem_bind": "${KAROLINA_MEM_BIND}",
  "wall_time": "${WALL_TIME}",
  "assembly_backend": "local_constitutiveAD",
  "solver_backend": "local_pmg_mumps",
  "pmg_strategy": "same_mesh_p4_p2_p1",
  "pmg_coarse_pc_type": "redundant",
  "pmg_coarse_redundant_number": ${REDUNDANT_NUMBER},
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
srun "${SRUN_PLACEMENT_ARGS[@]}" \
  bash -lc 'printf "%06d,%06d,%s\n" "${SLURM_PROCID}" "${SLURM_LOCALID}" "$(hostname)"' \
  | sort > "$RANK_HOST_ORDER"
awk -F, '{print $3}' "$RANK_HOST_ORDER" | sort | uniq -c > "$CASE_DIR/rank_hosts.txt"

srun "${SRUN_PLACEMENT_VERBOSE_ARGS[@]}" \
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
  "$PYTHON" -u experiments/runners/run_plasticity3d_backend_mix_case.py
  --assembly-backend local_constitutiveAD
  --solver-backend local_pmg_mumps
  --source-root "$SOURCE_ROOT"
  --out-dir "$CASE_DIR"
  --output-json "$RESULT_JSON"
  --mesh-name "$MESH_NAME"
  --constraint-variant "$CONSTRAINT_VARIANT"
  --lambda-target "$LAMBDA_TARGET"
  --ksp-rtol "$KSP_RTOL"
  --ksp-max-it "$KSP_MAX_IT"
  --convergence-mode gradient_only
  --grad-stop-tol "$GRAD_STOP_TOL"
  --stop-tol 0.0
  --maxit "$MAXIT"
  --line-search armijo
  --armijo-max-ls 40
)

printf "%q " srun "${SRUN_PLACEMENT_ARGS[@]}" "${COMMAND[@]}" \
  > "$CASE_DIR/command.txt"
printf "\n" >> "$CASE_DIR/command.txt"

srun "${SRUN_PLACEMENT_ARGS[@]}" "${COMMAND[@]}"
