#!/usr/bin/env bash
# Submit Karolina Plasticity3D P2(L1_2_3), lambda=1.55, P1-chain PMG jobs.
#
# Default campaign shape mirrors the local smoke and recent Karolina partial-node
# runs: 16 MPI ranks per node, MUMPS coarse solve replicated once per node, and
# the hierarchy P1(L1) -> P1(L1_2) -> P1(L1_2_3) -> P2(L1_2_3).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
NODES_LIST="${NODES_LIST:-1 2 4 8}"
RANKS_PER_NODE="${RANKS_PER_NODE:-16}"
MAXIT="${MAXIT:-5}"
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-4}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
EXCLUSIVE_NODE="${EXCLUSIVE_NODE:-1}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_karolina_plasticity3d_p2_l1_2_3_p1chain_mumps_pmg}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/karolina/plasticity3d_p2_l1_2_3_p1chain_mumps_pmg_scaling/${CAMPAIGN}}"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_karolina_plasticity3d_p2_p1chain_mumps_pmg_case.sbatch}"

MESH_NAME="${MESH_NAME:-hetero_ssr_L1_2_3}"
ELEM_DEGREE="${ELEM_DEGREE:-2}"
PMG_STRATEGY="${PMG_STRATEGY:-uniform_refined_p2_p1_chain}"
CONSTRAINT_VARIANT="${CONSTRAINT_VARIANT:-glued_bottom}"
LAMBDA_TARGET="${LAMBDA_TARGET:-1.55}"
KSP_RTOL="${KSP_RTOL:-1e-1}"
KSP_MAX_IT="${KSP_MAX_IT:-100}"
GRAD_STOP_TOL="${GRAD_STOP_TOL:-1e-2}"
KAROLINA_MEM_BIND="${KAROLINA_MEM_BIND:-local}"
PREPARE_SAME_MESH_HDF5="${PREPARE_SAME_MESH_HDF5:-0}"
HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
HE_ENV_SETUP="${HE_ENV_SETUP:-$REPO_ROOT/experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh}"

seconds_to_time() {
  local value="$1"
  printf "%02d:%02d:%02d\n" $((value / 3600)) $(((value % 3600) / 60)) $((value % 60))
}

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

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

default_cpu_map() {
  local count="$1"
  if (( count == 16 )); then
    printf "0,1,16,17,32,33,48,49,64,65,80,81,96,97,112,113\n"
  else
    build_sequential_cpu_map "$count"
  fi
}

wall_seconds_for_nodes() {
  local nodes="$1"
  case "$nodes" in
    1) echo "${TIME_LIMIT_1N_S:-1200}" ;;
    2) echo "${TIME_LIMIT_2N_S:-840}" ;;
    4) echo "${TIME_LIMIT_4N_S:-600}" ;;
    8) echo "${TIME_LIMIT_8N_S:-600}" ;;
    *)
      echo "Unsupported node count '$nodes'; default NODES_LIST is 1 2 4 8." >&2
      return 2
      ;;
  esac
}

partition_for_nodes() {
  local nodes="$1"
  if [[ -n "${PARTITION:-}" ]]; then
    printf "%s\n" "$PARTITION"
    return
  fi
  case "$nodes" in
    1|2) printf "%s\n" "${PARTITION_EXP:-qcpu_exp}" ;;
    *) printf "%s\n" "${PARTITION_PRODUCTION:-qcpu}" ;;
  esac
}

if [[ "$MESH_NAME" != "hetero_ssr_L1_2_3" ]]; then
  echo "This submitter is intended for MESH_NAME=hetero_ssr_L1_2_3, got '$MESH_NAME'." >&2
  exit 2
fi
if [[ "$ELEM_DEGREE" != "2" ]]; then
  echo "This submitter is intended for ELEM_DEGREE=2, got '$ELEM_DEGREE'." >&2
  exit 2
fi
if [[ "$PMG_STRATEGY" != "uniform_refined_p2_p1_chain" ]]; then
  echo "This submitter is intended for PMG_STRATEGY=uniform_refined_p2_p1_chain, got '$PMG_STRATEGY'." >&2
  exit 2
fi

if [[ -z "${KAROLINA_CPU_MAP:-}" ]]; then
  KAROLINA_CPU_MAP="$(default_cpu_map "$RANKS_PER_NODE")"
fi
IFS=, read -r -a KAROLINA_CPU_MAP_ENTRIES <<< "$KAROLINA_CPU_MAP"
if (( ${#KAROLINA_CPU_MAP_ENTRIES[@]} != RANKS_PER_NODE )); then
  echo "KAROLINA_CPU_MAP has ${#KAROLINA_CPU_MAP_ENTRIES[@]} entries, expected ${RANKS_PER_NODE}." >&2
  exit 2
fi
CPU_MAP_SUMMARY="custom_map_cpu:${KAROLINA_CPU_MAP//,/;}"

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/summary"

PLAN="$OUT_ROOT/campaign_plan.csv"
COMMANDS="$OUT_ROOT/sbatch_commands.txt"
SUBMITTED="$OUT_ROOT/submitted_jobs.txt"

echo "case,nodes,ranks_per_node,total_ranks,maxit,redundant_number,factor_solver,time_limit,estimated_node_hours,exclusive_node,placement,cpu_map,mem_bind,partition,qos,mesh_name,elem_degree,pmg_strategy" > "$PLAN"
: > "$COMMANDS"
: > "$SUBMITTED"

total_node_seconds=0

for nodes in $NODES_LIST; do
  wall_s="$(wall_seconds_for_nodes "$nodes")"
  time_limit="$(seconds_to_time "$wall_s")"
  total_ranks=$((nodes * RANKS_PER_NODE))
  redundant_number="$nodes"
  partition="$(partition_for_nodes "$nodes")"
  case_name="p3d_p2l123_p1chain_mumps_n${nodes}_rpn${RANKS_PER_NODE}_np${total_ranks}"
  estimated_node_hours="$(awk -v n="$nodes" -v s="$wall_s" 'BEGIN { printf "%.4f", n * s / 3600.0 }')"
  total_node_seconds=$((total_node_seconds + nodes * wall_s))

  echo "${case_name},${nodes},${RANKS_PER_NODE},${total_ranks},${MAXIT},${redundant_number},${FACTOR_SOLVER},${time_limit},${estimated_node_hours},${EXCLUSIVE_NODE},map_cpu,${CPU_MAP_SUMMARY},${KAROLINA_MEM_BIND},${partition},${QOS},${MESH_NAME},${ELEM_DEGREE},${PMG_STRATEGY}" >> "$PLAN"

  env_prefix=(
    env
    "HE_ENV_SETUP=$HE_ENV_SETUP"
    "MESH_NAME=$MESH_NAME"
    "ELEM_DEGREE=$ELEM_DEGREE"
    "PMG_STRATEGY=$PMG_STRATEGY"
    "CONSTRAINT_VARIANT=$CONSTRAINT_VARIANT"
    "LAMBDA_TARGET=$LAMBDA_TARGET"
    "KSP_RTOL=$KSP_RTOL"
    "KSP_MAX_IT=$KSP_MAX_IT"
    "GRAD_STOP_TOL=$GRAD_STOP_TOL"
    "KAROLINA_CPU_MAP=$KAROLINA_CPU_MAP"
    "KAROLINA_MEM_BIND=$KAROLINA_MEM_BIND"
    "PREPARE_SAME_MESH_HDF5=$PREPARE_SAME_MESH_HDF5"
    "HDF5_USE_FILE_LOCKING=$HDF5_USE_FILE_LOCKING"
  )

  cmd=(
    "${env_prefix[@]}"
    sbatch
    --job-name "$case_name"
    --account "$ACCOUNT"
    --qos "$QOS"
    --partition "$partition"
    --nodes "$nodes"
    --ntasks-per-node "$RANKS_PER_NODE"
    --cpus-per-task 1
    --time "$time_limit"
    --chdir "$REPO_ROOT"
    --output "$OUT_ROOT/slurm/%x-%j.out"
    --error "$OUT_ROOT/slurm/%x-%j.err"
  )
  if [[ "$EXCLUSIVE_NODE" == "1" ]]; then
    cmd+=(--exclusive)
  fi
  if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=(
    "$RUNNER"
    "$OUT_ROOT"
    "$case_name"
    "$nodes"
    "$RANKS_PER_NODE"
    "$total_ranks"
    "$MAXIT"
    "$redundant_number"
    "$FACTOR_SOLVER"
    "$time_limit"
  )

  quote_cmd "${cmd[@]}" >> "$COMMANDS"
done

total_node_hours="$(awk -v s="$total_node_seconds" 'BEGIN { printf "%.4f", s / 3600.0 }')"
echo "campaign=$CAMPAIGN"
echo "out_root=$OUT_ROOT"
echo "plan=$PLAN"
echo "commands=$COMMANDS"
echo "estimated_node_hours=$total_node_hours"

if awk -v total="$total_node_hours" -v max="$MAX_NODE_HOURS" 'BEGIN { exit !(total > max) }'; then
  echo "Refusing to submit: estimated ${total_node_hours} node-hours exceeds MAX_NODE_HOURS=${MAX_NODE_HOURS}." >&2
  echo "Raise MAX_NODE_HOURS intentionally if this campaign is expected." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting. Commands are in $COMMANDS"
  exit 0
fi

while IFS= read -r line; do
  echo "$line"
  eval "$line"
done < "$COMMANDS" | tee -a "$SUBMITTED"

echo "Submitted jobs recorded in $SUBMITTED"
