#!/usr/bin/env bash
# Submit Plasticity3D P4(L1_2), lambda=1.55, MUMPS-coarse PMG node scaling.
#
# This prepares one independent Slurm job for each node count.  Each job uses
# full Karolina CPU-node population and sets PETSc PCREDUNDANT groups to the
# number of requested nodes so the coarse solve is replicated per node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
NODES_LIST="${NODES_LIST:-1 2 4 8}"
RANKS_PER_NODE="${RANKS_PER_NODE:-128}"
MAXIT="${MAXIT:-5}"
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-3}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_karolina_plasticity3d_p4_l1_2_mumps_pmg}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/karolina/plasticity3d_p4_l1_2_mumps_pmg_scaling/${CAMPAIGN}}"
KAROLINA_MEM_BIND="${KAROLINA_MEM_BIND:-local}"

time_to_seconds() {
  local value="$1"
  local hh mm ss
  IFS=: read -r hh mm ss <<< "$value"
  if [[ -z "${hh:-}" || -z "${mm:-}" || -z "${ss:-}" ]]; then
    echo "TIME_LIMIT must be HH:MM:SS, got '$value'" >&2
    return 2
  fi
  echo $((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
}

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

wall_seconds_for_nodes() {
  local nodes="$1"
  case "$nodes" in
    1) echo "${TIME_LIMIT_1N_S:-720}" ;;
    2) echo "${TIME_LIMIT_2N_S:-600}" ;;
    4) echo "${TIME_LIMIT_4N_S:-480}" ;;
    8) echo "${TIME_LIMIT_8N_S:-480}" ;;
    *)
      echo "Unsupported node count '$nodes'; default NODES_LIST is 1 2 4 8." >&2
      return 2
      ;;
  esac
}

if (( RANKS_PER_NODE != 128 )); then
  echo "Karolina full CPU-node population expects RANKS_PER_NODE=128." >&2
  exit 2
fi

if [[ -z "${KAROLINA_CPU_MAP:-}" ]]; then
  KAROLINA_CPU_MAP="$(build_sequential_cpu_map "$RANKS_PER_NODE")"
  CPU_MAP_SUMMARY="map_cpu:0-$((RANKS_PER_NODE - 1))"
else
  CPU_MAP_SUMMARY="custom_map_cpu"
fi
IFS=, read -r -a KAROLINA_CPU_MAP_ENTRIES <<< "$KAROLINA_CPU_MAP"
if (( ${#KAROLINA_CPU_MAP_ENTRIES[@]} != RANKS_PER_NODE )); then
  echo "KAROLINA_CPU_MAP has ${#KAROLINA_CPU_MAP_ENTRIES[@]} entries, expected ${RANKS_PER_NODE}." >&2
  exit 2
fi
export KAROLINA_CPU_MAP KAROLINA_MEM_BIND

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/summary"

PLAN="$OUT_ROOT/campaign_plan.csv"
COMMANDS="$OUT_ROOT/sbatch_commands.txt"
SUBMITTED="$OUT_ROOT/submitted_jobs.txt"

echo "case,nodes,ranks_per_node,total_ranks,maxit,redundant_number,factor_solver,time_limit,estimated_node_hours,placement,cpu_map,mem_bind,partition,qos" > "$PLAN"
: > "$COMMANDS"
: > "$SUBMITTED"

total_node_seconds=0

for nodes in $NODES_LIST; do
  wall_s="$(wall_seconds_for_nodes "$nodes")"
  time_limit="$(seconds_to_time "$wall_s")"
  total_ranks=$((nodes * RANKS_PER_NODE))
  redundant_number="$nodes"
  case_name="p3d_p4l12_mumps_n${nodes}_np${total_ranks}"
  estimated_node_hours="$(awk -v n="$nodes" -v s="$wall_s" 'BEGIN { printf "%.4f", n * s / 3600.0 }')"
  total_node_seconds=$((total_node_seconds + nodes * wall_s))

  echo "${case_name},${nodes},${RANKS_PER_NODE},${total_ranks},${MAXIT},${redundant_number},${FACTOR_SOLVER},${time_limit},${estimated_node_hours},block:block,${CPU_MAP_SUMMARY},${KAROLINA_MEM_BIND},${PARTITION},${QOS}" >> "$PLAN"

  cmd=(
    sbatch
    --job-name "$case_name"
    --account "$ACCOUNT"
    --qos "$QOS"
    --partition "$PARTITION"
    --nodes "$nodes"
    --ntasks-per-node "$RANKS_PER_NODE"
    --cpus-per-task 1
    --distribution block:block
    --time "$time_limit"
    --chdir "$REPO_ROOT"
    --output "$OUT_ROOT/slurm/%x-%j.out"
    --error "$OUT_ROOT/slurm/%x-%j.err"
  )
  if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=(
    "$SCRIPT_DIR/run_karolina_plasticity3d_mumps_pmg_case.sbatch"
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
