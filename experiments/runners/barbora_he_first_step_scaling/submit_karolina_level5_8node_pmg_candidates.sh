#!/usr/bin/env bash
# Submit the focused Karolina level-5, 8-node PMG coarse-solver candidates.
#
# By default this submits four real jobs. Use DRY_RUN=1 to preview commands
# without submitting, or SBATCH_TEST_ONLY=1 for Slurm admission checks only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu}"
HE_LEVEL="${HE_LEVEL:-5}"
NODES="${NODES:-8}"
RANKS_PER_NODE="${RANKS_PER_NODE:-128}"
TOTAL_STEPS="${TOTAL_STEPS:-24}"
TIME_LIMIT="${TIME_LIMIT:-00:05:00}"
STEP_TIME_LIMIT_S="${STEP_TIME_LIMIT_S:-270}"
HE_MESH_SOURCE="${HE_MESH_SOURCE:-procedural}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-4}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_karolina_he_l${HE_LEVEL}_8n_pmg_candidates}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/karolina/he_l${HE_LEVEL}_8node_pmg_candidates/${CAMPAIGN}}"
CANDIDATES="${CANDIDATES:-pmg_l3_redundant8_mumps pmg_l3_redundant8_superlu pmg_l3_tel16_mumps pmg_l2_redundant8_mumps}"
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

candidate_settings() {
  local candidate="$1"
  case "$candidate" in
    pmg_l3_redundant8_mumps)
      COARSE_LEVEL=3
      COARSE_PC=redundant
      REDUNDANT_NUMBER=8
      TELESCOPE_REDUCTION=0
      FACTOR_SOLVER=mumps
      ;;
    pmg_l3_redundant8_superlu)
      COARSE_LEVEL=3
      COARSE_PC=redundant
      REDUNDANT_NUMBER=8
      TELESCOPE_REDUCTION=0
      FACTOR_SOLVER=superlu_dist
      ;;
    pmg_l3_tel16_mumps)
      COARSE_LEVEL=3
      COARSE_PC=telescope
      REDUNDANT_NUMBER=0
      TELESCOPE_REDUCTION=16
      FACTOR_SOLVER=mumps
      ;;
    pmg_l2_redundant8_mumps)
      COARSE_LEVEL=2
      COARSE_PC=redundant
      REDUNDANT_NUMBER=8
      TELESCOPE_REDUCTION=0
      FACTOR_SOLVER=mumps
      ;;
    *)
      echo "Unknown candidate '$candidate'." >&2
      return 2
      ;;
  esac
}

if (( NODES != 8 )); then
  echo "This focused campaign is intentionally fixed to 8 nodes; got NODES=$NODES." >&2
  exit 2
fi
if (( RANKS_PER_NODE != 128 )); then
  echo "Karolina full CPU-node population expects RANKS_PER_NODE=128." >&2
  exit 2
fi

TIME_LIMIT_S="$(time_to_seconds "$TIME_LIMIT")"
TOTAL_RANKS=$((NODES * RANKS_PER_NODE))
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

echo "candidate,level,nodes,ranks_per_node,total_ranks,mesh_source,coarsest_level,coarse_pc,redundant_number,telescope_reduction,factor_solver,time_limit,step_time_limit_s,estimated_node_hours,placement,cpu_map,mem_bind" > "$PLAN"
: > "$COMMANDS"
: > "$SUBMITTED"

total_node_seconds=0

for candidate in $CANDIDATES; do
  candidate_settings "$candidate"

  estimated_node_hours="$(awk -v n="$NODES" -v s="$TIME_LIMIT_S" 'BEGIN { printf "%.4f", n * s / 3600.0 }')"
  total_node_seconds=$((total_node_seconds + NODES * TIME_LIMIT_S))
  job_name="he5_${candidate}"

  echo "${candidate},${HE_LEVEL},${NODES},${RANKS_PER_NODE},${TOTAL_RANKS},${HE_MESH_SOURCE},${COARSE_LEVEL},${COARSE_PC},${REDUNDANT_NUMBER},${TELESCOPE_REDUCTION},${FACTOR_SOLVER},${TIME_LIMIT},${STEP_TIME_LIMIT_S},${estimated_node_hours},block:block,${CPU_MAP_SUMMARY},${KAROLINA_MEM_BIND}" >> "$PLAN"

  cmd=(
    sbatch
    --job-name "$job_name"
    --account "$ACCOUNT"
    --qos "$QOS"
    --partition "$PARTITION"
    --nodes "$NODES"
    --ntasks-per-node "$RANKS_PER_NODE"
    --cpus-per-task 1
    --distribution block:block
    --time "$TIME_LIMIT"
    --chdir "$REPO_ROOT"
    --output "$OUT_ROOT/slurm/%x-%j.out"
    --error "$OUT_ROOT/slurm/%x-%j.err"
  )
  if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=(
    "$SCRIPT_DIR/run_karolina_he_l5_pmg_candidate.sbatch"
    "$OUT_ROOT"
    "$candidate"
    "$HE_LEVEL"
    "$NODES"
    "$RANKS_PER_NODE"
    "$TOTAL_RANKS"
    "$TOTAL_STEPS"
    "$STEP_TIME_LIMIT_S"
    "$COARSE_LEVEL"
    "$COARSE_PC"
    "$REDUNDANT_NUMBER"
    "$TELESCOPE_REDUCTION"
    "$FACTOR_SOLVER"
    "$TIME_LIMIT"
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
  echo "Use fewer candidates or raise MAX_NODE_HOURS intentionally." >&2
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
