#!/usr/bin/env bash
# Submit level-4 one-node Barbora socket-layout scaling jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6324}"
PARTITION="${PARTITION:-qcpu_exp}"
HE_LEVEL="${HE_LEVEL:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-24}"
BACKEND="${BACKEND:-element}"
BASELINE_STEP_S="${BASELINE_STEP_S:-35}"
SLURM_OVERHEAD_S="${SLURM_OVERHEAD_S:-60}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_he_l${HE_LEVEL}_one_node_socket_scaling}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/barbora/he_first_step_socket_scaling/${CAMPAIGN}}"
HE_ENV_SETUP="${HE_ENV_SETUP:-$REPO_ROOT/experiments/runners/barbora_he_first_step_scaling/env_barbora.local.sh}"
HE_SINGLE_NODE_SMOKE_TRANSPORT="${HE_SINGLE_NODE_SMOKE_TRANSPORT:-1}"

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

seconds_to_time() {
  local total="$1"
  printf "%02d:%02d:%02d" "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

step_cap_for() {
  local total_ranks="$1"
  local active_sockets="$2"
  awk -v base="$BASELINE_STEP_S" -v ranks="$total_ranks" -v sockets="$active_sockets" \
    'BEGIN { printf "%.0f", base * (36.0 / ranks) * (2.0 / sockets) }'
}

cpu_map_for() {
  local socket0="$1"
  local socket1="$2"
  local cpus=()
  local i

  for ((i = 0; i < socket0; i++)); do
    cpus+=("$i")
  done
  for ((i = 0; i < socket1; i++)); do
    cpus+=("$((18 + i))")
  done

  local IFS=,
  echo "${cpus[*]}"
}

case "$BACKEND" in
  element|fenics) ;;
  *)
    echo "Unsupported backend '$BACKEND'; expected 'element' or 'fenics'." >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/summary"

PLAN="$OUT_ROOT/campaign_plan.csv"
INVALID="$OUT_ROOT/invalid_layouts.csv"
COMMANDS="$OUT_ROOT/sbatch_commands.txt"
SUBMITTED="$OUT_ROOT/submitted_jobs.txt"

echo "layout,backend,level,socket0_ranks,socket1_ranks,active_sockets,total_ranks,cpus_per_task,cpu_map,step_time_limit_s,slurm_time_limit,estimated_node_hours,status,reason" > "$PLAN"
echo "layout,reason" > "$INVALID"
: > "$COMMANDS"
: > "$SUBMITTED"

echo "36+0,invalid_without_oversubscription_on_18_core_socket" >> "$INVALID"
echo "36+0,${BACKEND},${HE_LEVEL},36,0,1,36,1,,,,0,invalid,invalid_without_oversubscription_on_18_core_socket" >> "$PLAN"

layouts=(
  "18+18:18:18:2"
  "18+0:18:0:1"
  "9+9:9:9:2"
  "9+0:9:0:1"
)

total_node_seconds=0

for spec in "${layouts[@]}"; do
  IFS=: read -r layout socket0 socket1 active_sockets <<< "$spec"
  total_ranks=$((socket0 + socket1))
  if (( total_ranks < 1 || total_ranks > 36 )); then
    echo "Invalid total ranks for layout $layout: $total_ranks" >&2
    exit 2
  fi
  if (( socket0 > 18 || socket1 > 18 )); then
    echo "Invalid layout $layout: a Barbora socket has 18 cores." >&2
    exit 2
  fi

  step_cap_s="$(step_cap_for "$total_ranks" "$active_sockets")"
  cpu_map="$(cpu_map_for "$socket0" "$socket1")"
  slurm_wall_s=$((step_cap_s + SLURM_OVERHEAD_S))
  slurm_time="$(seconds_to_time "$slurm_wall_s")"
  estimated_node_hours="$(awk -v s="$slurm_wall_s" 'BEGIN { printf "%.4f", s / 3600.0 }')"
  total_node_seconds=$((total_node_seconds + slurm_wall_s))
  layout_label="${layout//+/p}"
  job_name="he1_l${HE_LEVEL}_sock_${layout_label}_${BACKEND}"

  echo "${layout},${BACKEND},${HE_LEVEL},${socket0},${socket1},${active_sockets},${total_ranks},1,\"${cpu_map}\",${step_cap_s},${slurm_time},${estimated_node_hours},valid," >> "$PLAN"

  cmd=(
    sbatch
    --job-name "$job_name"
    --account "$ACCOUNT"
    --qos "$QOS"
    --partition "$PARTITION"
    --nodes 1
    --ntasks-per-node "$total_ranks"
    --cpus-per-task 1
    --sockets-per-node "$active_sockets"
  )
  if (( active_sockets == 2 )); then
    cmd+=(--ntasks-per-socket "$socket0")
  else
    cmd+=(--ntasks-per-socket "$total_ranks")
  fi
  cmd+=(
    --time "$slurm_time"
    --chdir "$REPO_ROOT"
    --output "$OUT_ROOT/slurm/%x-%j.out"
    --error "$OUT_ROOT/slurm/%x-%j.err"
    --export "ALL,HE_ENV_SETUP=${HE_ENV_SETUP},HE_SINGLE_NODE_SMOKE_TRANSPORT=${HE_SINGLE_NODE_SMOKE_TRANSPORT}"
  )
  if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=(
    "$SCRIPT_DIR/run_he_first_step_socket_case.sbatch"
    "$OUT_ROOT"
    "$HE_LEVEL"
    "$layout"
    "$socket0"
    "$socket1"
    "$active_sockets"
    "$total_ranks"
    "$step_cap_s"
    "$TOTAL_STEPS"
    "$BACKEND"
    "$slurm_time"
    "$cpu_map"
  )

  quote_cmd "${cmd[@]}" >> "$COMMANDS"
done

total_node_hours="$(awk -v s="$total_node_seconds" 'BEGIN { printf "%.4f", s / 3600.0 }')"
echo "campaign=$CAMPAIGN"
echo "out_root=$OUT_ROOT"
echo "plan=$PLAN"
echo "invalid=$INVALID"
echo "commands=$COMMANDS"
echo "estimated_node_hours=$total_node_hours"

if awk -v total="$total_node_hours" -v max="$MAX_NODE_HOURS" 'BEGIN { exit !(total > max) }'; then
  echo "Refusing to submit: estimated ${total_node_hours} node-hours exceeds MAX_NODE_HOURS=${MAX_NODE_HOURS}." >&2
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
