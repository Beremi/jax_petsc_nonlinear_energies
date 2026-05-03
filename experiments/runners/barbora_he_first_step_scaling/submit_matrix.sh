#!/usr/bin/env bash
# Submit the prepared Barbora HyperElasticity first-step scaling matrix.
#
# By default this submits real jobs. Use DRY_RUN=1 to preview commands without
# submitting, or SBATCH_TEST_ONLY=1 to ask Slurm to validate requests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6324}"
PARTITION="${PARTITION:-qcpu}"
HE_LEVEL="${HE_LEVEL:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-24}"
BACKENDS="${BACKENDS:-element}"
NODES_LIST="${NODES_LIST:-1 2 4 8 16}"
RPS_LIST="${RPS_LIST:-4 8 12 18}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-100}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_he_l${HE_LEVEL}_step1}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/barbora/he_first_step_scaling/${CAMPAIGN}}"

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

TIME_LIMIT_S="$(time_to_seconds "$TIME_LIMIT")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/summary"

PLAN="$OUT_ROOT/campaign_plan.csv"
COMMANDS="$OUT_ROOT/sbatch_commands.txt"
SUBMITTED="$OUT_ROOT/submitted_jobs.txt"

echo "backend,level,nodes,ranks_per_socket,ranks_per_node,total_ranks,cpus_per_task,time_limit,estimated_node_hours" > "$PLAN"
: > "$COMMANDS"
: > "$SUBMITTED"

total_node_seconds=0

for backend in $BACKENDS; do
  case "$backend" in
    element|fenics) ;;
    *)
      echo "Unsupported backend '$backend'; expected 'element' or 'fenics'." >&2
      exit 2
      ;;
  esac

  for nodes in $NODES_LIST; do
    for rps in $RPS_LIST; do
      if (( rps < 1 || rps > 18 )); then
        echo "Invalid ranks-per-socket '$rps'; Barbora CPU sockets have 18 cores." >&2
        exit 2
      fi

      ranks_per_node=$((2 * rps))
      total_ranks=$((nodes * ranks_per_node))
      estimated_node_hours="$(awk -v n="$nodes" -v s="$TIME_LIMIT_S" 'BEGIN { printf "%.4f", n * s / 3600.0 }')"
      total_node_seconds=$((total_node_seconds + nodes * TIME_LIMIT_S))
      job_name="he1_l${HE_LEVEL}_n${nodes}_rps${rps}_${backend}"

      echo "${backend},${HE_LEVEL},${nodes},${rps},${ranks_per_node},${total_ranks},1,${TIME_LIMIT},${estimated_node_hours}" >> "$PLAN"

      cmd=(
        sbatch
        --job-name "$job_name"
        --account "$ACCOUNT"
        --qos "$QOS"
        --partition "$PARTITION"
        --nodes "$nodes"
        --ntasks-per-node "$ranks_per_node"
        --ntasks-per-socket "$rps"
        --cpus-per-task 1
        --time "$TIME_LIMIT"
        --chdir "$REPO_ROOT"
        --output "$OUT_ROOT/slurm/%x-%j.out"
        --error "$OUT_ROOT/slurm/%x-%j.err"
      )
      if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
        cmd+=(--test-only)
      fi
      cmd+=(
        "$SCRIPT_DIR/run_he_first_step_case.sbatch"
        "$OUT_ROOT"
        "$HE_LEVEL"
        "$rps"
        "$ranks_per_node"
        "$TOTAL_STEPS"
        "$backend"
      )

      quote_cmd "${cmd[@]}" >> "$COMMANDS"
    done
  done
done

total_node_hours="$(awk -v s="$total_node_seconds" 'BEGIN { printf "%.4f", s / 3600.0 }')"
echo "campaign=$CAMPAIGN"
echo "out_root=$OUT_ROOT"
echo "plan=$PLAN"
echo "commands=$COMMANDS"
echo "estimated_node_hours=$total_node_hours"

if awk -v total="$total_node_hours" -v max="$MAX_NODE_HOURS" 'BEGIN { exit !(total > max) }'; then
  echo "Refusing to submit: estimated ${total_node_hours} node-hours exceeds MAX_NODE_HOURS=${MAX_NODE_HOURS}." >&2
  echo "Use a smaller matrix or raise MAX_NODE_HOURS intentionally." >&2
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
