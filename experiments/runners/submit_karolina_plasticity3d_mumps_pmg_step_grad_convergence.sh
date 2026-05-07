#!/usr/bin/env bash
# Submit corrected Karolina Plasticity3D P4(L1_2), lambda=1.55 convergence runs.
#
# This campaign mirrors the local corrected convergence check: one Karolina CPU
# node, 16 and 32 MPI ranks, MUMPS coarse solve, and a combined nonlinear stop
# requiring both relative correction and relative gradient reduction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
NODES="${NODES:-1}"
RANKS_LIST="${RANKS_LIST:-16 32}"
MAXIT="${MAXIT:-80}"
FACTOR_SOLVER="${FACTOR_SOLVER:-mumps}"
REDUNDANT_NUMBER="${REDUNDANT_NUMBER:-1}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-2}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
EXCLUSIVE_NODE="${EXCLUSIVE_NODE:-1}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_karolina_plasticity3d_p4_l1_2_mumps_pmg_step_grad_convergence}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/karolina/plasticity3d_p4_l1_2_mumps_pmg_step_grad_convergence/${CAMPAIGN}}"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_karolina_plasticity3d_mumps_pmg_step_grad_case.sbatch}"

SOURCE_ROOT="${SOURCE_ROOT:-tmp/source_compare/slope_stability_petsc4py}"
MESH_NAME="${MESH_NAME:-hetero_ssr_L1_2}"
ELEM_DEGREE="${ELEM_DEGREE:-4}"
PMG_STRATEGY="${PMG_STRATEGY:-same_mesh_p4_p2_p1}"
CONSTRAINT_VARIANT="${CONSTRAINT_VARIANT:-glued_bottom}"
LAMBDA_TARGET="${LAMBDA_TARGET:-1.55}"
KSP_RTOL="${KSP_RTOL:-1e-2}"
KSP_MAX_IT="${KSP_MAX_IT:-200}"
CONVERGENCE_MODE="${CONVERGENCE_MODE:-all}"
STOP_TOL="${STOP_TOL:-0.002}"
GRAD_STOP_RTOL="${GRAD_STOP_RTOL:-0.01}"
GRAD_STOP_TOL="${GRAD_STOP_TOL:-0.0}"
LINE_SEARCH="${LINE_SEARCH:-armijo}"
ARMIJO_MAX_LS="${ARMIJO_MAX_LS:-40}"
KAROLINA_MEM_BIND="${KAROLINA_MEM_BIND:-local}"
PREPARE_SAME_MESH_HDF5="${PREPARE_SAME_MESH_HDF5:-1}"
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

build_spread_cpu_map() {
  local count="$1"
  local block_width base offset cpu
  local map=""

  if (( count < 1 || count > 128 || count % 8 != 0 )); then
    echo "Karolina spread CPU map expects an 8-divisible rank count in 8..128, got ${count}." >&2
    return 2
  fi

  block_width=$((count / 8))
  for base in 0 16 32 48 64 80 96 112; do
    for ((offset = 0; offset < block_width; offset++)); do
      cpu=$((base + offset))
      if [[ -n "$map" ]]; then
        map+=","
      fi
      map+="$cpu"
    done
  done
  printf "%s\n" "$map"
}

wall_seconds_for_ranks() {
  local ranks="$1"
  case "$ranks" in
    16) echo "${TIME_LIMIT_16R_S:-3600}" ;;
    32) echo "${TIME_LIMIT_32R_S:-2700}" ;;
    *)
      echo "Unsupported rank count '$ranks'; default RANKS_LIST is 16 32." >&2
      return 2
      ;;
  esac
}

if [[ "$NODES" != "1" ]]; then
  echo "This convergence campaign is intentionally one-node only; got NODES=${NODES}." >&2
  exit 2
fi
if [[ "$MESH_NAME" != "hetero_ssr_L1_2" ]]; then
  echo "This submitter is intended for MESH_NAME=hetero_ssr_L1_2, got '$MESH_NAME'." >&2
  exit 2
fi
if [[ "$ELEM_DEGREE" != "4" ]]; then
  echo "This submitter is intended for ELEM_DEGREE=4, got '$ELEM_DEGREE'." >&2
  exit 2
fi
if [[ "$PMG_STRATEGY" != "same_mesh_p4_p2_p1" ]]; then
  echo "This submitter is intended for PMG_STRATEGY=same_mesh_p4_p2_p1, got '$PMG_STRATEGY'." >&2
  exit 2
fi
if [[ "$CONVERGENCE_MODE" != "all" ]]; then
  echo "This corrected run expects CONVERGENCE_MODE=all, got '$CONVERGENCE_MODE'." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm" "$OUT_ROOT/summary"

PLAN="$OUT_ROOT/campaign_plan.csv"
COMMANDS="$OUT_ROOT/sbatch_commands.txt"
SUBMITTED="$OUT_ROOT/submitted_jobs.txt"

echo "case,nodes,ranks_per_node,total_ranks,maxit,ksp_rtol,ksp_max_it,convergence_mode,stop_tol,grad_stop_rtol,grad_stop_tol,redundant_number,factor_solver,time_limit,estimated_node_hours,exclusive_node,placement,cpu_map,mem_bind,partition,qos,mesh_name,elem_degree,pmg_strategy" > "$PLAN"
: > "$COMMANDS"
: > "$SUBMITTED"

total_node_seconds=0

for ranks_per_node in $RANKS_LIST; do
  if (( ranks_per_node < 1 || ranks_per_node > 128 )); then
    echo "Karolina CPU nodes support 1..128 ranks per node, got ${ranks_per_node}." >&2
    exit 2
  fi

  wall_s="$(wall_seconds_for_ranks "$ranks_per_node")"
  time_limit="$(seconds_to_time "$wall_s")"
  total_ranks=$((NODES * ranks_per_node))
  cpu_map="${KAROLINA_CPU_MAP:-$(build_spread_cpu_map "$ranks_per_node")}"
  IFS=, read -r -a cpu_map_entries <<< "$cpu_map"
  if (( ${#cpu_map_entries[@]} != ranks_per_node )); then
    echo "CPU map for ${ranks_per_node} ranks has ${#cpu_map_entries[@]} entries." >&2
    exit 2
  fi
  cpu_map_summary="custom_map_cpu:${cpu_map//,/;}"
  case_name="p3d_p4l12_stepgrad_mumps_n${NODES}_rpn${ranks_per_node}_np${total_ranks}"
  estimated_node_hours="$(awk -v n="$NODES" -v s="$wall_s" 'BEGIN { printf "%.4f", n * s / 3600.0 }')"
  total_node_seconds=$((total_node_seconds + NODES * wall_s))

  echo "${case_name},${NODES},${ranks_per_node},${total_ranks},${MAXIT},${KSP_RTOL},${KSP_MAX_IT},${CONVERGENCE_MODE},${STOP_TOL},${GRAD_STOP_RTOL},${GRAD_STOP_TOL},${REDUNDANT_NUMBER},${FACTOR_SOLVER},${time_limit},${estimated_node_hours},${EXCLUSIVE_NODE},map_cpu,${cpu_map_summary},${KAROLINA_MEM_BIND},${PARTITION},${QOS},${MESH_NAME},${ELEM_DEGREE},${PMG_STRATEGY}" >> "$PLAN"

  env_prefix=(
    env
    "HE_ENV_SETUP=$HE_ENV_SETUP"
    "SOURCE_ROOT=$SOURCE_ROOT"
    "MESH_NAME=$MESH_NAME"
    "ELEM_DEGREE=$ELEM_DEGREE"
    "PMG_STRATEGY=$PMG_STRATEGY"
    "CONSTRAINT_VARIANT=$CONSTRAINT_VARIANT"
    "LAMBDA_TARGET=$LAMBDA_TARGET"
    "KSP_RTOL=$KSP_RTOL"
    "KSP_MAX_IT=$KSP_MAX_IT"
    "CONVERGENCE_MODE=$CONVERGENCE_MODE"
    "STOP_TOL=$STOP_TOL"
    "GRAD_STOP_RTOL=$GRAD_STOP_RTOL"
    "GRAD_STOP_TOL=$GRAD_STOP_TOL"
    "LINE_SEARCH=$LINE_SEARCH"
    "ARMIJO_MAX_LS=$ARMIJO_MAX_LS"
    "KAROLINA_CPU_MAP=$cpu_map"
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
    --partition "$PARTITION"
    --nodes "$NODES"
    --ntasks-per-node "$ranks_per_node"
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
    "$NODES"
    "$ranks_per_node"
    "$total_ranks"
    "$MAXIT"
    "$REDUNDANT_NUMBER"
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
