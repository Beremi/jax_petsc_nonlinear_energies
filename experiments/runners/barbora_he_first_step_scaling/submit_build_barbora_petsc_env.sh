#!/usr/bin/env bash
# Submit a Barbora build job for the PETSc 3.24.2 HyperElasticity environment.
#
# Use DRY_RUN=1 to print the sbatch command without submitting, or
# SBATCH_TEST_ONLY=1 to ask Slurm to validate the request.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6324}"
PARTITION="${PARTITION:-qcpu}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_barbora_petsc324_env_build}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/barbora/env_build/${CAMPAIGN}}"

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
mkdir -p "$OUT_ROOT/slurm"

cmd=(
  sbatch
  --job-name fne_petsc324_env
  --account "$ACCOUNT"
  --qos "$QOS"
  --partition "$PARTITION"
  --nodes 1
  --ntasks 1
  --cpus-per-task "$CPUS_PER_TASK"
  --time "$TIME_LIMIT"
  --chdir "$REPO_ROOT"
  --output "$OUT_ROOT/slurm/%x-%j.out"
  --error "$OUT_ROOT/slurm/%x-%j.err"
)

if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
  cmd+=(--test-only)
fi

cmd+=("$SCRIPT_DIR/build_barbora_petsc_env.sh")

echo "campaign=$CAMPAIGN"
echo "out_root=$OUT_ROOT"
quote_cmd "${cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

"${cmd[@]}"
