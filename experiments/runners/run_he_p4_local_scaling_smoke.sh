#!/usr/bin/env bash
# Local smoke/scaling harness for rank-local 3D P4 HyperElasticity.
#
# This intentionally defaults to the small level-1 P4 beam so the full public
# runner path can be exercised on a workstation before preparing larger
# Karolina/Barbora campaigns. Override HE_LEVEL, RANKS_LIST, and OUT_ROOT for
# larger local trials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-./.venv/bin/python}"
MPIEXEC="${MPIEXEC:-mpiexec}"
HE_LEVEL="${HE_LEVEL:-1}"
HE_ELEMENT_DEGREE="${HE_ELEMENT_DEGREE:-4}"
RANKS_LIST="${RANKS_LIST:-1 2}"
OUT_ROOT="${OUT_ROOT:-artifacts/raw_results/example_runs/he_p4_local_scaling_smoke/$(date +%Y%m%d_%H%M%S)}"
MAXIT="${MAXIT:-1}"
KSP_MAX_IT="${KSP_MAX_IT:-10}"
PC_TYPE="${PC_TYPE:-gamg}"
KSP_TYPE="${KSP_TYPE:-cg}"
STEP_TIME_LIMIT_S="${STEP_TIME_LIMIT_S:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable '$PYTHON' is not available." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
PLAN="$OUT_ROOT/plan.csv"
echo "level,element_degree,ranks,ksp_type,pc_type,maxit,ksp_max_it,out" > "$PLAN"

for ranks in $RANKS_LIST; do
  CASE_DIR="$OUT_ROOT/he_p${HE_ELEMENT_DEGREE}_l${HE_LEVEL}_np${ranks}"
  mkdir -p "$CASE_DIR"
  RESULT_JSON="$CASE_DIR/output.json"
  COMMAND=(
    "$PYTHON" -u experiments/runners/run_trust_region_case.py
    --problem he
    --backend element
    --level "$HE_LEVEL"
    --he-element-degree "$HE_ELEMENT_DEGREE"
    --steps 1
    --start-step 1
    --total-steps 24
    --profile performance
    --ksp-type "$KSP_TYPE"
    --pc-type "$PC_TYPE"
    --ksp-rtol 1e-1
    --ksp-max-it "$KSP_MAX_IT"
    --gamg-threshold 0.05
    --gamg-agg-nsmooths 1
    --gamg-set-coordinates
    --use-near-nullspace
    --no-pc-setup-on-ksp-cap
    --tolf 1e-4
    --tolg 1e-3
    --tolg-rel 1e-3
    --tolx-rel 1e-3
    --tolx-abs 1e-10
    --maxit "$MAXIT"
    --linesearch-a -0.5
    --linesearch-b 2.0
    --linesearch-tol 1e-1
    --line-search armijo
    --quiet
    --save-linear-timing
    --nproc-threads 1
    --element-reorder-mode block_xyz
    --local-hessian-mode element
    --problem-build-mode rank_local
    --he-mesh-source procedural
    --distribution-strategy overlap_p2p
    --assembly-backend coo_local
    --local-coloring
    --out "$RESULT_JSON"
  )
  if [[ -n "$STEP_TIME_LIMIT_S" ]]; then
    COMMAND+=(--step-time-limit-s "$STEP_TIME_LIMIT_S")
  fi

  printf "%q " "$MPIEXEC" -n "$ranks" "${COMMAND[@]}" > "$CASE_DIR/command.txt"
  printf "\n" >> "$CASE_DIR/command.txt"
  echo "${HE_LEVEL},${HE_ELEMENT_DEGREE},${ranks},${KSP_TYPE},${PC_TYPE},${MAXIT},${KSP_MAX_IT},${RESULT_JSON}" >> "$PLAN"

  "$MPIEXEC" -n "$ranks" "${COMMAND[@]}" 2>&1 | tee "$CASE_DIR/run.log"
done
