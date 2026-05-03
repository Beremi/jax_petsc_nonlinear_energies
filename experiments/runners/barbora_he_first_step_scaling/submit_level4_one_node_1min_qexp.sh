#!/usr/bin/env bash
# Submit exactly one post-build Barbora smoke test:
# level 4, one qcpu_exp node, full rank population, and a 1-minute wall cap.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HE_ENV_SETUP="${HE_ENV_SETUP:-$SCRIPT_DIR/env_barbora.local.sh}"
export HE_SINGLE_NODE_SMOKE_TRANSPORT=1
export HE_LEVEL=4
export NODES_LIST=1
export RPS_LIST=18
export TIME_LIMIT=00:01:00
export BACKENDS=element
export PARTITION=qcpu_exp
export MAX_NODE_HOURS=1
export CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_he_l4_1node_fullrank_1min_qexp}"

exec "$SCRIPT_DIR/submit_matrix.sh"
