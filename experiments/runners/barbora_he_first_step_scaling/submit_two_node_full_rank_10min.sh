#!/usr/bin/env bash
# Submit exactly one Barbora CPU first-step test:
# level 5, two nodes, full MPI rank population, and a 10-minute wall cap.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HE_LEVEL=5
export NODES_LIST=2
export RPS_LIST=18
export TIME_LIMIT=00:10:00
export BACKENDS=element
export MAX_NODE_HOURS=1
export CAMPAIGN="${CAMPAIGN:-$(date +%Y%m%d_%H%M%S)_he_l5_2node_fullrank_10min}"

exec "$SCRIPT_DIR/submit_matrix.sh"
