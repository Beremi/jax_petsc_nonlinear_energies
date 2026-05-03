#!/usr/bin/env bash
# Verify the Barbora runtime stack used by the HyperElasticity first-step jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
ENV_FILE="${HE_ENV_SETUP:-$SCRIPT_DIR/env_barbora.local.sh}"
EXPECTED_PETSC_VERSION="${EXPECTED_PETSC_VERSION:-3.24.2}"

cd "$REPO_ROOT"

if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: environment hook '$ENV_FILE' does not exist." >&2
    echo "Run build_barbora_petsc_env.sh first, or set HE_ENV_SETUP." >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable '$PYTHON' is not available." >&2
  exit 2
fi

echo "== Runtime commands =="
echo "repo:    $REPO_ROOT"
echo "python:  $PYTHON"
echo "mpicc:   $(command -v mpicc || true)"
echo "mpiexec: $(command -v mpiexec || true)"
echo "PETSC_DIR=${PETSC_DIR:-}"
echo

if type ml >/dev/null 2>&1; then
  echo "== Loaded modules =="
  ml list 2>&1 || true
  echo
elif type module >/dev/null 2>&1; then
  echo "== Loaded modules =="
  module list 2>&1 || true
  echo
fi

"$PYTHON" - "$EXPECTED_PETSC_VERSION" <<'PY'
from __future__ import annotations

import importlib
import sys

expected = tuple(int(part) for part in sys.argv[1].split("."))
required = ["numpy", "scipy", "h5py", "mpi4py", "petsc4py", "jax"]

print("== Python import check ==")
for name in required:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'OK')}")

from petsc4py import PETSc
import petsc4py

actual = tuple(PETSc.Sys.getVersion())
print(f"PETSc: {actual}")
print(f"petsc4py: {petsc4py.__version__}")
if actual != expected:
    raise SystemExit(f"Expected PETSc {expected}, got {actual}")
if tuple(int(part) for part in petsc4py.__version__.split(".")[:3]) != expected:
    raise SystemExit(f"Expected petsc4py {expected}, got {petsc4py.__version__}")

mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
if not hasattr(mat, "setPreallocationCOO"):
    raise SystemExit("PETSc.Mat.setPreallocationCOO is missing")
if not hasattr(mat, "setValuesCOO"):
    raise SystemExit("PETSc.Mat.setValuesCOO is missing")

print()
print("Barbora environment OK")
PY
