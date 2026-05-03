#!/usr/bin/env bash
# Example environment hook sourced by run_he_first_step_case.sbatch when
# HE_ENV_SETUP points to this file. Copy it to env_barbora.local.sh and edit
# locally; do not put credentials or private tokens here.

set -euo pipefail

if type ml >/dev/null 2>&1; then
  ml -f purge
  # Load the exact modules used to build/run the repository environment, e.g.
  # ml Python/...
  # ml PETSc/...
  # ml petsc4py/...
fi

# The Slurm scripts assume a full repository clone. Set PYTHON if the virtual
# environment is not at ./.venv/bin/python.
export PYTHON="${PYTHON:-./.venv/bin/python}"
