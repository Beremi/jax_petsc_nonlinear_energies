#!/usr/bin/env bash
# Example environment hook sourced by run_he_first_step_case.sbatch when
# HE_ENV_SETUP points to this file. Copy it to env_barbora.local.sh, or let
# build_barbora_petsc_env.sh generate env_barbora.local.sh after the build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
BARBORA_ENV_MODULES="${BARBORA_ENV_MODULES:-foss/2022b Python/3.10.8-GCCcore-12.2.0 CMake/3.24.3-GCCcore-12.2.0 Ninja/1.11.1-GCCcore-12.2.0 git/2.38.1-GCCcore-12.2.0-nodocs cURL/7.86.0-GCCcore-12.2.0}"

if type ml >/dev/null 2>&1; then
  ml -f purge
  # Intentionally split the module list on whitespace.
  # shellcheck disable=SC2086
  ml $BARBORA_ENV_MODULES
elif type module >/dev/null 2>&1; then
  module purge
  # shellcheck disable=SC2086
  module load $BARBORA_ENV_MODULES
fi

export PETSC_DIR="${PETSC_DIR:-$REPO_ROOT/local_env/prefix}"
unset PETSC_ARCH
export PATH="$REPO_ROOT/.venv/bin:$PETSC_DIR/bin:$PETSC_DIR/lib/petsc/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PETSC_DIR/lib:$PETSC_DIR/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PETSC_DIR/lib/pkgconfig:$PETSC_DIR/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PETSC_DIR:${CMAKE_PREFIX_PATH:-}"
export PYTHON="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [[ "${HE_SINGLE_NODE_SMOKE_TRANSPORT:-0}" == "1" ]]; then
  unset OMPI_MCA_btl_tcp_if_include
  unset OMPI_MCA_btl_openib_if_include
  export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
  export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader}"
fi
