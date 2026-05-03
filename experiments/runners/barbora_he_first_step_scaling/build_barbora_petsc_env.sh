#!/usr/bin/env bash
# Build the Barbora environment used by the HyperElasticity first-step jobs.
#
# This intentionally builds only the JAX+PETSc runtime stack needed by the
# prepared Slurm scripts, not the full FEniCSx/DOLFINx local workstation stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

PETSC_VERSION="${PETSC_VERSION:-3.24.2}"
PETSC_ARCH="${PETSC_ARCH:-barbora-foss2022b-real64-32}"
LOCAL_ENV="${LOCAL_ENV:-$REPO_ROOT/local_env}"
LOCAL_SRC="${LOCAL_SRC:-$LOCAL_ENV/src}"
PREFIX="${PREFIX:-$LOCAL_ENV/prefix}"
VENV="${VENV:-$REPO_ROOT/.venv}"
BARBORA_ENV_MODULES="${BARBORA_ENV_MODULES:-foss/2022b Python/3.10.8-GCCcore-12.2.0 CMake/3.24.3-GCCcore-12.2.0 Ninja/1.11.1-GCCcore-12.2.0 git/2.38.1-GCCcore-12.2.0-nodocs cURL/7.86.0-GCCcore-12.2.0}"
BARBORA_MPI4PY_SPEC="${BARBORA_MPI4PY_SPEC:-mpi4py==4.1.1}"
BARBORA_PYTHON_PACKAGES="${BARBORA_PYTHON_PACKAGES:-numpy==1.26.4 scipy==1.11.4 h5py==3.10.0 jax[cpu]==0.4.30}"
VERIFY_ENV="${VERIFY_ENV:-1}"

if [[ -z "${JOBS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK:-1}" != "1" ]]; then
    JOBS="$SLURM_CPUS_PER_TASK"
  elif [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    JOBS="$SLURM_CPUS_ON_NODE"
  else
    JOBS="$(nproc)"
  fi
fi

log() {
  printf "\n>>> %s\n\n" "$*"
}

ensure_ml() {
  if type ml >/dev/null 2>&1; then
    return
  fi
  if type module >/dev/null 2>&1; then
    ml() { module "$@"; }
    return
  fi
  if [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  fi
  if ! type ml >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    ml() { module "$@"; }
  fi
  if ! type ml >/dev/null 2>&1; then
    echo "ERROR: neither 'ml' nor 'module' is available. Run this on Barbora." >&2
    exit 2
  fi
}

load_modules() {
  if [[ "${SKIP_MODULE_LOAD:-0}" == "1" ]]; then
    return
  fi
  ensure_ml
  log "Loading Barbora module stack"
  ml -f purge
  # Intentionally split the module list on whitespace.
  # shellcheck disable=SC2086
  ml $BARBORA_ENV_MODULES
  ml list
}

download_petsc() {
  local tarball="$LOCAL_SRC/petsc-${PETSC_VERSION}.tar.gz"
  local url="https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz"
  mkdir -p "$LOCAL_SRC"
  if [[ -f "$tarball" ]]; then
    return
  fi
  log "Downloading PETSc ${PETSC_VERSION}"
  if type curl >/dev/null 2>&1; then
    curl -L --retry 3 --fail "$url" -o "$tarball"
  else
    wget -O "$tarball" "$url"
  fi
}

write_env_hook() {
  local env_file="$SCRIPT_DIR/env_barbora.local.sh"
  log "Writing $env_file"
  cat > "$env_file" <<'ENV_EOF'
#!/usr/bin/env bash
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
ENV_EOF
  chmod 700 "$env_file"
}

load_modules

log "Creating Python virtual environment"
mkdir -p "$LOCAL_SRC" "$PREFIX"
python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

export CC="${CC:-mpicc}"
export CXX="${CXX:-mpicxx}"
export FC="${FC:-mpifort}"
export MPICC="${MPICC:-mpicc}"
export MPICXX="${MPICXX:-mpicxx}"
export MPIFC="${MPIFC:-mpifort}"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"

log "Installing Python build/runtime prerequisites"
python -m pip install --upgrade pip "setuptools<80" wheel packaging "Cython>=3.0,<3.3"
MPICC="$MPICC" python -m pip install --no-cache-dir --no-binary=mpi4py "$BARBORA_MPI4PY_SPEC"
# Intentionally split the package list on whitespace.
# shellcheck disable=SC2086
python -m pip install --no-cache-dir $BARBORA_PYTHON_PACKAGES

if [[ "${REBUILD_PETSC:-0}" == "1" || ! -x "$PREFIX/lib/petsc/bin/petscversion" ]]; then
  download_petsc
  log "Unpacking PETSc ${PETSC_VERSION}"
  rm -rf "$LOCAL_SRC/petsc-${PETSC_VERSION}"
  tar xf "$LOCAL_SRC/petsc-${PETSC_VERSION}.tar.gz" -C "$LOCAL_SRC"
  cd "$LOCAL_SRC/petsc-${PETSC_VERSION}"

  export PETSC_DIR="$PWD"
  export PETSC_ARCH

  log "Configuring PETSc ${PETSC_VERSION}"
  python ./configure \
    PETSC_ARCH="$PETSC_ARCH" \
    --prefix="$PREFIX" \
    --COPTFLAGS="-O2" \
    --CXXOPTFLAGS="-O2" \
    --FOPTFLAGS="-O2" \
    --with-64-bit-indices=no \
    --with-debugging=no \
    --with-fortran-bindings=no \
    --with-shared-libraries=1 \
    --with-scalar-type=real \
    --with-precision=double \
    --download-hypre \
    --download-metis \
    --download-parmetis \
    --download-mumps \
    --download-mumps-avoid-mpi-in-place \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist

  log "Building PETSc ${PETSC_VERSION} with ${JOBS} jobs"
  make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all -j"$JOBS"
  make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" install
else
  log "PETSc prefix already exists; set REBUILD_PETSC=1 to rebuild"
fi

if [[ ! -d "$LOCAL_SRC/petsc-${PETSC_VERSION}/src/binding/petsc4py" ]]; then
  download_petsc
  log "Unpacking PETSc ${PETSC_VERSION} sources for petsc4py"
  rm -rf "$LOCAL_SRC/petsc-${PETSC_VERSION}"
  tar xf "$LOCAL_SRC/petsc-${PETSC_VERSION}.tar.gz" -C "$LOCAL_SRC"
fi

cd "$REPO_ROOT"
export PETSC_DIR="$PREFIX"
unset PETSC_ARCH
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"

log "Installing petsc4py ${PETSC_VERSION}"
python -m pip install --no-build-isolation --no-cache-dir \
  "$LOCAL_SRC/petsc-${PETSC_VERSION}/src/binding/petsc4py"

write_env_hook

if [[ "$VERIFY_ENV" == "1" ]]; then
  log "Verifying Barbora runtime environment"
  HE_ENV_SETUP="$SCRIPT_DIR/env_barbora.local.sh" "$SCRIPT_DIR/check_barbora_env.sh"
fi

log "Barbora PETSc ${PETSC_VERSION} environment is ready"
