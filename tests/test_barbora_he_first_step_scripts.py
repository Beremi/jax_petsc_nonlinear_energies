from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT_DIR = Path("experiments/runners/barbora_he_first_step_scaling")


def _script_text(*names: str) -> str:
    return "\n".join((SCRIPT_DIR / name).read_text(encoding="utf-8") for name in names)


def test_barbora_he_scripts_avoid_disallowed_slurm_options():
    text = _script_text(
        "run_he_first_step_case.sbatch",
        "submit_matrix.sh",
        "submit_two_node_full_rank_10min.sh",
        "submit_build_barbora_petsc_env.sh",
        "submit_level4_one_node_1min_qexp.sh",
    )

    assert "--exclusive" not in text
    assert "--mem" not in text


def test_barbora_he_default_rank_density_matrix_is_documented():
    text = (SCRIPT_DIR / "submit_matrix.sh").read_text(encoding="utf-8")

    assert 'NODES_LIST="${NODES_LIST:-1 2 4 8 16}"' in text
    assert 'RPS_LIST="${RPS_LIST:-4 8 12 18}"' in text
    assert 'HE_LEVEL="${HE_LEVEL:-5}"' in text
    assert 'TIME_LIMIT="${TIME_LIMIT:-00:20:00}"' in text


def test_two_node_full_rank_wrapper_is_fixed_shape():
    text = (SCRIPT_DIR / "submit_two_node_full_rank_10min.sh").read_text(
        encoding="utf-8"
    )

    assert "export HE_LEVEL=5" in text
    assert "export NODES_LIST=2" in text
    assert "export RPS_LIST=18" in text
    assert "export TIME_LIMIT=00:10:00" in text
    assert "export BACKENDS=element" in text
    assert "export MAX_NODE_HOURS=1" in text
    assert 'exec "$SCRIPT_DIR/submit_matrix.sh"' in text


def test_level4_qexp_smoke_wrapper_is_fixed_shape():
    text = (SCRIPT_DIR / "submit_level4_one_node_1min_qexp.sh").read_text(
        encoding="utf-8"
    )

    assert "export HE_LEVEL=4" in text
    assert "export NODES_LIST=1" in text
    assert "export RPS_LIST=18" in text
    assert "export TIME_LIMIT=00:01:00" in text
    assert "export PARTITION=qcpu_exp" in text
    assert "export HE_SINGLE_NODE_SMOKE_TRANSPORT=1" in text
    assert 'exec "$SCRIPT_DIR/submit_matrix.sh"' in text


def test_barbora_env_build_pins_petsc324_stack():
    build_text = (SCRIPT_DIR / "build_barbora_petsc_env.sh").read_text(
        encoding="utf-8"
    )
    check_text = (SCRIPT_DIR / "check_barbora_env.sh").read_text(encoding="utf-8")
    env_text = (SCRIPT_DIR / "env_barbora.example.sh").read_text(encoding="utf-8")

    assert 'PETSC_VERSION="${PETSC_VERSION:-3.24.2}"' in build_text
    assert "foss/2022b Python/3.10.8-GCCcore-12.2.0" in build_text
    assert 'export CC="${BARBORA_PETSC_CC:-mpicc}"' in build_text
    assert 'CC="$CC"' in build_text
    assert "--download-cmake" in build_text
    assert "--download-hypre" in build_text
    assert "petsc4py" in build_text
    assert "jax[cpu]" in build_text
    assert "EXPECTED_PETSC_VERSION=\"${EXPECTED_PETSC_VERSION:-3.24.2}\"" in check_text
    assert "setPreallocationCOO" in check_text
    assert "local_env/prefix" in env_text


def test_barbora_shell_scripts_parse():
    for path in sorted(SCRIPT_DIR.glob("*.sh")) + sorted(SCRIPT_DIR.glob("*.sbatch")):
        subprocess.run(["bash", "-n", str(path)], check=True)
