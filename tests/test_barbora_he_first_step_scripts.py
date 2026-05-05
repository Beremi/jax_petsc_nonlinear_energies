from __future__ import annotations

import csv
import os
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
        "submit_level4_one_node_socket_scaling.sh",
        "run_he_first_step_socket_case.sbatch",
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


def test_barbora_he_sbatch_defaults_to_distributed_element_path():
    text = _script_text(
        "run_he_first_step_case.sbatch",
        "run_he_first_step_socket_case.sbatch",
    )

    assert 'HE_PROBLEM_BUILD_MODE="${HE_PROBLEM_BUILD_MODE:-rank_local}"' in text
    assert 'HE_DISTRIBUTION_STRATEGY="${HE_DISTRIBUTION_STRATEGY:-overlap_p2p}"' in text
    assert 'HE_ASSEMBLY_BACKEND="${HE_ASSEMBLY_BACKEND:-coo_local}"' in text
    assert 'HE_LINE_SEARCH="${HE_LINE_SEARCH:-armijo}"' in text
    assert 'HE_TRUST_RADIUS_INIT="${HE_TRUST_RADIUS_INIT:-1.0}"' in text
    assert 'HE_TOLX_REL="${HE_TOLX_REL:-1e-4}"' in text
    assert '--problem-build-mode "$HE_PROBLEM_BUILD_MODE"' in text
    assert '--distribution-strategy "$HE_DISTRIBUTION_STRATEGY"' in text
    assert '--assembly-backend "$HE_ASSEMBLY_BACKEND"' in text
    assert '--line-search "$HE_LINE_SEARCH"' in text
    assert '--trust-radius-init "$HE_TRUST_RADIUS_INIT"' in text
    assert '--tolx-rel "$HE_TOLX_REL"' in text


def test_level4_socket_scaling_wrapper_shape_and_caps(tmp_path):
    script = SCRIPT_DIR / "submit_level4_one_node_socket_scaling.sh"
    out_root = tmp_path / "socket_scaling"
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "OUT_ROOT": str(out_root)})

    subprocess.run(["bash", str(script)], check=True, env=env)

    text = script.read_text(encoding="utf-8")
    assert 'PARTITION="${PARTITION:-qcpu_exp}"' in text
    assert 'HE_LEVEL="${HE_LEVEL:-4}"' in text
    assert 'BACKEND="${BACKEND:-element}"' in text
    assert 'BASELINE_STEP_S="${BASELINE_STEP_S:-35}"' in text
    assert 'SLURM_OVERHEAD_S="${SLURM_OVERHEAD_S:-60}"' in text
    assert "--sockets-per-node" in text
    assert "--ntasks-per-socket" in text
    runner_text = (SCRIPT_DIR / "run_he_first_step_socket_case.sbatch").read_text(
        encoding="utf-8"
    )
    assert "--step-time-limit-s" in runner_text
    assert "--cpu-bind=\"map_cpu:${CPU_MAP}\"" in runner_text
    assert 'SRUN_PLACEMENT_ARGS=(--sockets-per-node="$ACTIVE_SOCKETS")' in runner_text
    assert "SRUN_STEP_CPUS_PER_TASK=2" in runner_text
    assert 'echo "srun_step_cpus_per_task=$SRUN_STEP_CPUS_PER_TASK"' in runner_text
    assert "--ntasks-per-socket" in runner_text

    plan_rows = list(csv.DictReader((out_root / "campaign_plan.csv").open()))
    valid = {row["layout"]: row for row in plan_rows if row["status"] == "valid"}
    invalid = {row["layout"]: row for row in plan_rows if row["status"] == "invalid"}

    assert set(valid) == {"18+18", "18+0", "9+9", "9+0"}
    assert invalid["36+0"]["reason"] == "invalid_without_oversubscription_on_18_core_socket"
    assert valid["18+18"]["step_time_limit_s"] == "35"
    assert valid["18+18"]["slurm_time_limit"] == "00:01:35"
    assert valid["18+18"]["cpu_map"] == ",".join(str(i) for i in range(36))
    assert valid["18+0"]["step_time_limit_s"] == "140"
    assert valid["18+0"]["slurm_time_limit"] == "00:03:20"
    assert valid["18+0"]["cpu_map"] == ",".join(str(i) for i in range(18))
    assert valid["9+9"]["step_time_limit_s"] == "70"
    assert valid["9+9"]["slurm_time_limit"] == "00:02:10"
    assert valid["9+9"]["cpu_map"] == ",".join(
        str(i) for i in list(range(9)) + list(range(18, 27))
    )
    assert valid["9+0"]["step_time_limit_s"] == "280"
    assert valid["9+0"]["slurm_time_limit"] == "00:05:40"
    assert valid["9+0"]["cpu_map"] == ",".join(str(i) for i in range(9))

    commands = [
        line
        for line in (out_root / "sbatch_commands.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(commands) == 4
    assert not any("36+0" in command for command in commands)
    expected_9p9_map = "0\\,1\\,2\\,3\\,4\\,5\\,6\\,7\\,8\\,18\\,19\\,20\\,21\\,22\\,23\\,24\\,25\\,26"
    assert any(expected_9p9_map in command for command in commands)


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
