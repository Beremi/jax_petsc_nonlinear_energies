from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path("experiments/runners")
SUBMITTER = SCRIPT_DIR / "submit_karolina_plasticity3d_mumps_pmg_scaling.sh"
SBATCH = SCRIPT_DIR / "run_karolina_plasticity3d_mumps_pmg_case.sbatch"


def test_karolina_plasticity3d_mumps_pmg_submitter_plan(tmp_path: Path):
    out_root = tmp_path / "p3d"
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "OUT_ROOT": str(out_root)})

    subprocess.run(["bash", str(SUBMITTER)], check=True, env=env)

    rows = list(csv.DictReader((out_root / "campaign_plan.csv").open()))
    assert [row["nodes"] for row in rows] == ["1", "2", "4", "8"]
    assert [row["total_ranks"] for row in rows] == ["128", "256", "512", "1024"]
    assert [row["redundant_number"] for row in rows] == ["1", "2", "4", "8"]
    assert {row["factor_solver"] for row in rows} == {"mumps"}
    assert {row["maxit"] for row in rows} == {"5"}
    assert {row["ranks_per_node"] for row in rows} == {"128"}
    assert {row["partition"] for row in rows} == {"qcpu_exp"}
    assert {row["qos"] for row in rows} == {"3571_6328"}
    assert rows[0]["time_limit"] == "00:12:00"
    assert rows[1]["time_limit"] == "00:10:00"
    assert rows[2]["time_limit"] == "00:08:00"
    assert rows[3]["time_limit"] == "00:08:00"
    assert {row["cpu_map"] for row in rows} == {"map_cpu:0-127"}
    assert {row["mem_bind"] for row in rows} == {"local"}

    commands = [
        line
        for line in (out_root / "sbatch_commands.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(commands) == 4
    assert all("--ntasks-per-node 128" in command for command in commands)
    assert all("--distribution block:block" in command for command in commands)
    assert any("--nodes 8" in command for command in commands)
    assert any("p3d_p4l12_mumps_n8_np1024" in command for command in commands)


def test_karolina_plasticity3d_mumps_pmg_runner_shape():
    text = SBATCH.read_text(encoding="utf-8")

    assert "--cpu-bind=\"map_cpu:${KAROLINA_CPU_MAP}\"" in text
    assert "--mem-bind=\"$KAROLINA_MEM_BIND\"" in text
    assert "--distribution=block:block" in text
    assert 'MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER="$REDUNDANT_NUMBER"' in text
    assert 'MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER="$FACTOR_SOLVER"' in text
    assert 'PREPARE_SAME_MESH_HDF5="${PREPARE_SAME_MESH_HDF5:-1}"' in text
    assert "ensure_same_mesh_case_hdf5" in text
    assert "hetero_ssr_L1_2" in text
    assert ".karolina_l1_2_glued_same_mesh.lock" in text
    assert "rank_host_order.csv" in text
    assert "rank_node_layout.json" in text
    assert "--assembly-backend local_constitutiveAD" in text
    assert "--solver-backend local_pmg_mumps" in text
    assert '--mesh-name "$MESH_NAME"' in text
    assert '--constraint-variant "$CONSTRAINT_VARIANT"' in text
    assert '--lambda-target "$LAMBDA_TARGET"' in text
    assert '--maxit "$MAXIT"' in text


def test_karolina_plasticity3d_mumps_pmg_scripts_parse():
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
