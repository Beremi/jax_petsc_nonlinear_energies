from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path("experiments/runners")
SUBMITTER = SCRIPT_DIR / "submit_karolina_plasticity3d_mumps_pmg_scaling.sh"
SBATCH = SCRIPT_DIR / "run_karolina_plasticity3d_mumps_pmg_case.sbatch"
STEP_GRAD_SUBMITTER = (
    SCRIPT_DIR / "submit_karolina_plasticity3d_mumps_pmg_step_grad_convergence.sh"
)
STEP_GRAD_SBATCH = SCRIPT_DIR / "run_karolina_plasticity3d_mumps_pmg_step_grad_case.sbatch"
P2_CHAIN_SUBMITTER = (
    SCRIPT_DIR / "submit_karolina_plasticity3d_p2_p1chain_mumps_pmg_scaling.sh"
)
P2_CHAIN_SBATCH = SCRIPT_DIR / "run_karolina_plasticity3d_p2_p1chain_mumps_pmg_case.sbatch"


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


def test_karolina_plasticity3d_step_grad_submitter_plan(tmp_path: Path):
    out_root = tmp_path / "p3d_step_grad"
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "OUT_ROOT": str(out_root)})

    subprocess.run(["bash", str(STEP_GRAD_SUBMITTER)], check=True, env=env)

    rows = list(csv.DictReader((out_root / "campaign_plan.csv").open()))
    assert [row["nodes"] for row in rows] == ["1", "1"]
    assert [row["ranks_per_node"] for row in rows] == ["16", "32"]
    assert [row["total_ranks"] for row in rows] == ["16", "32"]
    assert {row["maxit"] for row in rows} == {"80"}
    assert {row["ksp_rtol"] for row in rows} == {"1e-2"}
    assert {row["ksp_max_it"] for row in rows} == {"200"}
    assert {row["convergence_mode"] for row in rows} == {"all"}
    assert {row["stop_tol"] for row in rows} == {"0.002"}
    assert {row["grad_stop_rtol"] for row in rows} == {"0.01"}
    assert {row["grad_stop_tol"] for row in rows} == {"0.0"}
    assert {row["redundant_number"] for row in rows} == {"1"}
    assert {row["factor_solver"] for row in rows} == {"mumps"}
    assert {row["partition"] for row in rows} == {"qcpu_exp"}
    assert {row["qos"] for row in rows} == {"3571_6328"}
    assert {row["mesh_name"] for row in rows} == {"hetero_ssr_L1_2"}
    assert {row["elem_degree"] for row in rows} == {"4"}
    assert {row["pmg_strategy"] for row in rows} == {"same_mesh_p4_p2_p1"}
    assert {row["exclusive_node"] for row in rows} == {"1"}
    assert rows[0]["time_limit"] == "01:00:00"
    assert rows[1]["time_limit"] == "00:45:00"
    assert (
        rows[0]["cpu_map"]
        == "custom_map_cpu:0;1;16;17;32;33;48;49;64;65;80;81;96;97;112;113"
    )
    assert (
        rows[1]["cpu_map"]
        == "custom_map_cpu:0;1;2;3;16;17;18;19;32;33;34;35;48;49;50;51;64;65;66;67;80;81;82;83;96;97;98;99;112;113;114;115"
    )

    commands = [
        line
        for line in (out_root / "sbatch_commands.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(commands) == 2
    assert all("run_karolina_plasticity3d_mumps_pmg_step_grad_case.sbatch" in command for command in commands)
    assert any("--ntasks-per-node 16" in command for command in commands)
    assert any("--ntasks-per-node 32" in command for command in commands)
    assert all("--nodes 1" in command for command in commands)
    assert all("--exclusive" in command for command in commands)
    assert all("CONVERGENCE_MODE=all" in command for command in commands)
    assert all("GRAD_STOP_RTOL=0.01" in command for command in commands)


def test_karolina_plasticity3d_step_grad_runner_shape():
    text = STEP_GRAD_SBATCH.read_text(encoding="utf-8")

    assert "same_mesh_p4_p2_p1" in text
    assert "hetero_ssr_L1_2" in text
    assert 'ELEM_DEGREE="${ELEM_DEGREE:-4}"' in text
    assert 'KSP_RTOL="${KSP_RTOL:-1e-2}"' in text
    assert 'KSP_MAX_IT="${KSP_MAX_IT:-200}"' in text
    assert 'CONVERGENCE_MODE="${CONVERGENCE_MODE:-all}"' in text
    assert 'STOP_TOL="${STOP_TOL:-0.002}"' in text
    assert 'GRAD_STOP_RTOL="${GRAD_STOP_RTOL:-0.01}"' in text
    assert 'GRAD_STOP_TOL="${GRAD_STOP_TOL:-0.0}"' in text
    assert 'MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER="$REDUNDANT_NUMBER"' in text
    assert 'MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER="$FACTOR_SOLVER"' in text
    assert "ensure_same_mesh_case_hdf5" in text
    assert "rank_host_order.csv" in text
    assert "rank_node_layout.json" in text
    assert "--assembly-backend local_constitutiveAD" in text
    assert "--solver-backend local_pmg_mumps" in text
    assert '--mesh-name "$MESH_NAME"' in text
    assert '--elem-degree "$ELEM_DEGREE"' in text
    assert '--pmg-strategy "$PMG_STRATEGY"' in text
    assert '--convergence-mode "$CONVERGENCE_MODE"' in text
    assert '--grad-stop-rtol "$GRAD_STOP_RTOL"' in text
    assert '--grad-stop-tol "$GRAD_STOP_TOL"' in text
    assert '--stop-tol "$STOP_TOL"' in text
    assert '--maxit "$MAXIT"' in text
    assert '--line-search "$LINE_SEARCH"' in text


def test_karolina_plasticity3d_step_grad_scripts_parse():
    subprocess.run(["bash", "-n", str(STEP_GRAD_SUBMITTER)], check=True)
    subprocess.run(["bash", "-n", str(STEP_GRAD_SBATCH)], check=True)


def test_karolina_plasticity3d_p2_p1chain_submitter_plan(tmp_path: Path):
    out_root = tmp_path / "p2_chain"
    env = os.environ.copy()
    env.update({"DRY_RUN": "1", "OUT_ROOT": str(out_root)})

    subprocess.run(["bash", str(P2_CHAIN_SUBMITTER)], check=True, env=env)

    rows = list(csv.DictReader((out_root / "campaign_plan.csv").open()))
    assert [row["nodes"] for row in rows] == ["1", "2", "4", "8"]
    assert [row["partition"] for row in rows] == ["qcpu_exp", "qcpu_exp", "qcpu", "qcpu"]
    assert [row["total_ranks"] for row in rows] == ["16", "32", "64", "128"]
    assert [row["redundant_number"] for row in rows] == ["1", "2", "4", "8"]
    assert {row["ranks_per_node"] for row in rows} == {"16"}
    assert {row["factor_solver"] for row in rows} == {"mumps"}
    assert {row["maxit"] for row in rows} == {"5"}
    assert {row["mesh_name"] for row in rows} == {"hetero_ssr_L1_2_3"}
    assert {row["elem_degree"] for row in rows} == {"2"}
    assert {row["pmg_strategy"] for row in rows} == {"uniform_refined_p2_p1_chain"}
    assert {row["exclusive_node"] for row in rows} == {"1"}
    assert rows[0]["time_limit"] == "00:20:00"
    assert rows[1]["time_limit"] == "00:14:00"
    assert rows[2]["time_limit"] == "00:10:00"
    assert rows[3]["time_limit"] == "00:10:00"
    assert {
        row["cpu_map"]
        for row in rows
    } == {"custom_map_cpu:0;1;16;17;32;33;48;49;64;65;80;81;96;97;112;113"}

    commands = [
        line
        for line in (out_root / "sbatch_commands.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(commands) == 4
    assert all("run_karolina_plasticity3d_p2_p1chain_mumps_pmg_case.sbatch" in command for command in commands)
    assert all("--ntasks-per-node 16" in command for command in commands)
    assert all("--exclusive" in command for command in commands)
    assert all("--distribution" not in command for command in commands)
    assert any("--nodes 8" in command for command in commands)
    assert any("p3d_p2l123_p1chain_mumps_n8_rpn16_np128" in command for command in commands)


def test_karolina_plasticity3d_p2_p1chain_runner_shape():
    text = P2_CHAIN_SBATCH.read_text(encoding="utf-8")

    assert "uniform_refined_p2_p1_chain" in text
    assert "hetero_ssr_L1_2_3" in text
    assert "hetero_ssr_L1_2" in text
    assert "hetero_ssr_L1" in text
    assert 'ELEM_DEGREE="${ELEM_DEGREE:-2}"' in text
    assert 'PREPARE_SAME_MESH_HDF5="${PREPARE_SAME_MESH_HDF5:-0}"' in text
    assert 'HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"' in text
    assert "--exclusive" in text
    assert "--distribution" not in text
    assert 'MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER="$REDUNDANT_NUMBER"' in text
    assert 'MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER="$FACTOR_SOLVER"' in text
    assert "--assembly-backend local_constitutiveAD" in text
    assert "--solver-backend local_pmg_mumps" in text
    assert '--mesh-name "$MESH_NAME"' in text
    assert '--elem-degree "$ELEM_DEGREE"' in text
    assert '--pmg-strategy "$PMG_STRATEGY"' in text
    assert "rank_host_order.csv" in text
    assert "rank_node_layout.json" in text


def test_karolina_plasticity3d_p2_p1chain_scripts_parse():
    subprocess.run(["bash", "-n", str(P2_CHAIN_SUBMITTER)], check=True)
    subprocess.run(["bash", "-n", str(P2_CHAIN_SBATCH)], check=True)
