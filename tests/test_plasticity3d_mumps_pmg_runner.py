from __future__ import annotations

import json
from pathlib import Path

from experiments.runners import run_plasticity3d_backend_mix_case as case_runner
from experiments.runners import (
    run_plasticity3d_p4_l1_2_lambda1p55_mumps_pmg_scaling as mumps_runner,
)


def test_local_pmg_mumps_profile_uses_redundant_lu_mumps(monkeypatch):
    monkeypatch.delenv("MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER", raising=False)
    monkeypatch.delenv("MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER", raising=False)

    profile = case_runner._local_pmg_linear_profile("local_pmg_mumps")

    assert profile.coarse_backend == "redundant_lu"
    assert profile.coarse_ksp_type == "preonly"
    assert profile.coarse_pc_type == "redundant"
    assert profile.coarse_redundant_number == 1
    assert profile.coarse_factor_solver_type == "mumps"


def test_local_pmg_hecoarse_alias_uses_same_mumps_profile():
    mumps_profile = case_runner._local_pmg_linear_profile("local_pmg_mumps")
    alias_profile = case_runner._local_pmg_linear_profile("local_pmg_hecoarse")

    assert alias_profile.coarse_backend == mumps_profile.coarse_backend
    assert alias_profile.coarse_pc_type == mumps_profile.coarse_pc_type
    assert alias_profile.coarse_factor_solver_type == mumps_profile.coarse_factor_solver_type


def test_backend_mix_parser_accepts_local_pmg_mumps(tmp_path: Path):
    args = case_runner._build_parser().parse_args(
        [
            "--assembly-backend",
            "local_constitutiveAD",
            "--solver-backend",
            "local_pmg_mumps",
            "--out-dir",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "output.json"),
        ]
    )

    assert args.solver_backend == "local_pmg_mumps"


def test_local_problem_args_use_p4_scatter_cache_auto_by_default(monkeypatch):
    monkeypatch.delenv("MIX_LOCAL_P4_CHUNK_SCATTER_CACHE", raising=False)
    monkeypatch.delenv("MIX_LOCAL_P4_CHUNK_SCATTER_CACHE_MAX_GIB", raising=False)
    monkeypatch.delenv("MIX_LOCAL_ASSEMBLY_BACKEND", raising=False)

    args = case_runner._local_problem_args()

    assert args.p4_chunk_scatter_cache == "auto"
    assert args.p4_chunk_scatter_cache_max_gib == 2.0
    assert args.assembly_backend == "coo"


def test_mumps_scaling_command_selects_canonical_case(tmp_path: Path):
    command = mumps_runner._build_case_command(
        source_root=tmp_path / "source",
        case_dir=tmp_path / "case",
        result_path=tmp_path / "case" / "output.json",
        ranks=32,
        maxit=5,
        grad_stop_tol=1.0e-2,
        ksp_rtol=1.0e-1,
        ksp_max_it=100,
        launcher="mpiexec",
        write_state=True,
    )

    assert command[:3] == ["mpiexec", "-n", "32"]
    assert command[command.index("--assembly-backend") + 1] == "local_constitutiveAD"
    assert command[command.index("--solver-backend") + 1] == "local_pmg_mumps"
    assert command[command.index("--mesh-name") + 1] == "hetero_ssr_L1_2"
    assert command[command.index("--constraint-variant") + 1] == "glued_bottom"
    assert command[command.index("--lambda-target") + 1] == "1.55"
    assert command[command.index("--maxit") + 1] == "5"
    assert "--state-out" in command


def test_mumps_scaling_env_records_redundant_mumps(tmp_path: Path):
    env = mumps_runner._build_env(
        source_root=tmp_path / "source",
        redundant_number=4,
        factor_solver="mumps",
        oversubscribe=True,
    )

    assert env["MIX_LOCAL_PMG_MUMPS_REDUNDANT_NUMBER"] == "4"
    assert env["MIX_LOCAL_PMG_MUMPS_FACTOR_SOLVER"] == "mumps"
    assert env["OMPI_MCA_rmaps_base_oversubscribe"] == "1"


def test_mumps_scaling_augments_summary_row_with_profile(tmp_path: Path):
    result = tmp_path / "output.json"
    result.write_text(
        json.dumps(
            {
                "pmg_linear_profile": {
                    "coarse_backend": "redundant_lu",
                    "coarse_ksp_type": "preonly",
                    "coarse_pc_type": "redundant",
                    "coarse_redundant_number": 1,
                    "coarse_factor_solver_type": "mumps",
                }
            }
        ),
        encoding="utf-8",
    )

    row = mumps_runner._augment_row({"case_id": "case"}, result)

    assert row["solver_profile"] == "same_mesh_p4_p2_p1_redundant_lu_mumps"
    assert row["pmg_coarse_backend"] == "redundant_lu"
    assert row["pmg_coarse_factor_solver_type"] == "mumps"
