from __future__ import annotations

from pathlib import Path

from experiments.runners import run_plasticity3d_validation as runner


def test_validation_runner_builds_expected_commands(tmp_path: Path) -> None:
    source_root = Path("/tmp/source-root")
    maintained_cmd = runner._maintained_command(
        source_root=source_root,
        case_dir=tmp_path / "maintained",
        output_json=tmp_path / "maintained" / "output.json",
        state_npz=tmp_path / "maintained" / "state.npz",
        lambda_target=1.55,
        ranks=1,
        stop_tol=2.0e-3,
        maxit=80,
    )
    source_cmd = runner._source_command(
        source_root=source_root,
        case_dir=tmp_path / "source",
        output_json=tmp_path / "source" / "output.json",
        lambda_target=1.55,
        ranks=1,
        stop_tol=2.0e-3,
        maxit=80,
    )
    assert "--assembly-backend" in maintained_cmd
    assert "local_constitutiveAD" in maintained_cmd
    assert "--state-out" in maintained_cmd
    assert "--elem-type" in source_cmd
    assert "P2" in source_cmd
    assert "--mesh-boundary-type" in source_cmd
    assert source_cmd[source_cmd.index("--mesh-boundary-type") + 1] == "1"
    assert "--stopping-criterion" in source_cmd
    assert "relative_correction" in source_cmd
