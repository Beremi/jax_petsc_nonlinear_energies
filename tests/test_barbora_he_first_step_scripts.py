from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path("experiments/runners/barbora_he_first_step_scaling")


def test_barbora_he_scripts_avoid_disallowed_slurm_options():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            SCRIPT_DIR / "run_he_first_step_case.sbatch",
            SCRIPT_DIR / "submit_matrix.sh",
            SCRIPT_DIR / "submit_two_node_full_rank_10min.sh",
        ]
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
