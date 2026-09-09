from pathlib import Path
import sys

import pytest

from scripts.eval_go2_policy import (
    main,
    parse_labeled_path,
    resolve_evaluation_backend,
    resolve_policy_checkpoints,
)


def test_parse_labeled_policy_preserves_path():
    label, path = parse_labeled_path("base gait=logs/go2trot/run")

    assert label == "base gait"
    assert path == Path("logs/go2trot/run")


def test_checkpoint_selection_supports_explicit_and_latest(tmp_path):
    for iteration in (0, 50, 100):
        (tmp_path / f"model_{iteration}.pt").touch()

    selected = resolve_policy_checkpoints(tmp_path, ["50", "latest"])

    assert [iteration for _, iteration in selected] == [50, 100]


def test_checkpoint_selection_reports_missing_iteration(tmp_path):
    (tmp_path / "model_50.pt").touch()

    with pytest.raises(FileNotFoundError, match="available: 50"):
        resolve_policy_checkpoints(tmp_path, ["100"])


@pytest.mark.parametrize(
    ("public_name", "expected"),
    [
        ("mjx", ("mujoco", "cuda:0", "mujoco-cuda:0")),
        ("mujoco", ("mujoco", "cpu", "mujoco-cpu")),
        ("vsim", ("vsim", "cuda:0", "vsim")),
    ],
)
def test_public_evaluation_backend_selects_engine_device_and_label(
    public_name, expected
):
    assert resolve_evaluation_backend(public_name) == expected


def test_go2trot_evaluation_records_policy_io_by_default(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_go2_policy.py",
            "--policy",
            f"test={checkpoint}",
            "--num_envs",
            "10",
            "--out_dir",
            str(tmp_path / "evaluation"),
            "--dry_run",
        ],
    )

    main()

    command = capsys.readouterr().out
    assert "--task go2trot" in command
    assert "--record_policy_io" in command
    assert "--eval_backend mujoco --eval_device cuda:0" in command
    assert "--eval_label mujoco-cuda:0" in command
