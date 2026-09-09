"""Exercise the SDK bootstrap against a tiny local Git repository, offline."""

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.fetch_unitree_sdk import fetch_sdk


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_unitree_sdk.py"


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def project(tmp_path):
    remote = tmp_path / "local remote"
    remote.mkdir()
    git(remote, "init")
    (remote / "sdk.py").write_text("PINNED = True\n")
    (remote / ".gitignore").write_text("*.egg-info/\n__pycache__/\n")
    git(remote, "add", "sdk.py", ".gitignore")
    git(
        remote,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "pinned SDK",
    )
    revision = git(remote, "rev-parse", "HEAD")
    (remote / "sdk.py").write_text("PINNED = False\n")
    git(remote, "add", "sdk.py")
    git(
        remote,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "newer SDK",
    )
    root = tmp_path / "Q2 project"
    root.mkdir()
    # Both the pin and destination come from the supplied project configuration.
    (root / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        'unitree-sdk2py = { path = "vendor/sdk", editable = true }\n'
        "[tool.q2.unitree-sdk]\n"
        f'git = "{remote.as_posix()}"\n'
        f'rev = "{revision}"\n'
    )
    return root, remote, revision


def test_fetch_checks_out_pinned_revision_and_allows_ignored_cache(project):
    root, _, revision = project

    target = fetch_sdk(root)

    assert target == root / "vendor" / "sdk"
    assert git(target, "rev-parse", "HEAD") == revision
    assert git(target, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert (target / "sdk.py").read_text() == "PINNED = True\n"
    ignored = target / "__pycache__" / "sdk.pyc"
    ignored.parent.mkdir()
    ignored.write_bytes(b"cache")
    assert fetch_sdk(root) == target
    assert ignored.read_bytes() == b"cache"


@pytest.mark.parametrize("staged", [False, True])
def test_existing_tracked_changes_are_rejected_without_overwrite(project, staged):
    root, _, revision = project
    target = fetch_sdk(root)
    source = target / "sdk.py"
    source.write_text("local changes\n")
    if staged:
        git(target, "add", "sdk.py")
    before = git(target, "status", "--porcelain")

    with pytest.raises(RuntimeError, match="has local changes"):
        fetch_sdk(root)

    assert source.read_text() == "local changes\n"
    assert git(target, "status", "--porcelain") == before
    assert git(target, "rev-parse", "HEAD") == revision


def test_untracked_source_is_rejected_without_deletion(project):
    root, _, _ = project
    target = fetch_sdk(root)
    git(target, "config", "status.showUntrackedFiles", "no")
    untracked = target / "shadow.py"
    untracked.write_text("local code\n")

    with pytest.raises(RuntimeError, match="has local changes"):
        fetch_sdk(root)

    assert untracked.read_text() == "local code\n"


def test_existing_wrong_revision_is_rejected_without_reset(project):
    root, remote, _ = project
    target = fetch_sdk(root)
    latest = git(remote, "rev-parse", "HEAD")
    git(target, "checkout", "--detach", latest)

    with pytest.raises(RuntimeError, match="pyproject.toml requires"):
        fetch_sdk(root)

    assert git(target, "rev-parse", "HEAD") == latest
    assert (target / "sdk.py").read_text() == "PINNED = False\n"


def test_existing_non_checkout_is_preserved(project):
    root, _, _ = project
    target = root / "vendor" / "sdk"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("existing work\n")

    with pytest.raises(RuntimeError, match="not a Git checkout"):
        fetch_sdk(root)

    assert (target / "keep.txt").read_text() == "existing work\n"
    assert not (target / ".git").exists()


def test_standalone_cli_reports_next_steps_and_actionable_failure(project):
    root, _, _ = project
    command = [sys.executable, str(SCRIPT), "--project-root", str(root)]

    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "uv sync --frozen --extra unitree_sdk" in result.stdout
    target = root / "vendor" / "sdk"
    (target / "sdk.py").write_text("local changes\n")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "has local changes" in failed.stderr
    assert "Traceback" not in failed.stderr
    assert (target / "sdk.py").read_text() == "local changes\n"
