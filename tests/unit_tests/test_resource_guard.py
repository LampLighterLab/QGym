"""Small policy tests; real cgroup enforcement is checked with bounded workers."""

import json
from pathlib import Path

import pytest

from scripts import run_resource_limited as guard


def test_numeric_limit_rejects_unlimited_controller(tmp_path):
    limit = tmp_path / "memory.max"
    limit.write_text("max\n")
    with pytest.raises(guard.GuardError, match="numeric limit"):
        guard.numeric_limit(limit)
    limit.write_text(f"{guard.MEMORY_MAX}\n")
    assert guard.numeric_limit(limit) == guard.MEMORY_MAX


def test_kernel_limits_must_match_shared_policy(tmp_path):
    (tmp_path / "memory.max").write_text(str(guard.MEMORY_MAX))
    (tmp_path / "memory.swap.max").write_text(str(guard.SWAP_MAX))
    assert guard.verify_limits(tmp_path)["memory_max_bytes"] == 40 * guard.GIB
    (tmp_path / "memory.swap.max").write_text("max")
    with pytest.raises(guard.GuardError):
        guard.verify_limits(tmp_path)


def test_host_reserve_stops_at_twelve_gib_boundary():
    before = {"memory_available_bytes": guard.MIN_AVAILABLE, "swap_io_bytes": 0}
    assert guard.stop_reason(before, before, 2.0, 0.0)[0] is None
    after = {**before, "memory_available_bytes": guard.MIN_AVAILABLE - 1}
    assert (
        guard.stop_reason(after, before, 2.0, 0.0)[0] == "host_available_below_12_gib"
    )


def test_swap_io_requires_sustained_activity():
    before = {"memory_available_bytes": guard.MIN_AVAILABLE, "swap_io_bytes": 0}
    after = {**before, "swap_io_bytes": guard.SWAP_IO_BYTES_PER_SECOND * 2}
    assert guard.stop_reason(after, before, 2.0, 6.0) == (None, 8.0)
    assert guard.stop_reason(after, before, 2.0, 8.0)[0] == "sustained_host_swap_io"
    assert guard.stop_reason(before, before, 2.0, 8.0) == (None, 0.0)


def test_parallel_workers_require_complete_peak_evidence():
    with pytest.raises(guard.GuardError, match="one worker"):
        guard.validate_command(["python3", "-m", "scripts.example", "--jobs", "6"])
    assert guard.command_worker_count(["python3", "run.py", "--workers=3"]) == 3
    with pytest.raises(guard.GuardError, match="positive worker count"):
        guard.command_worker_count(["python3", "run.py", "--jobs", "auto"])


def test_peak_budget_has_fifty_percent_headroom_and_aggregate_bound(tmp_path):
    path = tmp_path / "complete.json"
    evidence = {
        "status": "completed",
        "peak_is_complete_worker_measurement": True,
        "worker_count": 1,
        "worker": {"memory_peak_bytes": 10 * guard.GIB},
    }
    path.write_text(json.dumps(evidence))
    assert guard.projected_memory_budget(path, 2) == 30 * guard.GIB
    with pytest.raises(guard.GuardError, match="shared RAM cap"):
        guard.projected_memory_budget(path, 3)
    evidence["peak_is_complete_worker_measurement"] = False
    path.write_text(json.dumps(evidence))
    with pytest.raises(guard.GuardError, match="successful complete"):
        guard.projected_memory_budget(path, 1)


def test_python_venv_symlink_is_preserved(tmp_path):
    executable = tmp_path / "python"
    executable.symlink_to("/usr/bin/python3")
    assert guard.validate_command([str(executable), "-V"])[0] == str(executable)


def test_wrong_service_group_fails_before_spawning_workload(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "current_cgroup", lambda: Path("/unlimited/editor"))
    calls = []
    monkeypatch.setattr(
        guard.subprocess, "run", lambda *_args, **_kwargs: calls.append(1)
    )
    with pytest.raises(guard.GuardError, match="outside its assigned"):
        guard.service_worker(
            [str(tmp_path), str(tmp_path / "result"), str(tmp_path / "stop"), "true"]
        )
    assert not calls


def test_failed_systemctl_stop_kills_only_own_service(tmp_path, monkeypatch):
    service = tmp_path / "codex-worker-example.service"
    service.mkdir()
    own_kill = service / "cgroup.kill"
    own_kill.write_text("")
    shared_kill = tmp_path / "cgroup.kill"
    shared_kill.write_text("")

    def failed_stop(*_args):
        raise guard.GuardError("User bus is unavailable")

    def remaining_pids(path):
        assert path == service
        return [] if own_kill.read_text() == "1" else [123]

    monkeypatch.setattr(guard, "systemctl", failed_stop)
    monkeypatch.setattr(guard, "cgroup_pids", remaining_pids)
    guard.stop_service(service.name, tmp_path / "stop", tmp_path)
    assert own_kill.read_text() == "1"
    assert shared_kill.read_text() == ""
