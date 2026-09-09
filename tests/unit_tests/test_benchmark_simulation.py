"""Timing gates must reject slowdown and incompatible or profiled workloads."""

import copy

import pytest
import torch

from scripts.benchmark_simulation import (
    build_workload,
    compare_pairs,
    profiled_steps,
    state_evidence,
)


def result(seconds):
    return {
        "schema": 1,
        "protocol": {"steps": 100, "profile": "task_empty"},
        "config": {"sim_dt": 0.01},
        "hardware": {"gpu": "test-device"},
        "packages": {"engine": "test-version"},
        "native_topology": None,
        "reset_count": 0,
        "capture": False,
        "median_s": seconds,
        "initial_state": {
            "root_states": [[0.0]],
            "dof_pos": [[0.4]],
            "dof_vel": [[0.0]],
            "contact_forces": [[1.0]],
        },
    }


def test_paired_process_gate_detects_slowdown_and_noise():
    reference = [result(1)] * 5
    assert compare_pairs(reference, [result(1.01)] * 5)["status"] == "pass"
    assert compare_pairs(reference, [result(1.2)] * 5)["status"] == "regression"
    mixed = [result(t) for t in (0.95, 1.01, 1.04, 1.08, 1.15)]
    assert compare_pairs(reference, mixed)["status"] == "inconclusive"
    assert compare_pairs(reference[:1], reference[:1])["status"] == "insufficient_pairs"


@pytest.mark.parametrize(
    "key",
    ["protocol", "config", "hardware", "packages", "native_topology", "reset_count"],
)
def test_mismatched_workload_cannot_appear_as_a_speedup(key):
    reference, candidate = result(1), result(0.1)
    candidate[key] = "changed"
    with pytest.raises(ValueError, match=f"incomparable benchmark {key}"):
        compare_pairs([reference] * 5, [candidate] * 5)


def test_profiler_overhead_is_ineligible_for_speed_comparison():
    reference = result(1)
    candidate = copy.deepcopy(reference)
    candidate["capture"] = True
    with pytest.raises(ValueError, match="profiler timings"):
        compare_pairs([reference] * 5, [candidate] * 5)


def test_different_initial_physics_cannot_appear_as_a_speedup():
    reference, candidate = result(1), result(0.1)
    candidate["initial_state"]["dof_pos"] = [[0.0]]
    with pytest.raises(ValueError, match="incomparable initial state: dof_pos"):
        compare_pairs([reference] * 5, [candidate] * 5)


def test_derived_contact_roundoff_does_not_change_requested_initial_conditions():
    reference, candidate = result(1), result(1)
    candidate["initial_state"]["contact_forces"] = [[1.0002]]
    assert compare_pairs([reference] * 5, [candidate] * 5)["status"] == "pass"


@pytest.mark.parametrize("profile", ["backend_step", "task_empty", "task_timeout"])
def test_cpu_workload_advances_and_restores_physics_and_task_state(profile):
    workload = build_workload("pendulum", "cpu", 8, 7, 1, profile, 20)
    try:
        initial = workload.env.dof_state.clone()
        profiled_steps(workload, 20)
        advanced = workload.env.dof_state.clone()
        assert not torch.allclose(initial, advanced)
        workload.restore()
        torch.testing.assert_close(workload.env.dof_state, initial, rtol=0, atol=0)
        assert workload.env.common_step_counter == 0
        assert not workload.env.episode_length_buf.any()
        profiled_steps(workload, 20)
        torch.testing.assert_close(workload.env.dof_state, advanced, rtol=0, atol=0)
        state_evidence(workload)
    finally:
        workload.close()


def test_scheduled_timeout_masks_have_exact_rate_and_staggered_envs():
    workload = build_workload("pendulum", "cpu", 8, 7, 1, "task_timeout", 1000)
    try:
        assert sum(int(mask.sum()) for mask in workload.masks) == 8
        assert torch.stack(workload.masks).sum(dim=0).tolist() == [1] * 8
    finally:
        workload.close()
