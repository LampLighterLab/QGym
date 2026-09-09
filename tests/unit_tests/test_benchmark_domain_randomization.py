from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_domain_randomization import (
    BUNDLE_MODES,
    make_reset_schedule,
    native_topology_details,
)
from scripts.profile_vsim_domain_randomization import validate_results


class FakeGroup:
    def __init__(self, sizes):
        self.sizes = sizes

    def get_num_environment_sets(self):
        return len(self.sizes)

    def get_num_environments(self):
        return self.sizes


def result(topology, friction=None):
    return {
        "protocol": {"native_topology_details": topology},
        "applied_domain_randomization": {"contact_friction": friction},
    }


def test_fixed_friction_uses_physical_friction_topology_mode():
    assert BUNDLE_MODES["friction-fixed"] == "friction-only"


def test_reset_schedule_uses_fixed_shape_boolean_masks():
    schedule = make_reset_schedule("timeout", 8, 4, 4, "cpu")

    assert all(mask.dtype == torch.bool and mask.shape == (8,) for mask in schedule)
    assert [mask.count_nonzero().item() for mask in schedule] == [2, 2, 2, 2]
    torch.testing.assert_close(
        torch.stack(schedule).sum(dim=0),
        torch.ones(8, dtype=torch.long),
    )


def test_vsim_topology_details_reads_native_group():
    env = SimpleNamespace(_backend=SimpleNamespace(_grp=FakeGroup([1, 1, 1])))

    assert native_topology_details(env, "vsim") == {
        "environment_set_count": 3,
        "total_environment_count": 3,
        "min_environments_per_set": 1,
        "max_environments_per_set": 1,
    }
    assert native_topology_details(env, "warp") is None


def test_capture_validation_accepts_declared_topologies_and_fixed_friction():
    one = {
        "environment_set_count": 1,
        "total_environment_count": 4,
        "min_environments_per_set": 4,
        "max_environments_per_set": 4,
    }
    many = {
        "environment_set_count": 4,
        "total_environment_count": 4,
        "min_environments_per_set": 1,
        "max_environments_per_set": 1,
    }
    results = {
        "one_set_off": result(one),
        "many_sets_fixed_friction": result(
            many,
            {"min": 1.0, "mean": 1.0, "max": 1.0, "std": 0.0},
        ),
        "many_sets_sampled_friction": result(many),
        "one_set_episode_pd": result(one),
    }

    validate_results(results, 4)


def test_capture_validation_rejects_wrong_physical_topology():
    one = {
        "environment_set_count": 1,
        "total_environment_count": 4,
        "min_environments_per_set": 4,
        "max_environments_per_set": 4,
    }
    results = {
        "one_set_off": result(one),
        "many_sets_fixed_friction": result(
            one,
            {"min": 1.0, "mean": 1.0, "max": 1.0, "std": 0.0},
        ),
        "many_sets_sampled_friction": result(one),
        "one_set_episode_pd": result(one),
    }

    with pytest.raises(RuntimeError, match="singleton sets"):
        validate_results(results, 4)
