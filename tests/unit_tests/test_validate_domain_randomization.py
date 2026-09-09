from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval_policy import configure_contact_friction_dr
from scripts.validate_domain_randomization import (
    BACKENDS,
    cell_command,
    compare_vectors,
    ks_distance,
    wasserstein_distance,
)


def test_eval_contact_friction_mode_is_resolved_before_setup():
    startup = SimpleNamespace(
        contact_friction_range=[0.5, 1.0], link_mass_scale_range=None
    )
    cfg = SimpleNamespace(
        domain_randomization=SimpleNamespace(
            startup=startup,
            episode=SimpleNamespace(scale_ranges={}),
        )
    )

    configure_contact_friction_dr(cfg, "off")
    assert startup.contact_friction_range is None
    configure_contact_friction_dr(cfg, "on", [0.3, 0.9])
    assert startup.contact_friction_range == [0.3, 0.9]

    with pytest.raises(ValueError, match="cannot be combined"):
        configure_contact_friction_dr(cfg, "off", [0.3, 0.9])


def test_distribution_distances_are_zero_for_identical_samples():
    values = np.asarray([0.0, 0.5, 1.0, 2.0])

    assert wasserstein_distance(values, values) == 0.0
    assert ks_distance(values, values) == 0.0
    comparison = compare_vectors(values, values, "mean_reward")
    assert comparison["normalized_wasserstein"] == 0.0
    assert comparison["paired_rmse"] == 0.0
    assert comparison["agreement"] == "close"


def test_distribution_distances_quantify_a_constant_shift():
    first = np.asarray([0.0, 1.0, 2.0])
    second = first + 1.0

    assert wasserstein_distance(first, second) == pytest.approx(1.0)
    assert ks_distance(first, second) == pytest.approx(1.0 / 3.0)
    comparison = compare_vectors(first, second, "mean_reward")
    assert comparison["marginal_mean_delta_second_minus_first"] == pytest.approx(1.0)
    assert comparison["mean_delta_second_minus_first"] == pytest.approx(1.0)
    assert comparison["paired_mae"] == pytest.approx(1.0)
    assert comparison["paired_rmse"] == pytest.approx(1.0)
    assert comparison["normalized_wasserstein"] == pytest.approx(1.0)
    assert comparison["agreement"] == "large"


def test_marginal_and_paired_deltas_are_reported_separately():
    comparison = compare_vectors(
        np.asarray([0.0, 2.0, np.nan]),
        np.asarray([1.0, np.nan, 9.0]),
        "mean_reward",
    )

    assert comparison["marginal_mean_delta_second_minus_first"] == pytest.approx(4.0)
    assert comparison["mean_delta_second_minus_first"] == pytest.approx(1.0)
    assert comparison["paired_count"] == 1


def test_validation_cells_use_original_config_and_explicit_dr(tmp_path):
    checkpoint = tmp_path / "model_1000.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        checkpoint=checkpoint,
        num_envs=100,
        duration=10.0,
        seed=7,
        mujoco_njmax=256,
        friction_range=(0.5, 1.0),
    )
    artifact = tmp_path / "cell.npz"

    off = cell_command(args, BACKENDS[0], "off", artifact)
    on = cell_command(args, BACKENDS[1], "on", artifact)

    assert "--original_cfg" in off
    assert off[off.index("--mujoco_njmax") + 1] == "256"
    assert off[off.index("--contact-friction-dr") + 1] == "off"
    assert "--contact_friction_grid" not in off
    assert on[on.index("--contact-friction-dr") + 1] == "on"
    grid = on.index("--contact_friction_grid")
    assert on[grid + 1 : grid + 3] == ["0.5", "1.0"]
