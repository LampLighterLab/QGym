from types import SimpleNamespace

import pytest

from scripts.benchmark_vsim_environment_sets import (
    NominalSetGym,
    configure_task,
    environment_set_sizes,
    get_args,
)


@pytest.mark.parametrize("num_sets", [1, 64, 4096])
def test_set_partition_preserves_environment_count(num_sets):
    sizes = environment_set_sizes(4096, num_sets)

    assert len(sizes) == num_sets
    assert sum(sizes) == 4096
    assert set(sizes) == {4096 // num_sets}


@pytest.mark.parametrize(
    "num_envs,num_sets", [(0, 1), (-4, 2), (8, 0), (8, -2), (8, 3)]
)
def test_set_partition_rejects_invalid_counts(num_envs, num_sets):
    with pytest.raises(ValueError, match="positive divisor"):
        environment_set_sizes(num_envs, num_sets)


@pytest.mark.parametrize("graphs", [False, True])
def test_topology_config_keeps_nominal_physics_and_fixed_frequency(graphs):
    cfg = configure_task(16, seed=17, graph_captures=graphs)

    assert cfg.env.num_envs == 16
    assert cfg.seed == 17
    assert cfg.control.ctrl_frequency == 100
    assert cfg.control.desired_sim_frequency == 100
    assert cfg.control.decimation == 1
    assert cfg.control.ctrl_dt == pytest.approx(0.01)
    assert cfg.sim_dt == pytest.approx(0.01)
    assert cfg.domain_randomization.startup.contact_friction_range is None
    assert cfg.domain_randomization.startup.link_mass_scale_range is None
    assert cfg.domain_randomization.episode.scale_ranges == {}
    assert cfg.terrain.static_friction == cfg.terrain.dynamic_friction == 1.0
    assert not cfg.push_robots.toggle
    assert cfg.vsim_attributes.enable_graph_captures is graphs


def test_proxy_changes_group_sharing_and_forwards_other_operations():
    calls = []
    group = object()
    native = SimpleNamespace(
        create_environment_group=lambda definition, sizes: (
            calls.append((definition, sizes)) or group
        ),
        get_num_solver_iterations=lambda: 4,
    )
    proxy = NominalSetGym(native, [2, 2])

    assert proxy.create_environment_group("robot", [4]) is group
    assert calls == [("robot", [2, 2])]
    assert proxy.get_num_solver_iterations() == 4
    with pytest.raises(ValueError, match="one nominal"):
        proxy.create_environment_group("robot", [1, 1, 1, 1])
    assert len(calls) == 1


@pytest.mark.parametrize("option", ["--steps", "--repeats"])
def test_cli_rejects_empty_timing_trials(option, tmp_path):
    with pytest.raises(SystemExit):
        get_args(["--sets", "1", "--output", str(tmp_path / "cell.json"), option, "0"])
