from types import SimpleNamespace

import torch

from scripts.eval_domain_randomization_trajectory import (
    BACKENDS,
    cell_command,
    system_cog,
)


def test_system_cog_includes_rotated_local_com_velocity():
    body_states = torch.zeros(1, 2, 13)
    body_states[..., 6] = 1.0
    body_states[0, 0, 0:3] = torch.tensor([0.0, 0.0, 1.0])
    body_states[0, 1, 0:3] = torch.tensor([2.0, 0.0, 1.0])
    body_states[0, 0, 10:13] = torch.tensor([0.0, 0.0, 2.0])
    body_mass = torch.tensor([1.0, 3.0])
    local_com = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    position, velocity = system_cog(body_states, body_mass, local_com)

    torch.testing.assert_close(position, torch.tensor([[1.75, 0.0, 1.0]]))
    torch.testing.assert_close(velocity, torch.tensor([[0.0, 0.5, 0.0]]))


def test_campaign_cells_fix_command_friction_and_inference_protocol(tmp_path):
    checkpoint = tmp_path / "model_1000.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        checkpoint=checkpoint,
        output=tmp_path,
        duration=5.0,
        seed=7,
        dr_friction=0.75,
        command=(1.0, 0.0, 0.0),
        mujoco_njmax=256,
    )

    off = cell_command(args, BACKENDS[0], "off")
    on = cell_command(args, BACKENDS[2], "on")

    assert off[off.index("--command") + 1 : off.index("--command") + 4] == [
        "1.0",
        "0.0",
        "0.0",
    ]
    assert on[on.index("--dr-friction") + 1] == "0.75"
    assert on[on.index("--dr-mode") + 1] == "on"
    assert on[on.index("--cell") + 1] == "vsim"
    assert "--resume" not in on
