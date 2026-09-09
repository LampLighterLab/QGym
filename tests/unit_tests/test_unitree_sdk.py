"""Opt-in SDK/message/CRC smoke tests; no DDS participant or robot connection."""

import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unitree


@pytest.fixture(scope="module")
def sdk():
    if sys.platform != "linux":
        pytest.skip("The Unitree deployment extra supports Linux")
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ModuleNotFoundError as error:
        if error.name == "unitree_sdk2py" or error.name == "cyclonedds":
            pytest.fail(
                "Unitree tests requested but the SDK extra is missing; install "
                "it with `uv sync --frozen --extra unitree_sdk`.",
                pytrace=False,
            )
        raise

    from go2_deploy.deploy_config import DeployConfig
    from go2_deploy.utility import deploy_utility

    return SimpleNamespace(
        clients=(
            ChannelPublisher,
            ChannelSubscriber,
            SportClient,
            MotionSwitcherClient,
        ),
        lowcmd_type=LowCmd_,
        lowstate_type=LowState_,
        crc=CRC(),
        config=DeployConfig(),
        utility=deploy_utility,
    )


def test_sdk_imports_required_channel_and_message_types(sdk):
    assert all(callable(client) for client in sdk.clients)
    assert sdk.lowcmd_type.__idl_typename__ == "unitree_go.msg.dds_.LowCmd_"
    assert sdk.lowstate_type.__idl_typename__ == "unitree_go.msg.dds_.LowState_"


def test_default_lowcmd_has_valid_crc_and_configured_gains(sdk):
    command = sdk.utility.default_lowcmd()
    assert isinstance(command, sdk.lowcmd_type)
    assert command.head == [0xFE, 0xEF]
    assert command.level_flag == 0xFF
    assert command.gpio == 0
    assert len(command.motor_cmd) == 20
    for index, motor in enumerate(command.motor_cmd):
        assert motor.mode == 0x01
        assert (motor.q, motor.dq, motor.tau) == (0.0, 0.0, 0.0)
        assert motor.kp == (sdk.config.kp if index < 12 else 0.0)
        assert motor.kd == (sdk.config.kd if index < 12 else 0.0)
    assert command.crc == sdk.crc.Crc(command)
    assert 0 <= command.crc <= 0xFFFFFFFF


def test_emergency_lowcmd_has_damping_only_and_fresh_crc(sdk):
    original = sdk.utility.default_lowcmd()
    original_crc = original.crc
    emergency = sdk.utility.emergency_lowcmd()
    assert emergency is not original
    assert isinstance(emergency, sdk.lowcmd_type)
    for index, motor in enumerate(emergency.motor_cmd):
        assert (motor.q, motor.dq, motor.tau, motor.kp) == (0.0, 0.0, 0.0, 0.0)
        assert motor.kd == (5.0 if index < 12 else 0.0)
    assert emergency.crc == sdk.crc.Crc(emergency)
    assert emergency.crc != original_crc
    assert original.crc == original_crc == sdk.crc.Crc(original)
    assert all(motor.kp == sdk.config.kp for motor in original.motor_cmd[:12])


def test_crc_detects_a_changed_motor_command(sdk):
    command = sdk.utility.default_lowcmd()
    original_crc = command.crc
    command.motor_cmd[0].q = 0.125

    changed_crc = sdk.crc.Crc(command)

    assert changed_crc != original_crc
    command.crc = changed_crc
    # The CRC field itself must be excluded from the checksum calculation.
    assert sdk.crc.Crc(command) == changed_crc
