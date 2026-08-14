import time
import sys
import struct

from deploy_config import DeployConfig
from go2_deploy.state import State

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.core import channel


class UnitreeRemoteController:
    def __init__(self):
        # key
        self.Lx = 0
        self.Rx = 0
        self.Ry = 0
        self.Ly = 0

        # button
        self.L1 = [0, 0]
        self.L2 = [0, 0]
        self.R1 = [0, 0]
        self.R2 = [0, 0]
        self.A = [0, 0]
        self.B = [0, 0]
        self.X = [0, 0]
        self.Y = [0, 0]
        self.Up = [0, 0]
        self.Down = [0, 0]
        self.Left = [0, 0]
        self.Right = [0, 0]
        self.Select = [0, 0]
        self.F1 = [0, 0]
        self.F3 = [0, 0]
        self.Start = [0, 0]

        self.cfg = DeployConfig()

        # derived values
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.yaw_vel = 0.0

    def parse_button(self, data1, data2):
        self.R1[1] = self.R1[0]
        self.L1[1] = self.L1[0]
        self.Start[1] = self.Start[0]
        self.Select[1] = self.Select[0]
        self.R2[1] = self.R2[0]
        self.L2[1] = self.L2[0]
        self.F1[1] = self.F1[0]
        self.F3[1] = self.F3[0]
        self.A[1] = self.A[0]
        self.B[1] = self.B[0]
        self.X[1] = self.X[0]
        self.Y[1] = self.Y[0]
        self.Up[1] = self.Up[0]
        self.Right[1] = self.Right[0]
        self.Down[1] = self.Down[0]
        self.Left[1] = self.Left[0]

        self.R1[0] = (data1 >> 0) & 1
        self.L1[0] = (data1 >> 1) & 1
        self.Start[0] = (data1 >> 2) & 1
        self.Select[0] = (data1 >> 3) & 1
        self.R2[0] = (data1 >> 4) & 1
        self.L2[0] = (data1 >> 5) & 1
        self.F1[0] = (data1 >> 6) & 1
        self.F3[0] = (data1 >> 7) & 1
        self.A[0] = (data2 >> 0) & 1
        self.B[0] = (data2 >> 1) & 1
        self.X[0] = (data2 >> 2) & 1
        self.Y[0] = (data2 >> 3) & 1
        self.Up[0] = (data2 >> 4) & 1
        self.Right[0] = (data2 >> 5) & 1
        self.Down[0] = (data2 >> 6) & 1
        self.Left[0] = (data2 >> 7) & 1

    def parse_key(self, data):
        lx_offset = 4
        self.Lx = struct.unpack("<f", data[lx_offset : lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack("<f", data[rx_offset : rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack("<f", data[ry_offset : ry_offset + 4])[0]
        L2_offset = 16
        L2 = struct.unpack("<f", data[L2_offset : L2_offset + 4])[
            0
        ]  # Placeholder，unused
        L2 = L2
        ly_offset = 20
        self.Ly = struct.unpack("<f", data[ly_offset : ly_offset + 4])[0]

        # y direction on the joystick is the x direction of motion for the robot
        self.lin_vel_x = (
            self.Ly * self.cfg.command_limits["lin_vel_x"]
        )  # forward-backward
        self.lin_vel_y = self.Lx * self.cfg.command_limits["lin_vel_y"]  # left-right
        self.yaw_vel = self.Rx * self.cfg.command_limits["yaw_vel"]

    def parse(self, remoteData):
        self.parse_key(remoteData)
        self.parse_button(remoteData[2], remoteData[3])

        # print("debug unitreeRemoteController: ")
        # print("Lx:", self.Lx)
        # print("Rx:", self.Rx)
        # print("Ry:", self.Ry)
        # print("Ly:", self.Ly)
        # print("lin_vel_x:", self.lin_vel_x)
        # print("lin_vel_y:", self.lin_vel_y)
        # print("yaw_vel:", self.yaw_vel)

        # print("L1:", self.L1)
        # print("L2:", self.L2)
        # print("R1:", self.R1)
        # print("R2:", self.R2)
        # print("A:", self.A)
        # print("B:", self.B)
        # print("X:", self.X)
        # print("Y:", self.Y)
        # print("Up:", self.Up)
        # print("Down:", self.Down)
        # print("Left:", self.Left)
        # print("Right:", self.Right)
        # print("Select:", self.Select)
        # print("F1:", self.F1)
        # print("F3:", self.F3)
        # print("Start:", self.Start)
        # print("\n")


class RCHandler:
    def __init__(self, controller):
        self.controller = controller
        self.rc = self.controller.remote_controller
        # whether to switch states upon pressing A, B, X, or Y
        # Can disable this in recovery mode to use
        # Unitree's default button combinations
        self.ABXY_on = True

    def _process_input(self):
        if self.ABXY_on:
            if self.rc.X == [1, 0]:
                self.controller.request_emergency_stop()
                return
            # if self.rc.Y == [1, 0]:
            #     self.controller.request_recovery()
            elif self.rc.B == [1, 0]:
                self.controller.request_intermediate()
            elif self.rc.A == [1, 0]:
                self.controller.request_custom_ctrl()

        if self.rc.Up == [1, 0]:
            self.controller.kp_mult += 0.1
            print(
                f"kp increased to {self.controller.kp_mult * self.controller.cfg.kp:.4}"
            )
        if self.rc.Down == [1, 0]:
            if self.controller.kp_mult >= 0.1:
                self.controller.kp_mult -= 0.1
            print(
                f"kp decreased to {self.controller.kp_mult * self.controller.cfg.kp:.4}"
            )
        if self.rc.Right == [1, 0]:
            self.controller.kd_mult += 0.1
            print(
                f"kd increased to {self.controller.kd_mult * self.controller.cfg.kd:.4}"
            )
        if self.rc.Left == [1, 0]:
            if self.controller.kd_mult >= 0.1:
                self.controller.kd_mult -= 0.1
            print(
                f"kd decreased to {self.controller.kd_mult * self.controller.cfg.kd:.4}"
            )

        if self.rc.F1 == [1, 0]:
            if self.controller._state != State.RECOVERY and self.ABXY_on:
                print("ABXY can only be turned off in recovery mode")

            self.ABXY_on = not self.ABXY_on
            ABXY_on_text = "on" if self.ABXY_on else "off"
            print("ABXY buttons have been turned " + ABXY_on_text)

        if self.controller._state != State.RECOVERY and not self.ABXY_on:
            self.ABXY_on = True
            print("ABXY buttons have been turned on")


class Custom:
    def __init__(self):
        self.low_state = None
        self.remoteController = UnitreeRemoteController()

    def Init(self):
        self.lowstate_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        wireless_remote_data = self.low_state.wireless_remote
        self.remoteController.parse(wireless_remote_data)


if __name__ == "__main__":
    print(
        "WARNING: Please ensure there are no obstacles around the robot"
        + "while running this example."
    )
    input("Press Enter to continue...")

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    while True:
        time.sleep(1)
