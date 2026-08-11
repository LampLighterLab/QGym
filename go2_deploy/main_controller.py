import time
import threading
import math

from go2_deploy.state import State
from rl_controller import RLController
from unitree_remote_controller import UnitreeRemoteController
from deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility
from go2_deploy.utility.csv_logger import CSVLogger
from go2_deploy.utility.thread import RecurrentThread

import torch
from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    SportModeState_,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.utils.crc import CRC


class MainController:
    def __init__(self):
        self._state = State.RECOVERY
        self._estop_flag = False
        self.rl_controller = RLController()
        self.remote_controller = UnitreeRemoteController()
        self.cfg = DeployConfig()
        self.csv_logger = CSVLogger(self)

        self.obs_vec_size = 0
        for obs in self.cfg.obs_vector:
            self.obs_vec_size += self.cfg.obs_sizes[obs]

        self._init_buffers()

        if self.cfg.task_name == "go2trot":
            self._init_go2trot_buffers()

        self._init_unitree_clients()

        self.default_lowcmd = deploy_utility.default_lowcmd()
        self.emergency_lowcmd = deploy_utility.emergency_lowcmd()
        self.kp_mult = 1.0

        # Threads
        self._create_lowcmd_thread()
        self.emergency_lowcmd_thread = None
        self.watchdog_thread = RecurrentThread(
            interval=0.01, target=self._watchdog_loop
        )
        self.watchdog_thread.Start()

        self.switch_to_recovery()
        # At this point, buffers initialized, self._state = RECOVERY

    # Init --------------------------------------------------

    def _init_buffers(self):
        # SportModeState_: last recieved sportmode msg
        self.last_sportmodestate_msg = None
        self.last_sportmodestate_msg_lock = threading.Lock()

        # LowState_: last recieved lowstate msg
        self.last_lowstate_msg = None
        self.last_lowstate_msg_lock = threading.Lock()

        # torch tensor: obs vector computed from last_lowstate_msg
        self.last_obs = torch.zeros(self.obs_vec_size)
        self.last_obs_lock = threading.Lock()

        # torch.tensor(12): Target joint positions in radians
        # (difference from default_pos + gait_trajectory)
        self.last_action = torch.zeros(12)
        self.last_action_lock = threading.Lock()

        # torch.tensor(3): [x_vel, y_vel, yaw_vel]
        self.last_command = torch.zeros(3)
        self.last_command_lock = threading.Lock()

        # Log obs freq / control freq every 5 seconds
        self.last_terminal_output_time = time.monotonic()
        self.lowstate_obs_count = 0
        self.sportmodestate_obs_count = 0
        self.action_count = 0

    def _init_go2trot_buffers(self):
        self.phase_frequency = self.cfg.phase_frequency
        self.phase = 0.0
        self.gait_reference = torch.zeros(12)
        self._gait_phase_offsets = torch.tensor([0, math.pi, math.pi, 0])
        self._gait_dof_phase_offsets = self._gait_phase_offsets.repeat_interleave(3)
        self._gait_joint_offsets = torch.tensor(4 * [0.0, 0.96, -1.36])
        self._gait_joint_amplitudes = torch.tensor(4 * [0.0, -0.15, 0.30])

    # Unitree SDK pub/sub clients
    def _init_unitree_clients(self):
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate_msg, 10)
        self.sportmodestate_subscriber = ChannelSubscriber(
            "rt/sportmodestate", SportModeState_
        )
        self.sportmodestate_subscriber.Init(self._on_sportmodestate_msg, 10)
        self.motion_switcher_client = MotionSwitcherClient()
        self.motion_switcher_client.SetTimeout(10.0)
        self.motion_switcher_client.Init()
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.crc = CRC()

    # Read and write messages ---------------------------------------------

    # torch.tensor(12) -> LowCmd_
    def action_to_lowcmd(self, action):
        return deploy_utility.action_to_lowcmd(self, action, kp_mult=self.kp_mult)

    # LowState_ -> torch tensor
    def msg_to_obs(self, lowstate_msg):
        return deploy_utility.lowstate_to_obs(self, lowstate_msg)

    # Save the last lowstate msg, update velocity command
    def _on_lowstate_msg(self, msg):
        t = time.monotonic()
        with self.last_lowstate_msg_lock:
            self.last_lowstate_msg = msg
            self.remote_controller.parse(msg.wireless_remote)
        with self.last_command_lock:
            self.last_command = torch.tensor(
                [
                    self.remote_controller.lin_vel_x,
                    self.remote_controller.lin_vel_y,
                    self.remote_controller.yaw_vel,
                ]
            )
        self.csv_logger.log_lowstate(t, msg, self.remote_controller)
        self.lowstate_obs_count += 1

    # Save the last sportmodestate_msg
    def _on_sportmodestate_msg(self, msg):
        t = time.monotonic()
        with self.last_sportmodestate_msg_lock:
            self.last_sportmodestate_msg = msg
        self.csv_logger.log_sportmodestate(t, msg)
        self.sportmodestate_obs_count += 1

    # Main control loop ------------------------------------------

    # Update self.phase, self._gait_reference
    def _process_go2trot_buffers(self, t):
        self.phase = 2 * math.pi * self.phase_frequency * t % (2 * math.pi)
        joint_phase = self.phase + self._gait_dof_phase_offsets
        self.gait_reference = (
            self._gait_joint_offsets
            + self._gait_joint_amplitudes * torch.sin(joint_phase)
        )

    # Read last lowstate msg, publish LowCmd_
    def _control_loop(self):
        t = time.monotonic()
        if self.cfg.task_name == "go2trot":
            self._process_go2trot_buffers(t)

        with (
            self.last_lowstate_msg_lock,
            self.last_sportmodestate_msg_lock,
            self.last_obs_lock,
        ):
            self.last_obs = self.msg_to_obs(self.last_lowstate_msg)
            last_obs = self.last_obs.clone().detach()
        action = self._act(last_obs)

        with self.last_action_lock:
            self.last_action = action
            lowcmd = self.action_to_lowcmd(self.last_action)

        lowcmd.crc = self.crc.Crc(lowcmd)
        self.lowcmd_publisher.Write(lowcmd)

        self.csv_logger.log_control(t, self.last_obs, action, lowcmd)
        self.action_count += 1

    # Return torch tensor: target joint position (minus default pos, gait traj)
    def _act(self, last_obs):
        if self._state == State.CUSTOM_CTRL:
            return self.rl_controller.act(last_obs)
        elif self._state == State.INTERMEDIATE:
            return self.default_pos
        else:
            print("Should not be using RL controller! This should not happen")
            self._estop_flag = True
            return self.last_action

    # Emergency stop ----------------------------------

    def emergency_stop(self):
        if self._state == State.EMERGENCY_STOP:
            self._estop_flag = False
            return

        if self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Wait()

        self._estop_flag = False
        self._state = State.EMERGENCY_STOP

        self.emergency_lowcmd_thread = RecurrentThread(
            interval=0.01, target=self._emergency_control_loop
        )
        self.motion_switcher_client.ReleaseMode()
        self.emergency_lowcmd_thread.Start()
        print("Emergency stop activated! Recovering in 10 s")
        time.sleep(10)
        self.emergency_lowcmd_thread.Wait()
        self.emergency_lowcmd_thread = None
        self._create_lowcmd_thread()

        self.switch_to_recovery()

    # Can implement checking for unsafe conditions
    def _watchdog_loop(self):
        if self._estop_flag:
            self.emergency_stop()
        t = time.monotonic()
        if t > self.last_terminal_output_time + 5.0:
            self._output_terminal_info(t)

    # publish LowCmd_ damping messages
    def _emergency_control_loop(self):
        self.lowcmd_publisher.Write(self.emergency_lowcmd)

    # Switch between states -------------------------------------

    def switch_to_recovery(self):
        print("Switching to recovery state")

        if self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Wait()
            self._create_lowcmd_thread()

        self._state = State.RECOVERY

        self.motion_switcher_client.SelectMode("mcf")
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "mcf":
            print("Failed to switch to sport mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.SelectMode("mcf")
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        self.sport_client.RecoveryStand()
        time.sleep(10)
        print("Recovered")

    def switch_to_intermediate(self):
        if self._state != State.RECOVERY:
            print("Must be in recovery state to switch to intermediate")
            return

        self.default_pos = torch.tensor(
            deploy_utility._get_obs_dof_pos_obs(self, self.last_lowstate_msg)
        )
        print(self.default_pos)
        self._state = State.INTERMEDIATE
        print("Switching to intermediate")
        self.motion_switcher_client.ReleaseMode()
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "":
            print("Failed to switch to low state mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.ReleaseMode()
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        self.lowcmd_thread.Start()

    def switch_to_custom_controller(self):
        if self._state != State.INTERMEDIATE:
            print("Must be in intermediate state to switch to a custom controller!")
            return

        print("Switching to custom controller")
        self._state = State.CUSTOM_CTRL
        self.motion_switcher_client.ReleaseMode()
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "":
            print("Failed to switch to low state mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.ReleaseMode()
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        if not self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Start()

    def switch_to_default_controller(self):
        if self._state != State.RECOVERY:
            print("Must be in recovery state to switch to the default controller!")
            return

        print("Switching to default controller")
        self._state = State.DEFAULT_CTRL
        self.motion_switcher_client.SelectMode("mcf")
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "mcf":
            print("Failed to switch to sport mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.SelectMode("mcf")
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

    # Utility -----------------------------------------------------

    def _create_lowcmd_thread(self):
        self.lowcmd_thread = RecurrentThread(
            interval=1 / self.cfg.ctrl_freq, target=self._control_loop
        )

    def _output_terminal_info(self, current_time):
        print(
            "\nlowstate obs freq = "
            + f"{self.lowstate_obs_count / (current_time - self.last_terminal_output_time):.5} Hz, "  # noqa: E501
            + "sportmodestate obs freq = "
            + f"{self.sportmodestate_obs_count / (current_time - self.last_terminal_output_time):.5} Hz, "  # noqa: E501
            + "custom policy control freq = "
            + f"{self.action_count / (current_time - self.last_terminal_output_time):.5} Hz"  # noqa: E501
        )
        print("Current mode: " + self._state.name)
        print(
            f"Logging to {self.csv_logger.run_dir} "
            + f"({self.csv_logger.dropped_rows} rows dropped)"
        )
        print("\n<enter>: emergency stop")
        print("r: recovery (return to standing position)")
        print("q: intermediate")
        print("c: custom RL policy")
        print(
            "d: default controller (high level control with the wireless controller)\n"
        )
        print("i: increase kp by 10%")
        print("k: decrease kp by 10%")
        self.action_count = 0
        self.lowstate_obs_count = 0
        self.sportmodestate_obs_count = 0
        self.last_terminal_output_time = current_time
