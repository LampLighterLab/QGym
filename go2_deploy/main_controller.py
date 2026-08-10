from enum import Enum, auto
import time
import threading
import math

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


class State(Enum):
    # Robot is in sport mode ("mcf" mode), lowcmd_thread is not started
    # Can only be accessed from RECOVERY
    DEFAULT_CTRL = auto()

    # Robot is in low state mode, lowcmd_thread is running
    # Can only be accessed from RECOVERY
    CUSTOM_CTRL = auto()

    # After emergency_stop() finishes processing, the robot is in low state mode
    # a new lowcmd_thread is created + not started, self.emergency_lowcmd_thread = None
    # Can be accessed from any other state, triggered automatically in
    # dangerous conditions
    EMERGENCY_STOP = auto()

    # After recovering, the robot is in sport mode, is standing up,
    # lowcmd_thread is not started
    # Can be accessed from any other state
    RECOVERY = auto()

    INTERMEDIATE = auto()


class MainController:
    def __init__(self):
        self._state = State.RECOVERY
        self.rl_controller = RLController()
        self.remote_controller = UnitreeRemoteController()
        self.cfg = DeployConfig()

        # CSV logging (constructed before the subscribers so no callback can
        # fire against a missing logger)
        self.csv_logger = CSVLogger(self)

        self.obs_vec_size = 0
        for obs in self.cfg.obs_vector:
            self.obs_vec_size += self.cfg.obs_sizes[obs]

        # Last sportmodestate msg
        self.last_sportmodestate_msg = None
        self.last_sportmodestate_msg_lock = threading.Lock()

        # Last lowstate msg
        self.last_lowstate_msg = None
        self.last_lowstate_msg_lock = threading.Lock()

        # Last computed observation vector
        self.last_obs = torch.zeros(self.obs_vec_size)
        self.last_obs_lock = threading.Lock()

        # Last action (target joint positions)
        self.last_action = torch.zeros(12)
        self.last_action_lock = threading.Lock()

        # Last commanded velocity (x, y, yaw)
        self.last_command = torch.zeros(3)
        self.last_command_lock = threading.Lock()

        # Log obs freq / control freq every 5 seconds
        self.last_terminal_output_time = time.monotonic()
        self.lowstate_obs_count = 0
        self.sportmodestate_obs_count = 0
        self.action_count = 0

        if self.cfg.task_name == "go2trot":
            self.phase_frequency = self.cfg.phase_frequency
            self.phase = 0.0

        # Unitree SDK pub/sub
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

        # Unitree messaging utilities
        self.crc = CRC()
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

    # Process incoming states and outgoing commands
    def action_to_lowcmd(self, action):
        return deploy_utility.action_to_lowcmd(self, action, kp_mult=self.kp_mult)

    def msg_to_obs(self, lowstate_msg):
        return deploy_utility.lowstate_to_obs(self, lowstate_msg)

    # Save last lowstate msg
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

    # Save last sportmodestate_msg
    def _on_sportmodestate_msg(self, msg):
        t = time.monotonic()
        with self.last_sportmodestate_msg_lock:
            self.last_sportmodestate_msg = msg
        self.csv_logger.log_sportmodestate(t, msg)
        self.sportmodestate_obs_count += 1

    # Process most recent msgs and publish LowCmd_ msg
    def _control_loop(self):
        t = time.monotonic()
        if self.cfg.task_name == "go2trot":
            self.phase = t % (1.0 / self.phase_frequency) * 2 * math.pi

        with (
            self.last_lowstate_msg_lock,
            self.last_sportmodestate_msg_lock,
            self.last_obs_lock,
        ):
            self.last_obs = self.msg_to_obs(self.last_lowstate_msg)
            action = self._act()
        lowcmd = self.action_to_lowcmd(self.last_action)
        with self.last_action_lock:
            self.last_action = action

        lowcmd.crc = self.crc.Crc(lowcmd)
        self.lowcmd_publisher.Write(lowcmd)

        # lowcmd carries the previous tick's action, action is this tick's fresh
        # policy output -- both are logged rather than conflated
        self.csv_logger.log_control(t, self.last_obs, action, lowcmd)

        self.action_count += 1

    def _act(self):
        # act using lowcmd thread
        if self._state == State.CUSTOM_CTRL:
            actor_output = self.rl_controller.act(self.last_obs)
            return torch.clip(
                actor_output,
                min=self.cfg.lower_joint_limit,
                max=self.cfg.upper_joint_limit,
            )
        elif self._state == State.INTERMEDIATE:
            return self.default_pos
        else:
            print("Should not be using RL controller! This should not happen")
            self.emergency_stop()
            return None

    # Safety features

    def emergency_stop(self):
        if self._state == State.EMERGENCY_STOP:
            return

        if self.lowcmd_thread.IsAlive():
            while not self.lowcmd_thread.Wait():
                print(
                    "\nWARNING: RL control loop did not stop during emergency stop, "
                    + "press L2+B to manually stop. Trying again in 2s\n"
                )
                time.sleep(2)

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

    # Check that threads are alive, unsafe conditions
    def _watchdog_loop(self):
        t = time.monotonic()
        if t > self.last_terminal_output_time + 5.0:
            self._output_terminal_info(t)

    # publish LowCmd_ damping messages
    def _emergency_control_loop(self):
        self.lowcmd_publisher.Write(self.emergency_lowcmd)

    # Switch between states

    def switch_to_recovery(self):
        print("Switching to recovery state")

        if self.lowcmd_thread.IsAlive():
            while not self.lowcmd_thread.Wait():
                print(
                    "\nWARNING: RL control loop did not stop during recovery, "
                    + "press L2+B to manually stop. Trying again in 1s\n"
                )
                time.sleep(1)
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

    # Utility

    def _create_lowcmd_thread(self):
        self.lowcmd_thread = RecurrentThread(
            interval=1 / self.cfg.ctrl_freq, target=self._control_loop
        )

    def _output_terminal_info(self, current_time):
        print(
            "\nlowstate obs freq = "
            + f"{self.lowstate_obs_count / (current_time - self.last_terminal_output_time):.5}"  # noqa: E501
            + " Hz, "
            + "sportmodestate obs freq = "
            + f"{self.sportmodestate_obs_count / (current_time - self.last_terminal_output_time):.5}"  # noqa: E501
            + " Hz, "
            + "custom policy control freq = "
            + f"{self.action_count / (current_time - self.last_terminal_output_time):.5}"  # noqa: E501
            + " Hz"
        )
        print("Current mode: " + self._state.name)
        print(
            f"Logging to {self.csv_logger.run_dir} "
            + f"({self.csv_logger.dropped_rows} rows dropped)"
        )
        print("\nTo switch modes: enter in terminal <key> and press enter")
        print(
            "(enter key pressed without input): emergency stop "
            + "(or press L2+B on wireless controller)"
        )
        print("r: recovery (return to standing position)")
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
