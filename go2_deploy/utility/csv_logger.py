import csv
import json
import os
import queue
import time
from datetime import datetime

from go2_deploy.utility.thread import RecurrentThread

# Buttons parsed by UnitreeRemoteController, logged as one column each
REMOTE_BUTTONS = [
    "L1",
    "L2",
    "R1",
    "R2",
    "A",
    "B",
    "X",
    "Y",
    "Up",
    "Down",
    "Left",
    "Right",
    "Select",
    "Start",
    "F1",
    "F3",
]

# Rows are dropped rather than blocking a real-time thread. 20k rows is ~30 s of
# lowstate at 500 Hz, far more headroom than the writer thread should ever need.
QUEUE_MAXSIZE = 20000

FLUSH_INTERVAL = 1.0


def _numbered(prefix, count):
    return [f"{prefix}_{i}" for i in range(count)]


class _Stream:
    """One open CSV file plus its writer."""

    def __init__(self, path, header):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.width = len(header)


class CSVLogger:
    """Streams Unitree SDK messages, observations, and policy actions to CSV.

    Producers (the DDS callbacks and the control loop) only extract plain floats
    and hand a row to a queue; a single background thread does all file I/O.
    """

    def __init__(self, main_controller, log_root="logs/deploy"):
        self.main_controller = main_controller
        cfg = main_controller.cfg

        start_wall = datetime.now()
        self.start_monotonic = time.monotonic()
        self.run_dir = os.path.join(log_root, start_wall.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.run_dir, exist_ok=True)

        # Observation columns follow cfg.obs_vector so the header tracks any
        # config change instead of going stale.
        self.obs_columns = []
        for obs in cfg.obs_vector:
            self.obs_columns += _numbered("obs_" + obs, cfg.obs_sizes[obs])

        self.streams = {
            "lowstate": _Stream(
                os.path.join(self.run_dir, "lowstate.csv"), self._lowstate_header()
            ),
            "sportmodestate": _Stream(
                os.path.join(self.run_dir, "sportmodestate.csv"),
                self._sportmodestate_header(),
            ),
            "control": _Stream(
                os.path.join(self.run_dir, "control.csv"), self._control_header()
            ),
        }

        # Monotonic timestamps are only meaningful next to their wall-clock origin
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "start_wall_clock": start_wall.isoformat(),
                    "start_monotonic": self.start_monotonic,
                    "task_name": cfg.task_name,
                    "ctrl_freq": cfg.ctrl_freq,
                    "kp": cfg.kp,
                    "kd": cfg.kd,
                    "obs_vector": list(cfg.obs_vector),
                },
                f,
                indent=2,
            )

        self.queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.dropped_rows = 0
        self._last_flush_time = time.monotonic()
        self._closed = False

        self.writer_thread = RecurrentThread(interval=0.0, target=self._writer_loop)
        self.writer_thread.Start()

    # Headers

    def _lowstate_header(self):
        header = [
            "t",
            "state",
            "tick",
            "crc",
            "imu_quat_w",
            "imu_quat_x",
            "imu_quat_y",
            "imu_quat_z",
            "imu_gyro_x",
            "imu_gyro_y",
            "imu_gyro_z",
            "imu_accel_x",
            "imu_accel_y",
            "imu_accel_z",
            "imu_rpy_r",
            "imu_rpy_p",
            "imu_rpy_y",
        ]
        for i in range(12):
            header += [
                f"motor{i}_q",
                f"motor{i}_dq",
                f"motor{i}_ddq",
                f"motor{i}_tau_est",
                f"motor{i}_temperature",
            ]
        header += _numbered("foot_force", 4)
        header += ["power_v", "power_a"]
        header += ["Lx", "Ly", "Rx", "Ry", "lin_vel_x", "lin_vel_y", "yaw_vel"]
        header += REMOTE_BUTTONS
        return header

    def _sportmodestate_header(self):
        return (
            [
                "t",
                "state",
                "stamp_sec",
                "stamp_nanosec",
                "error_code",
                "mode",
                "progress",
                "gait_type",
                "body_height",
                "foot_raise_height",
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "yaw_speed",
            ]
            + _numbered("foot_force", 4)
            + ["imu_rpy_r", "imu_rpy_p", "imu_rpy_y"]
            + _numbered("foot_position_body", 12)
            + _numbered("foot_speed_body", 12)
        )

    def _control_header(self):
        return (
            ["t", "state", "phase"]
            + self.obs_columns
            + _numbered("action", 12)
            + _numbered("lowcmd_q", 12)
            + ["kp", "kd", "kp_mult"]
        )

    # Producers -- called from the DDS callbacks and the control loop

    def log_lowstate(self, t, msg, remote_controller):
        imu = msg.imu_state
        row = [
            t,
            self.main_controller._state.name,
            msg.tick,
            msg.crc,
            *imu.quaternion,
            *imu.gyroscope,
            *imu.accelerometer,
            *imu.rpy,
        ]
        for i in range(12):
            motor = msg.motor_state[i]
            row += [motor.q, motor.dq, motor.ddq, motor.tau_est, motor.temperature]
        row += list(msg.foot_force)
        row += [msg.power_v, msg.power_a]
        row += [
            remote_controller.Lx,
            remote_controller.Ly,
            remote_controller.Rx,
            remote_controller.Ry,
            remote_controller.lin_vel_x,
            remote_controller.lin_vel_y,
            remote_controller.yaw_vel,
        ]
        row += [getattr(remote_controller, button) for button in REMOTE_BUTTONS]
        self._enqueue("lowstate", row)

    def log_sportmodestate(self, t, msg):
        row = [
            t,
            self.main_controller._state.name,
            msg.stamp.sec,
            msg.stamp.nanosec,
            msg.error_code,
            msg.mode,
            msg.progress,
            msg.gait_type,
            msg.body_height,
            msg.foot_raise_height,
            *msg.position,
            *msg.velocity,
            msg.yaw_speed,
            *msg.foot_force,
            *msg.imu_state.rpy,
            *msg.foot_position_body,
            *msg.foot_speed_body,
        ]
        self._enqueue("sportmodestate", row)

    def log_control(self, t, obs, action, lowcmd):
        cfg = self.main_controller.cfg
        row = [
            t,
            self.main_controller._state.name,
            getattr(self.main_controller, "phase", ""),
        ]
        row += obs.tolist() if obs is not None else [""] * len(self.obs_columns)
        # _act() returns None on the emergency path; leave those columns blank
        row += action.tolist() if action is not None else [""] * 12
        row += [lowcmd.motor_cmd[i].q for i in range(12)]
        row += [cfg.kp, cfg.kd, self.main_controller.kp_mult]
        self._enqueue("control", row)

    def _enqueue(self, stream_name, row):
        if self._closed:
            return
        try:
            self.queue.put_nowait((stream_name, row))
        except queue.Full:
            self.dropped_rows += 1

    # Consumer

    def _writer_loop(self):
        try:
            stream_name, row = self.queue.get(timeout=0.1)
        except queue.Empty:
            self._flush_if_due()
            return
        self.streams[stream_name].writer.writerow(row)
        self._flush_if_due()

    def _flush_if_due(self):
        t = time.monotonic()
        if t < self._last_flush_time + FLUSH_INTERVAL:
            return
        for stream in self.streams.values():
            stream.file.flush()
        self._last_flush_time = t

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.writer_thread.Wait(timeout=1.0)
        # Drain whatever the writer thread did not get to
        while True:
            try:
                stream_name, row = self.queue.get_nowait()
            except queue.Empty:
                break
            self.streams[stream_name].writer.writerow(row)
        for stream in self.streams.values():
            stream.file.flush()
            stream.file.close()
