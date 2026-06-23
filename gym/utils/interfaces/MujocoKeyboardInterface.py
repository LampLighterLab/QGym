"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press maps straight to one command increment (no edge detection needed —
contrast VsimKeyboardInterface, which polls).

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letter keys, so the shared letter table gives
# the keycodes directly.  Hardcoded to avoid a glfw dependency.
KEYCODE_TO_ACTION = {ord(key): action for key, action in KEY_TO_ACTION.items()}


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)

        env._backend._viewer_key_callback = self._on_key
        self.commands.print_help("MuJoCo viewer")

    def _on_key(self, keycode: int) -> None:
        c = self.env.commands
        if keycode == KEY_UP:
            c[:, 0] = torch.clamp(c[:, 0] + self.increment_x, max=self.max_vel_forward)
        elif keycode == KEY_DOWN:
            c[:, 0] = torch.clamp(c[:, 0] - self.increment_x, min=self.max_vel_backward)
        elif keycode == KEY_COMMA:
            c[:, 1] = torch.clamp(c[:, 1] + self.increment_y, max=self.max_vel_sideways)
        elif keycode == KEY_PERIOD:
            c[:, 1] = torch.clamp(
                c[:, 1] - self.increment_y, min=-self.max_vel_sideways
            )
        elif keycode == KEY_LEFT:
            c[:, 2] = torch.clamp(c[:, 2] + self.increment_yaw, max=self.max_vel_yaw)
        elif keycode == KEY_RIGHT:
            c[:, 2] = torch.clamp(c[:, 2] - self.increment_yaw, min=-self.max_vel_yaw)
        elif keycode == KEY_R:
            self.env.timed_out[:] = True
            self.env.reset()
