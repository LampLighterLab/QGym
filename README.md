# Q2

Q2 is a robotics and reinforcement-learning framework descended from
[legged_gym](https://github.com/leggedrobotics/legged_gym). It supports MuJoCo CPU,
MuJoCo Warp, and optional licensed VSim physics backends.

Follow [the setup and training guide](README_MUJOCO.md) for installation,
backend selection, and the supported command-line workflows. From a checkout
with Python 3.11 and uv installed, the default setup and test commands are:

```bash
uv sync --frozen
uv run --frozen python -m pytest -q
```

Go2 hardware deployment uses the optional `unitree_sdk` extra. See the
[deployment guide](README_DEPLOY.md) for the SDK and native dependencies.

Historical project notes are available in the
[QGym wiki](https://github.com/sheim/QGym/wiki).
