# Running BigRedGym

## Setup

```sh
uv sync --frozen
```

Always pass `--frozen` to `uv run` (the optional VSim wheel / Unitree SDK are absent).

## Train

```sh
uv run --frozen scripts/train.py --task=go2trot --headless
```

Checkpoints land in `logs/<experiment_name>/<run>/model_<iter>.pt`.

## FLD models (legged-fld)

The VAE/PAE models live in [legged-fld](https://github.com/LampLighterLab/legged-fld),
pulled in as an optional, editable checkout (pinned in `gitman.yml`):

```sh
uvx gitman install            # clone legged-fld at the locked SHA into thirdparty/
uv sync --frozen --extra fld
uv run --frozen --extra fld python -c "import legged_fld; print(legged_fld.__file__)"
```

legged-fld is developed in its own standalone clone; treat
`thirdparty/legged-fld` as read-only. After pushing legged-fld, pull it in with
`uvx gitman update` (fetches the latest `rev`, rewrites `sources_locked` in
`gitman.yml` — commit that). `uvx gitman install` restores the locked SHA.
Both refuse if the checkout has local edits (`--force` discards them).

Without the checkout, plain `uv sync --frozen` still works (CI path); only
code importing `legged_fld` needs the extra.

## Play (with the MuJoCo viewer) — macOS

On macOS the MuJoCo passive viewer must run under `mjpython`, not `python`.
With uv-managed Python, a bare `.venv/bin/mjpython` fails:

```
failed to dlopen path '.../.venv/bin/python': Library not loaded: @executable_path/../lib/libpython3.11.dylib
```

Use the `mac_run` entry point (defined in `pyproject.toml` `[project.scripts]`,
implemented in `gym/utils/mac_run.py`); it forwards all args to `scripts/play.py`:

```sh
uv run --frozen mac_run --task=go2trot --num_envs 1
```

If `mac_run` isn't found, run `uv sync --frozen` to install the entry point.
On Linux it just runs `play.py` with plain python.

It's equivalent to pointing the dynamic loader at uv's Python `lib/` directory:

```sh
DYLD_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
  .venv/bin/mjpython scripts/play.py --task=go2trot --num_envs 1
```

Useful flags: `--load_run <run_dir>`, `--checkpoint <iter>`, `--viewer_ui`,
`--no-keyboard`. Keyboard teleop: I/K fwd/back, J/L strafe, N/M yaw, R reset.

Use `--device cpu` (the default); the MuJoCo warp/GPU backend has no viewer.

## Play without a viewer

```sh
uv run --frozen scripts/play.py --task=go2trot --headless
```

## Tests

```sh
uv run --frozen python -m pytest -q
```

## Gotchas

- `mjpython` + uv Python → set `DYLD_LIBRARY_PATH` as above (macOS only).
- `uv run scripts/play.py` without `--headless` on macOS raises
  "MuJoCo passive viewer on macOS requires mjpython" — use the mjpython command.
- `play.py` / `mac_run` with no `--load_run` picks the newest *directory* in
  `logs/<experiment>/` by ctime. MuJoCo writes `MUJOCO_LOG.TXT` into the CWD
  on sim warnings, so it can land in `logs/<experiment>/`; it used to be picked
  as the "latest run" (`NotADirectoryError`). Files are now ignored. Pass
  `--load_run <dir>` to choose a specific run.
- `uv lock` can't run on a machine without the licensed vlearn wheel (and the
  Unitree checkout). The `fld` extra's lock entries were produced by locking
  with the `vsim` extra stripped and transplanting the delta. If legged-fld
  gains a dependency, the lock must be updated the same way —
  `uv sync --frozen` won't install deps the lock doesn't know about.
