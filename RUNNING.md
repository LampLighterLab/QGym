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

### Train the conv autoencoder on retargeted motions

Inputs are kine2go-pipeline `.npy` files ((T, 61) at 60 Hz; only the 12 joint
angles are used). They are >100 KiB, so they live outside the repo: on the
Linux box in `kine2go-pipeline/motion_retargeting/results/`, copied by hand
elsewhere. Training uses CUDA, then MPS, then CPU.

Hold out whole clips (the real generalization test):

```sh
R=~/Documents/fourier-stuff/kine2go-pipeline/motion_retargeting/results
uv run --frozen --extra fld scripts/train_autoencoder.py \
    $R/dog_walk0{0,1,2,4,6}_joint_pos.npy \
    --val_motions $R/dog_walk03_joint_pos.npy --kernel_size 13 [--seed 0]
```

or the end of every clip (`--train_fraction 0.8` trains on the first 80% of
each; a tail shorter than one 61-frame window is skipped with a warning). The
two options are exclusive. Use `--kernel_size 13` (default 61 = the old
window-wide kernel, which memorizes). Normalization uses training data only and
no window spans two clips. Outputs are
`logs/autoencoder/conv_ae_L<L>_<split>_K<kernel>_s<seed>.{png,pt}`, with
`<split>` = `val-walk03` or `train0.8`; the `.pt` stores the loss history and
the clip basenames.

Leave-one-walk-out sweep (6 walks x 3 seeds = 18 runs; `R` as above):

```sh
W=(00 01 02 03 04 06); for V in "${W[@]}"; do T=(); for U in "${W[@]}"; do [ "$U" != "$V" ] && T+=($R/dog_walk${U}_joint_pos.npy); done; for S in 0 1 2; do uv run --frozen --extra fld scripts/train_autoencoder.py "${T[@]}" --val_motions $R/dog_walk${V}_joint_pos.npy --kernel_size 13 --seed $S; done; done
```

Evaluate checkpoints (`--data_dir` is where the clips are; plots go to
`logs/autoencoder/plots/`). It prints held-out MSE next to the nearest-window
(memorization) and same-size PCA baselines and the jitter floor, then a table
averaged over seeds:

```sh
uv run --frozen --extra fld scripts/evaluate_autoencoder.py \
    --data_dir $R logs/autoencoder/*val-walk*.pt
```

W&B works like `scripts/train.py`: copy `user/wandb_config_default.json` to
`user/wandb_config.json` (gitignored) with your entity/project, or pass
`--wandb_entity`/`--wandb_project`. Without either (or with `--disable_wandb`)
the script prints `WARNING: W&B disabled (...)` and only saves locally — check
for that line before a long sweep. Run `uv run --frozen wandb login` once per
machine. Logged keys are listed in the `train_autoencoder.py` docstring
(per-clip `val/<clip>` plus pooled `val/mse`, gradient/weight stats, per-joint
RMSE, latent stats, window-0 plots, the `.pt` as a `model` artifact).
`WANDB_MODE=offline` logs to `logs/wandb/` without uploading
(`wandb sync <run dir>` later).

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

- A stale `WANDB_API_KEY` in your shell profile overrides a valid `~/.netrc`
  login (W&B then fails with 401). Remove the export, or run with
  `env -u WANDB_API_KEY`.
- legged-fld history rewritten (amend + force-push): `uvx gitman update` keeps
  the old SHA because the checkout has diverged. Run
  `git -C thirdparty/legged-fld reset --hard origin/<rev>` (it is a managed,
  read-only checkout), then `uvx gitman update` and commit `gitman.yml`.

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
