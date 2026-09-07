"""Replay logged deployment observations through the actor network.

Reads `control.csv` from a `logs/deploy` run, feeds the recorded (already
scaled) observation columns through the actor loaded by
`rl_controller.setup_actor()`, and writes the resulting actions to a CSV.

Usage (from the repository root):

    uv run --frozen python go2_deploy/utility/replay_actor.py 2026-08-12_11-16-05
    uv run --frozen python go2_deploy/utility/replay_actor.py \
        logs/deploy/2026-08-12_11-16-05 --run-name <policy_run> --compare
"""

import argparse
import csv
import json
import os
import sys

import torch

_UTILITY_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_DIR = os.path.dirname(_UTILITY_DIR)
_REPO_ROOT = os.path.dirname(_DEPLOY_DIR)

# rl_controller / deploy_config are imported by bare module name on the robot,
# so both the repository root and go2_deploy have to be importable here too.
for _path in (_REPO_ROOT, _DEPLOY_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from deploy_config import DeployConfig  # noqa: E402
from rl_controller import setup_actor  # noqa: E402

DEFAULT_LOG_ROOT = os.path.join(_REPO_ROOT, "logs", "deploy")
CHUNK_SIZE = 4096


def resolve_log_dir(log_dir, log_root=DEFAULT_LOG_ROOT):
    """Accept either a path to a run directory or a bare run name."""
    if os.path.isdir(log_dir):
        return log_dir
    candidate = os.path.join(log_root, log_dir)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"No deployment log directory found for '{log_dir}'")


def obs_columns(log_dir, cfg):
    """Observation column names, in obs_vector order.

    meta.json records the obs_vector that was live during the run, which may
    differ from the current DeployConfig; prefer it when present.
    """
    obs_vector = list(cfg.obs_vector)
    meta_path = os.path.join(log_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        obs_vector = meta.get("obs_vector", obs_vector)

    columns = []
    for obs in obs_vector:
        if obs not in cfg.obs_sizes:
            raise KeyError(f"observation '{obs}' from meta.json is not in obs_sizes")
        columns += [f"obs_{obs}_{i}" for i in range(cfg.obs_sizes[obs])]
    return columns


def read_control_csv(log_dir, columns):
    """Return (rows, meta_rows): observation values and their t/state/phase."""
    path = os.path.join(log_dir, "control.csv")
    rows = []
    meta_rows = []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"{path} is missing observation columns {missing}; the log was "
                "written with a different obs_vector than this config expects"
            )
        for row in reader:
            values = [row[c] for c in columns]
            # log_control() writes blanks when the control loop had no obs yet
            if any(v == "" or v is None for v in values):
                skipped += 1
                continue
            rows.append([float(v) for v in values])
            meta_rows.append(
                {
                    "t": row.get("t", ""),
                    "state": row.get("state", ""),
                    "phase": row.get("phase", ""),
                    "logged_action": [row.get(f"action_{i}", "") for i in range(12)],
                }
            )
    return rows, meta_rows, skipped


def run_actor(actor, rows, scale):
    """Feed observations through the actor in chunks, returning a (N, 12) tensor."""
    obs = torch.tensor(rows, dtype=torch.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], CHUNK_SIZE):
            outputs.append(actor.act_inference(obs[start : start + CHUNK_SIZE]))
    if not outputs:
        return torch.zeros((0, 12))
    return torch.cat(outputs) * scale


def write_actions_csv(out_path, meta_rows, actions, compare):
    header = ["t", "state", "phase"] + [f"action_{i}" for i in range(12)]
    if compare:
        header += [f"logged_action_{i}" for i in range(12)]
        header += [f"diff_{i}" for i in range(12)]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for meta, action in zip(meta_rows, actions.tolist()):
            row = [meta["t"], meta["state"], meta["phase"]] + action
            if compare:
                logged = meta["logged_action"]
                row += logged
                row += [
                    (a - float(logged_value)) if logged_value != "" else ""
                    for a, logged_value in zip(action, logged)
                ]
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        help="Run directory under logs/deploy, or a bare run name such as "
        "2026-08-12_11-16-05",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Policy run under logs/<experiment> to load (default: most recent)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <log_dir>/replayed_actions.csv)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also emit the actions logged on the robot and their difference",
    )
    args = parser.parse_args(argv)

    log_dir = resolve_log_dir(args.log_dir)
    cfg = DeployConfig()
    columns = obs_columns(log_dir, cfg)
    rows, meta_rows, skipped = read_control_csv(log_dir, columns)
    print(f"Read {len(rows)} rows ({skipped} skipped for blank obs) from {log_dir}")

    actor = setup_actor(args.run_name)
    if actor.num_obs != len(columns):
        raise ValueError(
            f"actor expects {actor.num_obs} observations but the log provides "
            f"{len(columns)}; the checkpoint does not match this obs_vector"
        )

    # Matches RLController.act(): the actor output is scaled to radians.
    scale = torch.tensor(getattr(cfg.DeployScaling, "dof_pos_target", 1.0))
    actions = run_actor(actor, rows, scale)

    out_path = args.output or os.path.join(log_dir, "replayed_actions.csv")
    write_actions_csv(out_path, meta_rows, actions, args.compare)
    print(f"Wrote {actions.shape[0]} actions to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
