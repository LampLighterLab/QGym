#!/usr/bin/env python3
"""Train one domain-randomization campaign cell.

The public trainer follows the task config exactly. This worker is intentionally
campaign-specific: it selects an ablation on a private config copy before any
backend topology is constructed, then delegates to the normal training setup.
"""

import copy

from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.utils.task_registry import task_registry
from scripts.train import make_train_parser, setup, train


BUNDLES = {
    "off": "off",
    "friction": "friction-only",
    "pd": "pd-only",
    "mass": "mass-only",
    "all": "config",
}


def get_args(argv=None):
    parser = make_train_parser()
    parser.description = "Train one Q2 domain-randomization campaign cell"
    parser.add_argument("--dr-bundle", choices=BUNDLES, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    import gym.envs  # noqa: F401 — triggers task registration

    registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(args.task)
    env_cfg = copy.deepcopy(registered_env_cfg)
    train_cfg = copy.deepcopy(registered_train_cfg)
    apply_domain_randomization_override(env_cfg, BUNDLES[args.dr_bundle])

    train_cfg, policy_runner = setup(args, env_cfg, train_cfg)
    train(train_cfg, policy_runner)


if __name__ == "__main__":
    main()
