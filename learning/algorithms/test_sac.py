"""Target-network updates follow optimizer steps, not rollout boundaries."""

import torch
from tensordict import TensorDict

from learning.algorithms.sac import SAC
from learning.modules import ChimeraActor, Critic


def test_targets_follow_every_gradient_step():
    actor = ChimeraActor(
        3,
        1,
        nn_params={
            "latent": {"hidden_dims": [4]},
            "mean": {"hidden_dims": []},
            "std": {"hidden_dims": []},
        },
        normalize_obs=False,
    )
    critics = [Critic(4, hidden_dims=[], normalize_obs=False) for _ in range(4)]
    for critic in critics:
        for parameter in critic.parameters():
            parameter.data.zero_()
    alg = SAC(
        actor,
        *critics,
        batch_size=2,
        max_gradient_steps=2,
        polyak=0.5,
        target_entropy=0.0,
    )
    assert alg.target_entropy == 0.0

    def critic_step(_):
        with torch.no_grad():
            for critic in (alg.critic_1, alg.critic_2):
                for parameter in critic.parameters():
                    parameter.add_(1)

    alg.update_critic = critic_step
    alg.update_actor_and_alpha = lambda _: None
    alg.update(TensorDict({"dummy": torch.zeros(1, 2)}, batch_size=(1, 2)))

    # Starting at zero: target = .5 after step 1, then .5*.5 + .5*2 = 1.25.
    for critic in (alg.target_critic_1, alg.target_critic_2):
        for parameter in critic.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 1.25))
