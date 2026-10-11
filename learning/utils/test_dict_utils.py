"""GAE separates bootstrap eligibility from trajectory continuation."""

import pytest
import torch
from tensordict import TensorDict

from learning.utils import compute_generalized_advantages


class ObservationValue:
    def evaluate(self, observations):
        return observations[..., 0]


@pytest.mark.parametrize("lam", [0.0, 0.8, 1.0])
def test_gae_bootstraps_time_limits_without_crossing_episode_boundaries(lam):
    # Columns: timeout, terminal, ongoing, and a timeout on the rollout's end.
    # The second row's large values belong to newly reset episodes, so they
    # must never become bootstrap values for the preceding episode.
    values = torch.tensor([[2.0, 2.0, 2.0, 2.0], [100.0, 100.0, 4.0, 4.0]])
    data = TensorDict(
        {
            "values": values,
            "rewards": torch.tensor([[1.0] * 4, [3.0] * 4]),
            "next_critic_obs": torch.tensor(
                [[[10.0], [10.0], [4.0], [4.0]], [[120.0], [120.0], [8.0], [8.0]]]
            ),
            "dones": torch.tensor([[True, True, False, False], [False] * 3 + [True]]),
            "timed_out": torch.tensor(
                [[True, False, False, False], [False] * 3 + [True]]
            ),
        },
        batch_size=(2, 4),
    )

    advantages = compute_generalized_advantages(data, 0.9, lam, ObservationValue())

    # Timeout uses final V=10, terminal uses zero; both stop the trace. The
    # continuing transition includes the next residual, weighted by gamma*lam.
    expected = torch.tensor(
        [
            [8.0, -1.0, 2.6 + 0.9 * lam * 6.2, 2.6 + 0.9 * lam * 6.2],
            [11.0, 11.0, 6.2, 6.2],
        ]
    )
    torch.testing.assert_close(advantages, expected)


def test_gae_single_terminal_transition_does_not_bootstrap():
    data = TensorDict(
        {
            "values": torch.tensor([[2.0]]),
            "rewards": torch.tensor([[1.0]]),
            "next_critic_obs": torch.tensor([[[100.0]]]),
            "dones": torch.tensor([[True]]),
            "timed_out": torch.tensor([[False]]),
        },
        batch_size=(1, 1),
    )
    result = compute_generalized_advantages(data, 0.9, 1.0, ObservationValue())
    torch.testing.assert_close(result, torch.tensor([[-1.0]]))
