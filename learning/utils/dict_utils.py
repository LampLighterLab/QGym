import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad
def compute_MC_returns(data: TensorDict, gamma, critic=None):
    # todo not as accurate as taking
    if critic is None:
        last_values = torch.zeros_like(data["rewards"][0])
    else:
        last_values = critic.evaluate(data["critic_obs"][-1])

    returns = torch.zeros_like(data["rewards"])
    returns[-1] = data["rewards"][-1] + gamma * last_values * ~data["terminated"][-1]
    for k in reversed(range(data["rewards"].shape[0] - 1)):
        not_done = ~data["dones"][k]
        returns[k] = data["rewards"][k] + gamma * returns[k + 1] * not_done
        if critic is not None:
            returns[k] += (
                gamma
                * critic.evaluate(data["critic_obs"][k])
                * data["timed_out"][k]
                * ~data["terminated"][k]
            )
    return returns


@torch.no_grad
def normalize(input, eps=1e-8):
    return (input - input.mean()) / (input.std() + eps)


@torch.no_grad
def compute_generalized_advantages(data, gamma, lam, critic):
    # Within an uninterrupted trajectory the next stored value is V(s_next).
    # At a timeout it instead belongs to a reset state, so evaluate the saved
    # physical successor. The last rollout step also needs an explicit value.
    next_values = data["values"].roll(-1, dims=0)
    bootstrap_boundary = data["timed_out"].clone()
    bootstrap_boundary[-1] = True
    next_values[bootstrap_boundary] = critic.evaluate(
        data["next_critic_obs"][bootstrap_boundary]
    )
    not_done = ~data["dones"]
    can_bootstrap = not_done | data["timed_out"]
    td_errors = data["rewards"] + gamma * next_values * can_bootstrap - data["values"]
    advantages = torch.zeros_like(data["values"])
    advantages[-1] = td_errors[-1]
    for k in reversed(range(data["values"].shape[0] - 1)):
        advantages[k] = td_errors[k] + gamma * lam * not_done[k] * advantages[k + 1]

    return advantages


# todo change num_epochs to num_batches
@torch.no_grad
def create_uniform_generator(data, batch_size, max_gradient_steps=100):
    n, m = data.shape
    total_data = n * m

    if batch_size > total_data:
        batch_size = total_data

    num_batches_per_epoch = total_data // batch_size
    num_epochs = max_gradient_steps // num_batches_per_epoch
    last_epoch_batches = max_gradient_steps - num_epochs * num_batches_per_epoch

    for epoch in range(num_epochs):
        indices = torch.randperm(total_data, device=data.device)
        for i in range(num_batches_per_epoch):
            batched_data = data.flatten(0, 1)[
                indices[i * batch_size : (i + 1) * batch_size]
            ]
            yield batched_data
    else:
        indices = torch.randperm(total_data, device=data.device)
        for i in range(last_epoch_batches):
            batched_data = data.flatten(0, 1)[
                indices[i * batch_size : (i + 1) * batch_size]
            ]
            yield batched_data


@torch.no_grad
def export_to_numpy(data, path):
    # check if path ends iwth ".npz", and if not append it.
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **{key: val.cpu().numpy() for key, val in data.items()})
    return
