import torch


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# @ torch.jit.script
def random_sample(env_ids, low, high, device):
    """
    Generate random samples for each entry of env_ids
    """
    rand_pos = torch_rand_float(0, 1, (len(env_ids), len(low)), device=device)
    diff_pos = (high - low).repeat(len(env_ids), 1)
    random_dof_pos = rand_pos * diff_pos + low.repeat(len(env_ids), 1)
    return random_dof_pos
