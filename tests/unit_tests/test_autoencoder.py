import math
import pytest
import torch

from learning.modules.autoencoder import Autoencoder

NUM_CHANNELS = 3
WINDOW_LENGTH = 51
LATENT_DIM = 8

@pytest.fixture
def ae():
    torch.manual_seed(0)
    return Autoencoder(NUM_CHANNELS, WINDOW_LENGTH, LATENT_DIM)

@pytest.mark.parametrize("batch_size",[1,16])
def test_shapes(ae, batch_size):
    x = torch.randn(batch_size, NUM_CHANNELS, WINDOW_LENGTH)
    x_hat, z = ae(x)
    assert x_hat.shape == x.shape
    assert z.shape == (batch_size, LATENT_DIM)

def test_samples_are_indpendent(ae):
    x = torch.randn(16, NUM_CHANNELS, WINDOW_LENGTH)
    x_hat, _ = ae(x)
    x_hat_single, _ = ae(x[3:4])
    torch.testing.assert_close(x_hat[3:4], x_hat_single)


def test_gradient_reaches_every_parameter(ae):
    ae.loss(torch.randn(16, NUM_CHANNELS, WINDOW_LENGTH)).backward()
    for name, param in ae.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().sum() >0, name

def sinusoid_windows(num_windows, window_length, num_cycles=2.0, noise_std=0.0):
    """
    x(t) = a cos(wt) + b sin(wt)
    """
    t = torch.arange(window_length) / window_length
    omega = 2 * math.pi * num_cycles
    a, b = torch.randn(2, num_windows, 1, 1)
    x = a * torch.cos(omega * t) + b * torch.sin(omega * t)
    return x + noise_std * torch.randn_like(x)

def fit(ae, x, num_steps=2000, lr=1e-2):
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = ae.loss(x)
        loss.backward()
        optimizer.step()
    return ae.loss(x).item()

def test_rank2_sinusoids_reconstruct_with_latent_2():
    torch.manual_seed(0)
    x = sinusoid_windows(256, WINDOW_LENGTH)
    ae = Autoencoder(1, WINDOW_LENGTH, latent_dim = 2, hidden_dims = (), activation = None) #noqa
    assert fit(ae, x) < 1e-4 * x.pow(2).mean()

def test_linear_latent_1_matches_pca_bound():
    torch.manual_seed(0)
    x = sinusoid_windows(256, WINDOW_LENGTH)
    X = x.flatten(start_dim=1)
    X = X - X.mean(dim=0) # PCA is about the mean, and AE bias learns this
    singular_vals = torch.linalg.svdvals(X)
    pca_mse = singular_vals[1:].pow(2).sum() / X.numel()
    ae = Autoencoder(1, WINDOW_LENGTH, latent_dim=1, hidden_dims=(), activation = None)
    ae_mse = fit(ae, x)
    assert ae_mse >= pca_mse * (1 - 1e-3) # based on Eckhart Young formula (still confused on this) #noqa
    assert ae_mse <= pca_mse * 1.05
