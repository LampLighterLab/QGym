import torch.nn as nn

from .utils import create_MLP

class Autoencoder(nn.Module):
    """Baseline MLP autoencoder over windows of motion.

    Shares the interface the periodic autoencoder (PAE) will use:
        x: (batch, num_channels, window_length)  ->  z: (batch, latent_dim)

    Keyword arguments:
    num_channels -- motion channels per time step (e.g. joint angles)
    window_length -- time steps per window
    latent_dim -- size of the bottleneck
    hidden_dims -- encoder hidden layers; the decoder mirrors them (default (128, 64))
    activation -- activation type, None for a linear autoencoder (default 'elu')
    """
    def __init__(
        self,
        num_channels,
        window_length,
        latent_dim,
        hidden_dims=(128,64),
        activation="elu"
    ):
        super().__init__()
        self.num_channels = num_channels
        self.window_length = window_length
        self.latent_dim = latent_dim
        num_inputs = num_channels * window_length
        self.encoder = create_MLP(num_inputs, latent_dim, hidden_dims, activation)
        self.decoder = create_MLP(latent_dim, num_inputs, hidden_dims[::-1], activation)

    def encode(self, x):
        return self.encoder(x.flatten(start_dim=1))

    def decode(self, z):
        return self.decoder(z).unflatten(1, (self.num_channels, self.window_length))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x):
        x_hat, _ = self(x)
        return (x_hat - x).pow(2).mean()
