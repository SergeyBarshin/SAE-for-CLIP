import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x):
        return self.activation(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        x_hat, _ = self.forward(x)
        return x_hat
