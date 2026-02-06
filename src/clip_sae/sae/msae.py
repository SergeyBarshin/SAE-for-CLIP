from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn


class MatryoshkaSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        dict_size: int,
        k_list: List[int],
        alpha: str = "reverse",
        use_pre_bias: bool = True,
        use_enc_bias: bool = True,
        tied: bool = False,
        input_centering: str = "dataset",
        input_scaling: str = "dataset",
        soft_cap: float | None = None,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not k_list:
            raise ValueError("k_list must be non-empty")
        k_list = sorted(set(int(k) for k in k_list))
        if k_list[0] <= 0 or k_list[-1] > dict_size:
            raise ValueError("k_list values must be in (0, dict_size]")

        self.d_in = d_in
        self.dict_size = dict_size
        self.k_list = k_list
        self.k_main = max(k_list)
        self.alpha_mode = alpha
        self.input_centering = input_centering
        self.input_scaling = input_scaling
        self.soft_cap = soft_cap
        self.eps = eps
        self.tied = tied

        self.encoder = nn.Linear(d_in, dict_size, bias=use_enc_bias)
        if not tied:
            self.decoder = nn.Linear(dict_size, d_in, bias=False)
        else:
            self.decoder = None

        if use_pre_bias:
            self.pre_bias = nn.Parameter(torch.zeros(d_in))
        else:
            self.pre_bias = None

        mean = input_mean if input_mean is not None else torch.zeros(d_in)
        std = input_std if input_std is not None else torch.ones(d_in)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

    def _decode(self, z):
        if self.tied:
            x_hat = torch.matmul(z, self.encoder.weight)
        else:
            x_hat = self.decoder(z)
        if self.pre_bias is not None:
            x_hat = x_hat + self.pre_bias
        return x_hat

    def normalize(self, x):
        x_norm = x
        if self.input_centering == "dataset":
            x_norm = x_norm - self.input_mean
        if self.input_scaling == "dataset":
            x_norm = x_norm / (self.input_std + self.eps)
        return x_norm

    def denormalize(self, x_norm):
        x = x_norm
        if self.input_scaling == "dataset":
            x = x * (self.input_std + self.eps)
        if self.input_centering == "dataset":
            x = x + self.input_mean
        return x

    def _apply_topk(self, a: torch.Tensor, k: int) -> torch.Tensor:
        values, indices = torch.topk(a, k, dim=-1)
        z = torch.zeros_like(a)
        z.scatter_(-1, indices, values)
        return z

    def encode(self, x):
        x_norm = self.normalize(x)
        h = self.encoder(x_norm)
        a = torch.relu(h)
        if self.soft_cap is not None:
            a = self.soft_cap * torch.tanh(a / self.soft_cap)
        z_main = self._apply_topk(a, self.k_main)
        return z_main

    def decode(self, z):
        return self._decode(z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        x_norm = self.normalize(x)
        h = self.encoder(x_norm)
        a = torch.relu(h)
        if self.soft_cap is not None:
            a = self.soft_cap * torch.tanh(a / self.soft_cap)

        aux: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k in self.k_list:
            z_k = self._apply_topk(a, k)
            x_hat_k = self._decode(z_k)
            aux[k] = (x_hat_k, z_k)

        x_hat_main, z_main = aux[self.k_main]
        return x_hat_main, z_main, aux

    def reconstruct(self, x):
        x_hat_norm, _, _ = self.forward(x)
        return self.denormalize(x_hat_norm)
