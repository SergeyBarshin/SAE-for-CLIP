from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _alpha_weights(k_list: List[int], alpha_mode: str) -> Dict[int, float]:
    if alpha_mode == "uniform":
        return {k: 1.0 for k in k_list}
    if alpha_mode == "reverse":
        max_k = max(k_list)
        return {k: max_k / k for k in k_list}
    raise ValueError(f"Unknown alpha_mode: {alpha_mode}")


def msae_loss(
    x_norm: torch.Tensor,
    outputs: Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    l1_lambda: float,
    k_list: List[int],
    alpha_mode: str,
    k_main: int,
):
    x_hat_main, z_main, aux = outputs
    alphas = _alpha_weights(k_list, alpha_mode)

    mse_k = {}
    total = 0.0
    for k in k_list:
        x_hat_k, _ = aux[k]
        mse = torch.mean((x_norm - x_hat_k) ** 2)
        mse_k[k] = mse
        total = total + alphas[k] * mse

    l1 = torch.mean(torch.abs(z_main))
    total = total + l1_lambda * l1

    return total, {
        "mse_k": mse_k,
        "l1": l1,
        "x_hat_main": x_hat_main,
        "z_main": z_main,
    }
