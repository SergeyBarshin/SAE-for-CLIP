from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def compute_r2(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    x_mean = torch.mean(x, dim=0, keepdim=True)
    sst = torch.sum((x - x_mean) ** 2)
    sse = torch.sum((x - x_hat) ** 2)
    if sst.item() == 0:
        return torch.tensor(0.0, device=x.device)
    return 1.0 - (sse / sst)


def batch_metrics_sae(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((x - x_hat) ** 2)
    l1 = torch.mean(torch.abs(z))
    l0 = torch.mean((z > 0).float().sum(dim=1))
    r2 = compute_r2(x, x_hat)
    return {
        "mse": mse.item(),
        "l1": l1.item(),
        "l0": l0.item(),
        "r2": r2.item(),
    }


def batch_metrics_msae(
    x_norm: torch.Tensor,
    outputs: Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    k_list: List[int],
    k_main: int,
) -> Dict[str, float]:
    x_hat_main, z_main, aux = outputs
    metrics: Dict[str, float] = {}

    mse_main = torch.mean((x_norm - x_hat_main) ** 2)
    r2_main = compute_r2(x_norm, x_hat_main)
    l0_main = torch.mean((z_main > 0).float().sum(dim=1))

    metrics["mse"] = mse_main.item()
    metrics["r2"] = r2_main.item()
    metrics["l0"] = l0_main.item()

    for k in k_list:
        x_hat_k, z_k = aux[k]
        mse_k = torch.mean((x_norm - x_hat_k) ** 2)
        r2_k = compute_r2(x_norm, x_hat_k)
        l0_k = torch.mean((z_k > 0).float().sum(dim=1))
        metrics[f"mse_k{k}"] = mse_k.item()
        metrics[f"r2_k{k}"] = r2_k.item()
        metrics[f"l0_k{k}"] = l0_k.item()

    return metrics


def update_sse_sst(accum, x, x_hat):
    if accum is None:
        accum = {"sse": 0.0, "sst": 0.0}
    x_mean = torch.mean(x, dim=0, keepdim=True)
    sst = torch.sum((x - x_mean) ** 2)
    sse = torch.sum((x - x_hat) ** 2)
    accum["sse"] += sse.item()
    accum["sst"] += sst.item()
    return accum


def finalize_evr(accum):
    if accum is None or accum["sst"] == 0:
        return 0.0
    return 1.0 - (accum["sse"] / accum["sst"])
