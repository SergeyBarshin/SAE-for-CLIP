from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .msae import MatryoshkaSAE
from .sae import SparseAutoencoder


def build_sae(
    sae_type: str,
    input_dim: int,
    dict_size: int,
    k_list: Optional[List[int]] = None,
    alpha_mode: str = "reverse",
    use_pre_bias: bool = True,
    use_enc_bias: bool = True,
    tied: bool = False,
    input_centering: str = "dataset",
    input_scaling: str = "dataset",
    soft_cap: float | None = None,
    input_mean: torch.Tensor | None = None,
    input_std: torch.Tensor | None = None,
):
    if sae_type == "sae":
        return SparseAutoencoder(input_dim, dict_size)
    if sae_type == "msae":
        if not k_list:
            raise ValueError("k_list must be provided for msae")
        return MatryoshkaSAE(
            d_in=input_dim,
            dict_size=dict_size,
            k_list=k_list,
            alpha=alpha_mode,
            use_pre_bias=use_pre_bias,
            use_enc_bias=use_enc_bias,
            tied=tied,
            input_centering=input_centering,
            input_scaling=input_scaling,
            soft_cap=soft_cap,
            input_mean=input_mean,
            input_std=input_std,
        )
    raise ValueError(f"Unknown sae_type: {sae_type}")


def load_sae_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    config_path = ckpt_path.parent / "config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    sae_type = config.get("sae_type", ckpt.get("sae_type", "sae"))
    dict_size = config.get("dict_size", ckpt.get("dict_size"))
    input_dim = config.get("input_dim", ckpt.get("input_dim"))

    k_list = config.get("k_list")
    alpha_mode = config.get("alpha_mode", "reverse")
    use_pre_bias = config.get("use_pre_bias", True)
    use_enc_bias = config.get("use_enc_bias", True)
    # Backward compatibility: older configs stored default flags as False.
    if "no_pre_bias" in config and "use_pre_bias" in config:
        if config["no_pre_bias"] is False and config["use_pre_bias"] is False:
            use_pre_bias = True
    if "no_enc_bias" in config and "use_enc_bias" in config:
        if config["no_enc_bias"] is False and config["use_enc_bias"] is False:
            use_enc_bias = True
    tied = config.get("tied", False)
    input_centering = config.get("input_centering", "dataset")
    input_scaling = config.get("input_scaling", "dataset")
    soft_cap = config.get("soft_cap")

    input_mean = None
    input_std = None
    if "input_mean" in config:
        input_mean = torch.tensor(config["input_mean"], dtype=torch.float32)
    if "input_std" in config:
        input_std = torch.tensor(config["input_std"], dtype=torch.float32)

    model = build_sae(
        sae_type,
        input_dim,
        dict_size,
        k_list=k_list,
        alpha_mode=alpha_mode,
        use_pre_bias=use_pre_bias,
        use_enc_bias=use_enc_bias,
        tied=tied,
        input_centering=input_centering,
        input_scaling=input_scaling,
        soft_cap=soft_cap,
        input_mean=input_mean,
        input_std=input_std,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model
