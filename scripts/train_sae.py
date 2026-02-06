import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from torch import optim

from clip_sae.logging import get_logger
from clip_sae.sae import build_sae
from clip_sae.sae.losses import msae_loss
from clip_sae.sae.metrics import (
    batch_metrics_msae,
    batch_metrics_sae,
    finalize_evr,
    update_sse_sst,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--dict_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1_lambda", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_shards", type=int, default=None)
    p.add_argument("--sae_type", choices=["sae", "msae"], default="sae")
    p.add_argument("--k_list", default=None)
    p.add_argument("--alpha_mode", choices=["uniform", "reverse"], default="reverse")
    p.add_argument("--soft_cap", type=float, default=None)
    p.add_argument("--input_centering", choices=["none", "dataset"], default="dataset")
    p.add_argument("--input_scaling", choices=["none", "dataset"], default="dataset")
    p.add_argument("--use_pre_bias", action="store_true")
    p.add_argument("--no_pre_bias", action="store_true")
    p.add_argument("--use_enc_bias", action="store_true")
    p.add_argument("--no_enc_bias", action="store_true")
    p.add_argument("--tied", action="store_true")
    return p.parse_args()


def list_shards(cache_dir: str):
    paths = sorted(Path(cache_dir).glob("shard_*.pt"))
    return paths


def iter_batches(acts, batch_size):
    n = acts.shape[0]
    perm = torch.randperm(n)
    acts = acts[perm]
    for i in range(0, n, batch_size):
        yield acts[i : i + batch_size]


def parse_k_list(k_list_str: str | None) -> List[int] | None:
    if k_list_str is None:
        return None
    return [int(k.strip()) for k in k_list_str.split(",") if k.strip()]


def compute_dataset_stats(shard_paths: List[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    total = 0
    sum_vec = None
    sum_sq = None

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")["activations"].float()
        if sum_vec is None:
            sum_vec = torch.zeros(shard.shape[1], dtype=torch.float64)
            sum_sq = torch.zeros(shard.shape[1], dtype=torch.float64)
        sum_vec += shard.sum(dim=0, dtype=torch.float64)
        sum_sq += (shard * shard).sum(dim=0, dtype=torch.float64)
        total += shard.shape[0]

    mean = sum_vec / max(total, 1)
    var = (sum_sq / max(total, 1)) - (mean * mean)
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)
    return mean.float(), std.float()


def main():
    args = parse_args()
    logger = get_logger("train_sae")

    shard_paths = list_shards(args.cache_dir)
    if not shard_paths:
        logger.error("No shard_*.pt found in %s", args.cache_dir)
        raise SystemExit(1)

    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]

    sample = torch.load(shard_paths[0], map_location="cpu")["activations"]
    input_dim = sample.shape[1]

    k_list = parse_k_list(args.k_list)
    if args.sae_type == "msae" and not k_list:
        logger.error("--k_list is required for msae")
        raise SystemExit(1)

    use_pre_bias = True
    if args.no_pre_bias:
        use_pre_bias = False
    if args.use_pre_bias:
        use_pre_bias = True

    use_enc_bias = True
    if args.no_enc_bias:
        use_enc_bias = False
    if args.use_enc_bias:
        use_enc_bias = True

    input_mean = None
    input_std = None
    if args.sae_type == "msae" and (
        args.input_centering == "dataset" or args.input_scaling == "dataset"
    ):
        logger.info("Computing dataset mean/std for normalization...")
        input_mean, input_std = compute_dataset_stats(shard_paths)

    model = build_sae(
        args.sae_type,
        input_dim,
        args.dict_size,
        k_list=k_list,
        alpha_mode=args.alpha_mode,
        use_pre_bias=use_pre_bias,
        use_enc_bias=use_enc_bias,
        tied=args.tied,
        input_centering=args.input_centering,
        input_scaling=args.input_scaling,
        soft_cap=args.soft_cap,
        input_mean=input_mean,
        input_std=input_std,
    ).to(args.device)

    logger.info(
        "sae_type=%s dict_size=%d k_list=%s alpha_mode=%s",
        args.sae_type,
        args.dict_size,
        k_list,
        args.alpha_mode,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_metric_fields: List[str] = []
    if args.sae_type == "msae":
        for k in k_list:
            extra_metric_fields.extend([f"mse_k{k}", f"r2_k{k}", f"l0_k{k}"])

    metrics_fields = [
        "epoch",
        "mse",
        "l1",
        "l0",
        "r2",
        "evr_global",
        "dict_size",
        "k_list",
        "alpha_mode",
    ] + extra_metric_fields

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fields)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"mse": 0.0, "l1": 0.0, "l0": 0.0, "r2": 0.0}
        for field in extra_metric_fields:
            agg[field] = 0.0
        steps = 0

        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")["activations"].float()
            for batch in iter_batches(shard, args.batch_size):
                batch = batch.to(args.device)

                if args.sae_type == "msae":
                    x_norm = model.normalize(batch)
                    outputs = model(batch)
                    loss, loss_info = msae_loss(
                        x_norm,
                        outputs,
                        args.l1_lambda,
                        k_list,
                        args.alpha_mode,
                        model.k_main,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    metrics = batch_metrics_msae(x_norm, outputs, k_list, model.k_main)
                    agg["mse"] += metrics["mse"]
                    agg["r2"] += metrics["r2"]
                    agg["l0"] += metrics["l0"]
                    agg["l1"] += loss_info["l1"].item()
                    for field in extra_metric_fields:
                        agg[field] += metrics[field]
                else:
                    x_hat, z = model(batch)
                    mse = torch.mean((batch - x_hat) ** 2)
                    l1 = torch.mean(torch.abs(z))
                    loss = mse + args.l1_lambda * l1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    metrics = batch_metrics_sae(batch, x_hat.detach(), z.detach())
                    for k in agg:
                        if k in metrics:
                            agg[k] += metrics[k]

                steps += 1

        if steps == 0:
            logger.error("No training steps executed.")
            raise SystemExit(1)

        for k in agg:
            agg[k] /= steps

        # global EVR for this epoch
        model.eval()
        evr_accum = None
        with torch.no_grad():
            for shard_path in shard_paths:
                shard = torch.load(shard_path, map_location="cpu")["activations"].float()
                for batch in iter_batches(shard, args.batch_size):
                    batch = batch.to(args.device)
                    if args.sae_type == "msae":
                        x_norm = model.normalize(batch)
                        x_hat_norm, _, _ = model(batch)
                        evr_accum = update_sse_sst(evr_accum, x_norm, x_hat_norm)
                    else:
                        x_hat, _ = model(batch)
                        evr_accum = update_sse_sst(evr_accum, batch, x_hat)
        evr_global = finalize_evr(evr_accum)

        logger.info(
            "epoch %d | mse=%.6f l1=%.6f l0=%.2f r2=%.4f evr_global=%.4f",
            epoch,
            agg["mse"],
            agg["l1"],
            agg["l0"],
            agg["r2"],
            evr_global,
        )

        row = {
            "epoch": epoch,
            "mse": agg["mse"],
            "l1": agg["l1"],
            "l0": agg["l0"],
            "r2": agg["r2"],
            "evr_global": evr_global,
            "dict_size": args.dict_size,
            "k_list": ",".join(str(k) for k in k_list) if k_list else "",
            "alpha_mode": args.alpha_mode,
        }
        for field in extra_metric_fields:
            row[field] = agg[field]

        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            writer.writerow(row)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": input_dim,
        "dict_size": args.dict_size,
        "sae_type": args.sae_type,
    }
    torch.save(ckpt, out_dir / "last.pt")

    config = vars(args)
    config["input_dim"] = input_dim
    if k_list:
        config["k_list"] = k_list
    if input_mean is not None:
        config["input_mean"] = input_mean.tolist()
    if input_std is not None:
        config["input_std"] = input_std.tolist()

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved checkpoint to %s", out_dir)


if __name__ == "__main__":
    main()
