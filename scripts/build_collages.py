import argparse
import csv
import math
from pathlib import Path
from typing import List

import torch
from PIL import Image

from clip_sae.logging import get_logger
from clip_sae.sae import load_sae_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_latents", type=int, default=50)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="artifacts/autointerp/collages")
    p.add_argument(
        "--manifest_path", default="artifacts/autointerp/collages_manifest.csv"
    )
    p.add_argument("--max_shards", type=int, default=None)
    return p.parse_args()


def list_shards(cache_dir: str) -> List[Path]:
    return sorted(Path(cache_dir).glob("shard_*.pt"))


def create_collage(paths: List[str], out_path: Path, image_size: int) -> None:
    n = len(paths)
    if n == 0:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = Image.new("RGB", (cols * image_size, rows * image_size), (20, 20, 20))

    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = img.resize((image_size, image_size))
        r = i // cols
        c = i % cols
        canvas.paste(img, (c * image_size, r * image_size))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    logger = get_logger("build_collages")

    shard_paths = list_shards(args.cache_dir)
    if not shard_paths:
        logger.error("No shard_*.pt found in %s", args.cache_dir)
        raise SystemExit(1)
    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]

    sae = load_sae_from_checkpoint(args.checkpoint, args.device)

    all_paths: List[str] = []
    dict_size = None
    top_vals = None
    top_indices = None

    global_idx = 0

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        activations = shard["activations"].float()
        paths = shard["paths"]

        if dict_size is None:
            dict_size = sae.dict_size
            top_vals = torch.full((args.top_k, dict_size), -float("inf"))
            top_indices = torch.full((args.top_k, dict_size), -1, dtype=torch.long)

        for start in range(0, activations.shape[0], args.batch_size):
            batch = activations[start : start + args.batch_size]
            batch_paths = paths[start : start + args.batch_size]

            batch = batch.to(args.device)
            with torch.no_grad():
                z = sae.encode(batch).detach().cpu()  # (B, dict_size)

            b = z.shape[0]
            batch_indices = torch.arange(global_idx, global_idx + b).unsqueeze(1)
            batch_indices = batch_indices.expand(b, dict_size)

            combined_vals = torch.cat([top_vals, z], dim=0)
            combined_indices = torch.cat([top_indices, batch_indices], dim=0)

            new_vals, new_pos = torch.topk(combined_vals, k=args.top_k, dim=0)
            new_indices = combined_indices.gather(0, new_pos)

            top_vals = new_vals
            top_indices = new_indices

            all_paths.extend(batch_paths)
            global_idx += b

    if dict_size is None:
        logger.error("No activations found.")
        raise SystemExit(1)

    max_per_latent = top_vals[0]
    if args.num_latents is None or args.num_latents >= dict_size:
        latent_ids = torch.arange(dict_size)
    else:
        _, latent_ids = torch.topk(max_per_latent, k=args.num_latents)
        latent_ids = latent_ids.sort().values

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_id",
                "max_activation",
                "collage_path",
                "top_paths",
                "top_values",
            ],
        )
        writer.writeheader()

        for latent_id in latent_ids.tolist():
            top_idx = top_indices[:, latent_id].tolist()
            top_vals_latent = top_vals[:, latent_id].tolist()
            top_paths = [all_paths[i] for i in top_idx if i >= 0]

            collage_path = out_dir / f"latent_{latent_id}.png"
            create_collage(top_paths, collage_path, args.image_size)

            writer.writerow(
                {
                    "latent_id": latent_id,
                    "max_activation": max_per_latent[latent_id].item(),
                    "collage_path": str(collage_path),
                    "top_paths": ";".join(top_paths),
                    "top_values": ";".join([f"{v:.6f}" for v in top_vals_latent]),
                }
            )

    logger.info("Wrote collages to %s", out_dir)
    logger.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
