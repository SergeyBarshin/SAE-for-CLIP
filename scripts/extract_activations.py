import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from clip_sae.logging import get_logger
from clip_sae.utils import ensure_dir, list_images


def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        return None
    return modules[name]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--layer", required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--shard_size", type=int, default=2048)
    p.add_argument("--out_dir", default="artifacts/activation_cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger = get_logger("extract_activations")

    try:
        import open_clip
    except Exception as exc:
        logger.error("open_clip import failed: %s", exc)
        raise SystemExit(1)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval().to(args.device)

    module = get_module_by_name(model, args.layer)
    if module is None:
        names = list(dict(model.named_modules()).keys())
        sample = "\n".join(names[:50])
        logger.error("Layer not found: %s", args.layer)
        logger.error("First 50 module names:\n%s", sample)
        raise SystemExit(1)

    act_buffer = {}

    def hook_fn(_, __, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        act_buffer["act"] = out

    hook = module.register_forward_hook(hook_fn)

    paths = list_images(args.image_dir)
    if not paths:
        logger.error("No images found in %s", args.image_dir)
        raise SystemExit(1)

    if args.num_samples:
        paths = paths[: args.num_samples]

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    ensure_dir(run_dir)

    shard_idx = 0
    shard_acts = []
    shard_paths = []

    with torch.no_grad():
        for start in tqdm(range(0, len(paths), args.batch_size)):
            batch_paths = paths[start : start + args.batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            batch = torch.stack(images).to(args.device)

            _ = model.encode_image(batch)
            acts = act_buffer.get("act")
            if acts is None:
                logger.error("Hook did not capture activations.")
                raise SystemExit(1)

            if acts.dim() == 3:
                acts = acts[:, 0]
            elif acts.dim() != 2:
                logger.error("Unexpected activation dims: %s", tuple(acts.shape))
                raise SystemExit(1)
            acts = acts.detach().cpu()

            shard_acts.append(acts)
            shard_paths.extend([str(p) for p in batch_paths])

            if sum(a.shape[0] for a in shard_acts) >= args.shard_size:
                shard_tensor = torch.cat(shard_acts, dim=0)
                out_path = run_dir / f"shard_{shard_idx:04d}.pt"
                torch.save({"activations": shard_tensor, "paths": shard_paths}, out_path)
                shard_idx += 1
                shard_acts = []
                shard_paths = []

        if shard_acts:
            shard_tensor = torch.cat(shard_acts, dim=0)
            out_path = run_dir / f"shard_{shard_idx:04d}.pt"
            torch.save({"activations": shard_tensor, "paths": shard_paths}, out_path)

    hook.remove()

    metadata = {
        "model": args.model,
        "pretrained": args.pretrained,
        "layer": args.layer,
        "num_samples": len(paths),
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "out_dir": str(run_dir),
        "device": args.device,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote shards to %s", run_dir)


if __name__ == "__main__":
    main()
