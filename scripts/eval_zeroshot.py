import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision

from clip_sae.logging import get_logger
from clip_sae.sae import load_sae_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "stl10"], required=True
    )
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="artifacts/eval")
    p.add_argument("--run_name", default=None)
    p.add_argument("--sae_checkpoint", default=None)
    p.add_argument("--sae_layer", default=None)
    return p.parse_args()


def get_dataset(name: str, split: str, transform, download: bool):
    if name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root="data/torchvision",
            train=(split == "train"),
            download=download,
            transform=transform,
        )
    elif name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root="data/torchvision",
            train=(split == "train"),
            download=download,
            transform=transform,
        )
    else:
        # STL10 uses split names "train" and "test"
        ds = torchvision.datasets.STL10(
            root="data/torchvision",
            split=split,
            download=download,
            transform=transform,
        )
    return ds


def build_zeroshot_weights(open_clip, model, classnames, device):
    templates = ["a photo of a {}."]
    texts = [t.format(c) for c in classnames for t in templates]
    tokens = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    dim = text_features.shape[-1]
    weights = text_features.view(len(classnames), -1, dim).mean(dim=1)
    weights = weights / weights.norm(dim=-1, keepdim=True)
    return weights


def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    return modules.get(name)


def eval_accuracy(model, dataloader, text_weights, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_weights.t()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    return correct / max(total, 1)


def main():
    args = parse_args()
    logger = get_logger("eval_zeroshot")

    try:
        import open_clip
    except Exception as exc:
        logger.error("open_clip import failed: %s", exc)
        raise SystemExit(1)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval().to(args.device)

    transform = preprocess
    download = True
    if args.no_download:
        download = False
    if args.download:
        download = True
    ds = get_dataset(args.dataset, args.split, transform, download)

    if args.num_samples is not None:
        indices = list(range(min(args.num_samples, len(ds))))
        ds = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    classnames = ds.dataset.classes if isinstance(ds, torch.utils.data.Subset) else ds.classes
    text_weights = build_zeroshot_weights(open_clip, model, classnames, args.device)

    baseline_acc = eval_accuracy(model, loader, text_weights, args.device)

    sae_acc = None
    if args.sae_checkpoint:
        if not args.sae_layer:
            logger.error("--sae_layer is required when --sae_checkpoint is set")
            raise SystemExit(1)
        sae = load_sae_from_checkpoint(args.sae_checkpoint, args.device)
        module = get_module_by_name(model, args.sae_layer)
        if module is None:
            logger.error("Layer not found: %s", args.sae_layer)
            raise SystemExit(1)

        def hook_fn(_, __, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out.dim() == 3:
                cls = out[:, 0]
                x_hat = sae.reconstruct(cls)
                out = out.clone()
                out[:, 0] = x_hat
                return out
            if out.dim() == 2:
                x_hat = sae.reconstruct(out)
                return x_hat
            return out

        hook = module.register_forward_hook(hook_fn)
        sae_acc = eval_accuracy(model, loader, text_weights, args.device)
        hook.remove()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(ds),
        "model": args.model,
        "pretrained": args.pretrained,
        "baseline_acc": baseline_acc,
        "sae_acc": sae_acc,
        "sae_checkpoint": args.sae_checkpoint,
        "sae_layer": args.sae_layer,
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "num_samples",
                "model",
                "pretrained",
                "baseline_acc",
                "sae_acc",
                "sae_checkpoint",
                "sae_layer",
            ],
        )
        writer.writeheader()
        writer.writerow(result)

    logger.info("Saved eval results to %s", out_dir)


if __name__ == "__main__":
    main()
