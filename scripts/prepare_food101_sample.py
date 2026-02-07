import argparse
import random
from pathlib import Path

from PIL import Image
import torchvision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/food101_sample")
    p.add_argument("--num_images", type=int, default=1000)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = torchvision.datasets.Food101(
        root="data/torchvision", split=args.split, download=True
    )
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_images]

    count = 0
    for idx in indices:
        img, label = ds[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        class_name = ds.classes[label]
        filename = out_dir / f"food101_{args.split}_{idx:06d}_{class_name}.jpg"
        if filename.exists():
            continue
        img.save(filename)
        count += 1

    print(f"[prepare_food101_sample] Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()
