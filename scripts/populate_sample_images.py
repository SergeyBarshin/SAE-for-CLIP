import argparse
from pathlib import Path

from PIL import Image
import torchvision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/sample_images")
    p.add_argument("--num_images", type=int, default=200)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--split", choices=["train", "test"], default="train")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cls = (
        torchvision.datasets.CIFAR10
        if args.dataset == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    dataset = dataset_cls(
        root="data/torchvision", train=(args.split == "train"), download=True
    )

    count = 0
    for idx, (img, label) in enumerate(dataset):
        if count >= args.num_images:
            break
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        filename = out_dir / f"{args.dataset}_{args.split}_{idx:05d}_label{label}.png"
        if filename.exists():
            continue
        img.save(filename)
        count += 1

    print(f"[populate_sample_images] Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()
