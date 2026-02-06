import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="data/sample_images")
    p.add_argument("--layer", default="visual.transformer.resblocks.0")
    return p.parse_args()


def run(cmd):
    print("[check_pipeline]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    args = parse_args()

    run([
        "python",
        "scripts/extract_activations.py",
        "--image_dir",
        args.image_dir,
        "--model",
        "ViT-B-32",
        "--pretrained",
        "openai",
        "--layer",
        args.layer,
        "--num_samples",
        "64",
        "--batch_size",
        "16",
        "--shard_size",
        "64",
        "--out_dir",
        "artifacts/activation_cache",
    ])

    cache_root = Path("artifacts/activation_cache")
    runs = sorted([p for p in cache_root.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit("No activation cache runs found")
    cache_dir = str(runs[-1])

    run([
        "python",
        "scripts/train_sae.py",
        "--cache_dir",
        cache_dir,
        "--dict_size",
        "512",
        "--epochs",
        "1",
        "--batch_size",
        "64",
        "--lr",
        "1e-3",
        "--l1_lambda",
        "1e-3",
        "--device",
        "cpu",
    ])

    run([
        "python",
        "scripts/train_sae.py",
        "--sae_type",
        "msae",
        "--k_list",
        "16,32",
        "--cache_dir",
        cache_dir,
        "--dict_size",
        "256",
        "--epochs",
        "1",
        "--batch_size",
        "32",
        "--lr",
        "1e-3",
        "--l1_lambda",
        "1e-4",
        "--device",
        "cpu",
    ])

    ckpt_root = Path("artifacts/checkpoints")
    runs = sorted([p for p in ckpt_root.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit("No checkpoints found for autointerp")
    ckpt_dir = runs[-1]

    run([
        "python",
        "scripts/build_collages.py",
        "--cache_dir",
        cache_dir,
        "--checkpoint",
        str(ckpt_dir / "last.pt"),
        "--num_latents",
        "2",
        "--top_k",
        "4",
        "--image_size",
        "64",
        "--out_dir",
        "artifacts/autointerp/collages",
        "--manifest_path",
        "artifacts/autointerp/collages_manifest.csv",
    ])

    run([
        "python",
        "scripts/autointerp.py",
        "--manifest_path",
        "artifacts/autointerp/collages_manifest.csv",
        "--out_csv",
        "artifacts/autointerp/autointerp.csv",
        "--max_latents",
        "2",
    ])


if __name__ == "__main__":
    main()
