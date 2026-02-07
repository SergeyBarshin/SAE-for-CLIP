# SAE for CLIP

Educational reimplementation project for Sparse Autoencoders (SAE) on CLIP.

See the current progress report: `REPORT.md`.

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quickstart (local)

```bash
make check
make sample-images
```

This downloads CIFAR-10 via `torchvision` and saves 200 images into `data/sample_images/`.

What this does:
- Ensures required repo directories exist.
- Verifies `open_clip` import (non-fatal until deps are installed).

## Extract CLIP Activations

Prepare a small set of images in `data/sample_images/`.

```bash
python scripts/extract_activations.py \
  --image_dir data/sample_images \
  --model ViT-B-32 \
  --pretrained openai \
  --layer visual.transformer.resblocks.0 \
  --num_samples 200 \
  --batch_size 16 \
  --shard_size 256 \
  --out_dir artifacts/activation_cache
```

Outputs:
- `artifacts/activation_cache/<run>/shard_*.pt`
- `artifacts/activation_cache/<run>/metadata.json`

Tip: Use `scripts/check_clip.py --layer <name>` to validate a layer name.

## Colab Full Run

Use `notebooks/colab_run_full.ipynb` for the full Food101 + STL‑10 pipeline in Colab.  
It clones the repo, installs deps, extracts activations, trains MSAE, runs zero‑shot eval, and builds auto‑interpretation outputs.

Artifacts are written under `artifacts/` in the repo path inside Colab. After the run, download:
- `artifacts/checkpoints/colab_msae/`
- `artifacts/eval/`
- `artifacts/autointerp/`

## Train SAE (local)

```bash
PYTHONPATH=src python scripts/train_sae.py \
  --cache_dir artifacts/activation_cache/<run> \
  --dict_size 4096 \
  --epochs 5 \
  --batch_size 256 \
  --lr 1e-3 \
  --l1_lambda 1e-3 \
  --device cpu
```

Outputs:
- `artifacts/checkpoints/<run>/last.pt`
- `artifacts/checkpoints/<run>/metrics.csv`
- `artifacts/checkpoints/<run>/config.json`

Notes:
- `metrics.csv` includes `evr_global`, computed as global explained variance ratio over the full cache.
- This matches the EVR/R2 definition used in the report (1 - SSE/SST on full activations set).

## SAE Variants

- `sae`: baseline sparse autoencoder (ReLU + L1).
- `msae`: Matryoshka Top-K SAE with multiple sparsity levels.

Example (MSAE):

```bash
PYTHONPATH=src python scripts/train_sae.py \
  --sae_type msae \
  --k_list 32,64,128,256 \
  --alpha_mode reverse \
  --input_centering dataset \
  --input_scaling dataset \
  --cache_dir artifacts/activation_cache/<run> \
  --dict_size 4096 \
  --epochs 5 \
  --batch_size 256 \
  --lr 1e-3 \
  --l1_lambda 1e-4 \
  --device cpu
```

## Quick Pipeline Check

```bash
make check-pipeline
```

## Zero-shot Eval (CIFAR-10/100/STL-10)

Baseline CLIP:

```bash
PYTHONPATH=src python scripts/eval_zeroshot.py \
  --dataset cifar10 \
  --split test \
  --model ViT-B-32 \
  --pretrained openai \
  --batch_size 64 \
  --num_workers 0
```

```bash
PYTHONPATH=src python scripts/eval_zeroshot.py \
  --dataset stl10 \
  --split test \
  --model ViT-B-32 \
  --pretrained openai \
  --batch_size 64 \
  --num_workers 0
```

If you are offline and STL‑10 is already downloaded, add `--no_download`.

With SAE intervention (requires checkpoint + layer):

```bash
PYTHONPATH=src python scripts/eval_zeroshot.py \
  --dataset cifar10 \
  --split test \
  --model ViT-B-32 \
  --pretrained openai \
  --batch_size 64 \
  --num_workers 0 \
  --sae_checkpoint artifacts/checkpoints/<run>/last.pt \
  --sae_layer visual.transformer.resblocks.0
```

## Zero-shot Table

```bash
PYTHONPATH=src python scripts/make_p4_table.py \
  --eval_runs artifacts/eval/<run1> artifacts/eval/<run2> \
  --checkpoint_dir artifacts/checkpoints/<sae_run> \
  --out_path artifacts/eval/zeroshot_eval_table.md
```

## Auto-Interpretation Table

Build collages from cached activations:

```bash
PYTHONPATH=src python scripts/build_collages.py \
  --cache_dir artifacts/activation_cache/<run> \
  --checkpoint artifacts/checkpoints/<sae_run>/last.pt \
  --num_latents 300 \
  --top_k 16 \
  --image_size 128 \
  --out_dir artifacts/autointerp/collages \
  --manifest_path artifacts/autointerp/collages_manifest.csv
```

Generate one-sentence interpretations (requires OpenRouter key):

```bash
export OPENROUTER_API_KEY=...
PYTHONPATH=src python scripts/autointerp.py \
  --manifest_path artifacts/autointerp/collages_manifest.csv \
  --out_csv artifacts/autointerp/autointerp.csv
```

Build the P5 markdown table:

```bash
PYTHONPATH=src python scripts/make_p5_table.py \
  --autointerp_csv artifacts/autointerp/autointerp.csv \
  --out_path artifacts/autointerp/autointerp_table.md
```

## Repo Layout

```
src/clip_sae/          # library code
scripts/               # CLI scripts
configs/               # YAML configs
notebooks/             # orchestration notebooks
logs/                  # training/eval logs
artifacts/             # outputs (checkpoints, tables, collages)
data/sample_images/    # local images for quick checks
report_assets/         # images embedded in REPORT.md
```
