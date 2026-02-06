.PHONY: check

check:
	PYTHONPATH=src python scripts/check_repo.py
	PYTHONPATH=src python scripts/check_clip.py

.PHONY: extract-min

extract-min:
	PYTHONPATH=src python scripts/extract_activations.py \
	  --image_dir data/sample_images \
	  --model ViT-B-32 \
	  --pretrained openai \
	  --layer visual.transformer.resblocks.0 \
	  --num_samples 200 \
	  --batch_size 16 \
	  --shard_size 256 \
	  --out_dir artifacts/activation_cache

.PHONY: sample-images

sample-images:
	python scripts/populate_sample_images.py --num_images 200 --dataset cifar10 --split train

.PHONY: check-pipeline

check-pipeline:
	PYTHONPATH=src python scripts/check_pipeline.py

.PHONY: eval-cifar10 eval-cifar100

eval-cifar10:
	PYTHONPATH=src python scripts/eval_zeroshot.py --dataset cifar10 --split test --batch_size 64 --num_workers 0

eval-cifar100:
	PYTHONPATH=src python scripts/eval_zeroshot.py --dataset cifar100 --split test --batch_size 64 --num_workers 0
