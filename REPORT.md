# SAE for CLIP — отчет (локальный прогон)

Дата: 2026-02-07

## 1. Что такое CLIP

CLIP — это мультимодальная модель, которая учит совместное пространство эмбеддингов для изображений и текстов и обучается контрастивной задачей (соответствие пар изображение–текст). В zero‑shot она классифицирует изображение через сравнение эмбеддинга изображения с эмбеддингами текстовых промптов классов.

## 2. Прошлые работы (реимплементируемые)

- LessWrong: *Towards Multimodal Interpretability: Learning Sparse Autoencoders for CLIP* — обучение SAE на активациях CLIP и анализ признаков.
- LessWrong: *Interpreting and Steering Features in Images* — SAE для управления генерацией Kandinsky 2.2.

## 3. Обучение SAE (локально, малый корпус)

**Экстракция активаций**
- Модель: `open_clip` `ViT-B-32` (`openai`)
- Слой: `visual.transformer.resblocks.0`
- Данные: `data/sample_images` (CIFAR-10, 200 картинок)
- Кеш: `artifacts/activation_cache/20260206_213525`

**Гиперпараметры обучения SAE**
- `dict_size`: 2048
- `epochs`: 10
- `batch_size`: 256
- `lr`: 1e-3
- `l1_lambda`: 1e-3
- `device`: cpu
- Чекпоинт: `artifacts/checkpoints/20260206_222126`

**Метрики (последняя эпоха, SAE)**
- `mse`: 0.006117
- `l1`: 0.091471
- `l0`: 554.09
- `r2`: -38.9255
- `evr_global`: -34.7332

Примечание: локальный прогон маленький, EVR отрицательный. Для адекватных метрик требуется больше данных и вычислений (планируется в Colab).

## 4. Zero‑shot eval (CIFAR‑10 + STL‑10)

Таблица из `artifacts/eval/zeroshot_eval_table.md`:

| dataset | baseline_acc | sae_acc | acc_delta | dict_size | l0 | evr_global | mse | l1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 0.8545 | 0.8535 | -0.0010 | 512 | 32.00 | 0.3222 | 0.857268 | 0.097542 |
| stl10 | 0.9530 | 0.9530 | 0.0000 | 512 | 32.00 | 0.3222 | 0.857268 | 0.097542 |

## 5. Авто‑интерпретация (п.5) (п.5)

**Коллажи и CSV**
- Коллажи: `artifacts/autointerp/collages/latent_*.png`
- Манифест: `artifacts/autointerp/collages_manifest.csv`
- CSV: `artifacts/autointerp/autointerp.csv`

**Таблица auto‑interpretation**

Таблица из `artifacts/autointerp/autointerp_table.md`.

| latent_id | collage | interpretation | status |
|---:|:---:|---|---|
| 13 | ![](artifacts/autointerp/collages/latent_13.png) | The images predominantly feature various animals, including mammals and birds, in natural or semi-natural settings. | ok |
| 27 | ![](artifacts/autointerp/collages/latent_27.png) | The images feature a diverse array of animals, vehicles, and scenes, showcasing both wildlife and domesticated creatures alongside cars in various settings. | ok |
| 39 | ![](artifacts/autointerp/collages/latent_39.png) | The common visual pattern across these images features a variety of animals and plants in natural settings. | ok |
| 45 | ![](artifacts/autointerp/collages/latent_45.png) | The common visual pattern across these images features various animals in natural settings, showcasing a diversity of species and habitats. | ok |
| 51 | ![](artifacts/autointerp/collages/latent_51.png) | The common visual pattern across these images is the depiction of various animals in different environments and poses. | ok |
| 157 | ![](artifacts/autointerp/collages/latent_157.png) | The images depict a variety of animals, including mammals, birds, and amphibians, often in natural or outdoor settings. | ok |
| 201 | ![](artifacts/autointerp/collages/latent_201.png) | The common visual pattern across these images is the presence of various animals in diverse natural and semi-natural environments. | ok |
| 216 | ![](artifacts/autointerp/collages/latent_216.png) | The images depict a variety of animals, including mammals, amphibians, and birds, showcasing their diverse forms and habitats. | ok |
| 220 | ![](artifacts/autointerp/collages/latent_220.png) | The images depict a variety of animals, including mammals, birds, and amphibians, showcasing diverse species in natural settings. | ok |

## 6. MSAE (Matryoshka Top‑K SAE)

**Что добавлено**
- Архитектура Matryoshka Top‑K SAE (MSAE) с несколькими уровнями разреженности.
- Нормировка входа (dataset mean/std) и EVR в нормированном пространстве.

**Dev‑прогон (локально, Food101 sample)**
- Чекпоинт: `artifacts/checkpoints/food101_dev_msae`
- `dict_size`: 512
- `k_list`: [16, 32]
- `evr_global`: 0.3222 (положительный)

**Пример команды**

```bash
PYTHONPATH=src python scripts/train_sae.py \
  --sae_type msae \
  --k_list 16,32 \
  --cache_dir artifacts/activation_cache/food101_dev_acts \
  --dict_size 512 \
  --epochs 1 \
  --batch_size 64 \
  --lr 1e-3 \
  --l1_lambda 1e-4 \
  --device cpu \
  --run_name food101_dev_msae
```

## 7. Что осталось

Локально:
- Подготовка сборочного ноутбука `notebooks/assemble_local.ipynb`.

Colab (основной прогон):
- Обучение SAE/MSAE на Food101 (50k–100k изображений).
- Метрики EVR > 0.8.
- Auto‑interpretation ≥ 300 латентов + финальная таблица.

Фаза 3:
- Экспорт SAE в HuggingFace Hub.
- Steering Kandinsky 2.2 + итоговые картинки.

## Команды запуска (локально)

```bash
# 1) Подготовка данных
make sample-images

# 2) Экстракт активаций
PYTHONPATH=src python scripts/extract_activations.py \
  --image_dir data/sample_images \
  --model ViT-B-32 \
  --pretrained openai \
  --layer visual.transformer.resblocks.0 \
  --num_samples 200 \
  --batch_size 16 \
  --shard_size 256 \
  --out_dir artifacts/activation_cache

# 3) Обучение SAE
PYTHONPATH=src python scripts/train_sae.py \
  --cache_dir artifacts/activation_cache/20260206_213525 \
  --dict_size 2048 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --l1_lambda 1e-3 \
  --device cpu

# 4) Zero-shot eval
PYTHONPATH=src python scripts/eval_zeroshot.py \
  --dataset cifar10 --split test --batch_size 64 --num_workers 0

PYTHONPATH=src python scripts/eval_zeroshot.py \
  --dataset stl10 --split test --batch_size 64 --num_workers 0

# 5) Auto-interpretation (коллажи + CSV)
PYTHONPATH=src python scripts/build_collages.py \
  --cache_dir artifacts/activation_cache/20260206_213525 \
  --checkpoint artifacts/checkpoints/20260206_222126/last.pt \
  --num_latents 50 --top_k 16 --image_size 128 \
  --out_dir artifacts/autointerp/collages \
  --manifest_path artifacts/autointerp/collages_manifest.csv

PYTHONPATH=src python scripts/autointerp.py \
  --manifest_path artifacts/autointerp/collages_manifest.csv \
  --out_csv artifacts/autointerp/autointerp.csv

PYTHONPATH=src python scripts/make_p5_table.py \
  --autointerp_csv artifacts/autointerp/autointerp.csv \
  --out_path artifacts/autointerp/autointerp_table.md
```
