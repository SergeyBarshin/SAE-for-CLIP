import os
from pathlib import Path
from typing import Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chunked(iterable: Iterable, n: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
