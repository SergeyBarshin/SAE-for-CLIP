import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import List

import urllib.request

from clip_sae.logging import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_path", default="artifacts/autointerp/collages_manifest.csv"
    )
    p.add_argument("--out_csv", default="artifacts/autointerp/autointerp.csv")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--max_latents", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.5)
    return p.parse_args()


def load_manifest(path: str, max_latents: int | None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_latents is not None:
        rows = rows[:max_latents]
    return rows


def one_sentence(text: str) -> str:
    text = text.strip().replace("\n", " ")
    for sep in [".", "!", "?"]:
        if sep in text:
            first = text.split(sep)[0].strip()
            if first:
                return first + sep
    return text


def load_dotenv_key(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'")
        if k == key:
            return v
    return None


def call_openrouter(api_key: str, model: str, image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a visual feature interpreter. Respond with one sentence.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the common visual pattern across these images in one sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"OpenRouter HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    return one_sentence(text)


def main():
    args = parse_args()
    logger = get_logger("autointerp")

    manifest = load_manifest(args.manifest_path, args.max_latents)
    if not manifest:
        logger.error("Manifest is empty: %s", args.manifest_path)
        raise SystemExit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        repo_root = Path(__file__).resolve().parents[1]
        api_key = load_dotenv_key(repo_root / ".env", "OPENROUTER_API_KEY")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "latent_id",
                "collage_path",
                "max_activation",
                "interpretation",
                "status",
                "model",
            ],
        )
        writer.writeheader()

        for row in manifest:
            latent_id = row["latent_id"]
            collage_path = row["collage_path"]
            max_activation = row["max_activation"]

            if not api_key:
                writer.writerow(
                    {
                        "latent_id": latent_id,
                        "collage_path": collage_path,
                        "max_activation": max_activation,
                        "interpretation": "NO_API_KEY",
                        "status": "missing_key",
                        "model": args.model,
                    }
                )
                continue

            try:
                text = call_openrouter(api_key, args.model, collage_path)
                status = "ok"
            except Exception as exc:
                text = f"ERROR: {exc}"
                status = "error"

            writer.writerow(
                {
                    "latent_id": latent_id,
                    "collage_path": collage_path,
                    "max_activation": max_activation,
                    "interpretation": text,
                    "status": status,
                    "model": args.model,
                }
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    logger.info("Wrote autointerp CSV to %s", out_path)


if __name__ == "__main__":
    main()
