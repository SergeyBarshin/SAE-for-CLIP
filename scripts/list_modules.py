import argparse
import importlib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        open_clip = importlib.import_module("open_clip")
    except Exception as exc:
        print("[list_modules] open_clip import: FAILED")
        print(f"[list_modules] NOTE: {exc}")
        return

    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    names = list(dict(model.named_modules()).keys())
    print("[list_modules] total modules:", len(names))
    for n in names:
        print(n)


if __name__ == "__main__":
    main()
