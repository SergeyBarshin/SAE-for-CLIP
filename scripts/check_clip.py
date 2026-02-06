import argparse
import importlib
from PIL import Image
import torch


def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        return None
    return modules[name]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--layer", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        open_clip = importlib.import_module("open_clip")
    except Exception as exc:
        print("[check_clip] open_clip import: FAILED")
        print(f"[check_clip] NOTE: {exc}")
        print("[check_clip] This is expected before dependencies are installed.")
        return

    print("[check_clip] open_clip import: OK")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval().to(args.device)

    if args.layer:
        module = get_module_by_name(model, args.layer)
        if module is None:
            names = list(dict(model.named_modules()).keys())
            print("[check_clip] Layer not found. First 30 modules:")
            for n in names[:30]:
                print(" ", n)
            return

        act_buffer = {}

        def hook_fn(_, __, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            act_buffer["act"] = out

        hook = module.register_forward_hook(hook_fn)

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        batch = preprocess(img).unsqueeze(0).to(args.device)
        _ = model.encode_image(batch)
        acts = act_buffer.get("act")
        if acts is None:
            print("[check_clip] Hook failed to capture activations")
        else:
            print(f"[check_clip] Hook captured shape: {tuple(acts.shape)}")

        hook.remove()


if __name__ == "__main__":
    main()
