from pathlib import Path


def ensure_dirs(dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    required_dirs = [
        repo_root / "src/clip_sae",
        repo_root / "scripts",
        repo_root / "configs",
        repo_root / "notebooks",
        repo_root / "logs",
        repo_root / "artifacts",
        repo_root / "data/sample_images",
        repo_root / "data/torchvision",
    ]

    ensure_dirs(required_dirs)

    print("[check_repo] OK: required directories exist.")
    sample_dir = repo_root / "data/sample_images"
    if not any(sample_dir.iterdir()):
        print("[check_repo] NOTE: data/sample_images is empty.")


if __name__ == "__main__":
    main()
