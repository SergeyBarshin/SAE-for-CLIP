import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_runs", nargs="+", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument(
        "--out_path", default="artifacts/eval/zeroshot_eval_table.md"
    )
    return p.parse_args()


def read_last_metrics(metrics_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)

    with open(ckpt_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    metrics = read_last_metrics(ckpt_dir / "metrics.csv")
    if metrics is None:
        raise SystemExit("metrics.csv is empty")

    eval_rows = []
    for run_dir in args.eval_runs:
        run_path = Path(run_dir)
        with open(run_path / "results.json", "r", encoding="utf-8") as f:
            eval_rows.append(json.load(f))

    lines = []
    lines.append("| dataset | baseline_acc | sae_acc | acc_delta | dict_size | l0 | evr_global | mse | l1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in eval_rows:
        baseline = row["baseline_acc"]
        sae = row["sae_acc"]
        delta = None if sae is None else (sae - baseline)
        lines.append(
            "| {dataset} | {baseline:.4f} | {sae_val} | {delta_val} | {dict_size} | {l0:.2f} | {evr:.4f} | {mse:.6f} | {l1:.6f} |".format(
                dataset=row["dataset"],
                baseline=baseline,
                sae_val=f"{sae:.4f}" if sae is not None else "-",
                delta_val=f"{delta:.4f}" if delta is not None else "-",
                dict_size=config.get("dict_size", "-"),
                l0=float(metrics.get("l0", 0.0)),
                evr=float(metrics.get("evr_global", 0.0)),
                mse=float(metrics.get("mse", 0.0)),
                l1=float(metrics.get("l1", 0.0)),
            )
        )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_p4_table] Wrote {out_path}")


if __name__ == "__main__":
    main()
