import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--autointerp_csv", required=True)
    p.add_argument("--latent_ids", default=None)
    p.add_argument("--num_rows", type=int, default=9)
    p.add_argument(
        "--out_path", default="artifacts/autointerp/autointerp_table.md"
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.autointerp_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.latent_ids:
        wanted = {int(x.strip()) for x in args.latent_ids.split(",") if x.strip()}
        rows = [r for r in rows if int(r["latent_id"]) in wanted]
    else:
        rows = rows[: args.num_rows]

    lines = []
    lines.append("| latent_id | collage | interpretation | status |")
    lines.append("|---:|:---:|---|---|")

    for r in rows:
        collage_md = f"![]({r['collage_path']})"
        lines.append(
            f"| {r['latent_id']} | {collage_md} | {r['interpretation']} | {r['status']} |"
        )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[make_p5_table] Wrote {out_path}")


if __name__ == "__main__":
    main()
