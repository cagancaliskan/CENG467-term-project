"""Merge teacher outputs into supervised train/val datasets for the student.

Reads:
    data/raw/mlsum_tr/train.jsonl     (article, reference)
    data/synthetic/<teacher>/<prompt>/<id>.json   (teacher summary per article)

Writes:
    data/processed/<exp_name>/train.jsonl
    data/processed/<exp_name>/validation.jsonl

Each output row: {"id", "article", "target", "source": "synthetic|human", ...}.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("data.prep")


def _load_teacher_dir(teacher_dir: Path) -> dict[str, str]:
    """Map article_id -> teacher summary string."""
    out: dict[str, str] = {}
    if not teacher_dir.exists():
        LOG.warning("Teacher dir does not exist: %s", teacher_dir)
        return out
    for p in teacher_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            aid = rec.get("id") or p.stem
            summary = (rec.get("summary") or "").strip()
            if summary:
                out[aid] = summary
        except Exception as e:
            LOG.warning("Skipping %s: %s", p, e)
    LOG.info("Loaded %d teacher summaries from %s", len(out), teacher_dir)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build student training data from teacher outputs.")
    p.add_argument("--mlsum-dir", default="data/raw/mlsum_tr",
                    help="Directory holding train.jsonl + validation.jsonl with reference summaries.")
    p.add_argument("--teacher", required=True, choices=["openai", "anthropic", "human"],
                    help="Source of supervision. 'human' uses MLSUM-TR reference summaries directly.")
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"],
                    help="Only used when --teacher is openai or anthropic.")
    p.add_argument("--size", type=int, required=True, help="Number of training pairs to keep.")
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--val-source", default="human", choices=["human", "teacher"],
                    help="Where validation targets come from. Default 'human' uses MLSUM-TR\n"
                          "reference summaries, which matches test-time evaluation; 'teacher'\n"
                          "uses the synthetic teacher cache (only valid if cache covers val rows).")
    p.add_argument("--out-dir", required=True,
                    help="Where to write train.jsonl and validation.jsonl.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    mlsum_dir = Path(args.mlsum_dir)
    train_rows = list(read_jsonl(mlsum_dir / "train.jsonl"))
    val_rows = list(read_jsonl(mlsum_dir / "validation.jsonl"))
    LOG.info("MLSUM-TR train=%d val=%d", len(train_rows), len(val_rows))

    if args.teacher == "human":
        def get_target(row: dict) -> str | None:
            return row.get("reference")
    else:
        teacher_dir = Path(f"data/synthetic/{args.teacher}/{args.prompt_variant}")
        teacher_map = _load_teacher_dir(teacher_dir)

        def get_target(row: dict) -> str | None:
            return teacher_map.get(row["id"])

    def build(rows: list[dict], target_n: int, split_name: str) -> list[dict]:
        rng.shuffle(rows)
        out: list[dict] = []
        for r in rows:
            tgt = get_target(r)
            if not tgt:
                continue
            out.append({
                "id": r["id"],
                "article": r["article"],
                "target": tgt,
                "split": split_name,
                "supervision": args.teacher,
                "prompt_variant": args.prompt_variant if args.teacher != "human" else None,
            })
            if len(out) >= target_n:
                break
        LOG.info("%s: kept %d / %d", split_name, len(out), target_n)
        return out

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = build(train_rows, args.size, "train")

    # Validation: by default use human references (val cache rarely populated).
    if args.val_source == "human":
        def get_target(row):
            return row.get("reference")
    val_out = build(val_rows, args.val_size, "validation")

    n_train = write_jsonl(out_dir / "train.jsonl", train_out)
    n_val = write_jsonl(out_dir / "validation.jsonl", val_out)
    LOG.info("Wrote train=%d val=%d -> %s", n_train, n_val, out_dir)


if __name__ == "__main__":
    main()
