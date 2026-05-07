"""Generate teacher predictions on a held-out test set, with caching.

This is just src.teachers.generate retargeted at a test JSONL. It writes a
predictions file in the same shape as src.student.infer, so the evaluator can
treat the teacher as another system.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.teachers.anthropic_teacher import AnthropicTeacher
from src.teachers.openai_teacher import OpenAITeacher
from src.teachers.prompts import get_prompt
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("student.infer_teacher")


def main() -> None:
    p = argparse.ArgumentParser(description="Run teacher LLM as a system on a test set.")
    p.add_argument("--teacher", required=True, choices=["openai", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None,
                    help="If set, cache per-article JSON files here (default: outputs/predictions/_cache/<teacher>/<prompt>).")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-article-chars", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    load_dotenv()
    set_seed(args.seed)

    prompt = get_prompt(args.prompt_variant)
    teacher = (
        OpenAITeacher(model=args.model or "gpt-4o-mini", temperature=args.temperature)
        if args.teacher == "openai"
        else AnthropicTeacher(model=args.model or "claude-3-haiku-20240307", temperature=args.temperature)
    )

    cache_dir = Path(args.cache_dir or f"outputs/predictions/_cache/{args.teacher}/{prompt.name}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.input))
    if args.limit:
        rows = rows[: args.limit]

    out_records: list[dict] = []
    for row in tqdm(rows, desc=f"{args.teacher}/{prompt.name}"):
        aid = row["id"]
        cache_path = cache_dir / f"{aid}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            pred = cached["summary"]
        else:
            article = (row.get("article") or "")[: args.max_article_chars]
            try:
                resp = teacher.summarize(article, prompt)
            except Exception as e:
                LOG.warning("Failed id=%s: %s", aid, e)
                continue
            pred = resp.summary
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "id": aid,
                    "summary": pred,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "teacher": args.teacher,
                    "prompt_variant": prompt.name,
                }, f, ensure_ascii=False, indent=2)

        out_records.append({
            "id": aid,
            "prediction": pred,
            "reference": row.get("reference"),
            "article": row.get("article"),
            "source": row.get("source"),
        })

    write_jsonl(args.out, out_records)
    LOG.info("Wrote %d predictions -> %s", len(out_records), args.out)


if __name__ == "__main__":
    main()
