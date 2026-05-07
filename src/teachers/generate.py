"""CLI for generating teacher summaries with on-disk caching.

Usage:
    python -m src.teachers.generate --teacher openai --prompt-variant concise \
        --input data/raw/mlsum_tr/train.jsonl --n 10000 \
        --out-dir data/synthetic/openai/concise

Cache scheme:
    data/synthetic/<teacher>/<prompt>/<article_id>.json
    plus a sidecar manifest.jsonl with one record per generation.

Re-runs are zero-cost as long as the cache files are intact.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.teachers.anthropic_teacher import AnthropicTeacher
from src.teachers.base import BaseTeacher
from src.teachers.openai_teacher import OpenAITeacher
from src.teachers.prompts import get_prompt
from src.utils.io import dump_run_config, read_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("teachers.generate")


def build_teacher(name: str, model: str | None, temperature: float) -> BaseTeacher:
    if name == "openai":
        return OpenAITeacher(model=model or "gpt-4o-mini", temperature=temperature)
    if name == "anthropic":
        return AnthropicTeacher(model=model or "claude-haiku-4-5-20251001", temperature=temperature)
    raise ValueError(f"Unknown teacher: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate teacher summaries with caching.")
    p.add_argument("--teacher", required=True, choices=["openai", "anthropic"])
    p.add_argument("--model", default=None,
                    help="Override model name (e.g. gpt-4o, claude-sonnet-4-6).")
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"])
    p.add_argument("--input", required=True, help="JSONL with id + article fields.")
    p.add_argument("--n", type=int, default=10000, help="Maximum number of articles to summarize.")
    p.add_argument("--out-dir", required=True, help="Per-article cache directory.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-article-chars", type=int, default=3000,
                    help="Truncate very long articles to keep API costs predictable.")
    p.add_argument("--sleep-between", type=float, default=0.0,
                    help="Seconds to sleep between calls (manual rate limiting).")
    args = p.parse_args()

    load_dotenv()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    prompt = get_prompt(args.prompt_variant)
    teacher = build_teacher(args.teacher, args.model, args.temperature)

    dump_run_config(out_dir, {
        "teacher": args.teacher,
        "model": teacher.model,
        "prompt_variant": prompt.name,
        "temperature": args.temperature,
        "input": args.input,
        "n_requested": args.n,
        "seed": args.seed,
        "max_article_chars": args.max_article_chars,
    })

    rows = list(read_jsonl(args.input))
    rows = rows[: args.n]
    LOG.info("Generating up to %d summaries with %s/%s", len(rows), args.teacher, prompt.name)

    n_cached = 0
    n_called = 0
    n_failed = 0
    total_in = 0
    total_out = 0
    start = time.time()

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for row in tqdm(rows, desc=f"{args.teacher}/{prompt.name}"):
            aid = row["id"]
            cache_path = out_dir / f"{aid}.json"
            if cache_path.exists():
                n_cached += 1
                continue

            article = (row.get("article") or "")[: args.max_article_chars]
            try:
                resp = teacher.summarize(article, prompt)
            except Exception as e:
                LOG.warning("Failed id=%s: %s", aid, e)
                n_failed += 1
                continue

            rec = {
                "id": aid,
                "summary": resp.summary,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "teacher": args.teacher,
                "model": teacher.model,
                "prompt_variant": prompt.name,
                "temperature": args.temperature,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.flush()

            n_called += 1
            total_in += resp.input_tokens
            total_out += resp.output_tokens
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    elapsed = time.time() - start
    LOG.info(
        "Done in %.1fs | cached=%d called=%d failed=%d | tokens in=%d out=%d",
        elapsed, n_cached, n_called, n_failed, total_in, total_out,
    )


if __name__ == "__main__":
    main()
