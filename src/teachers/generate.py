"""CLI for generating teacher summaries with on-disk caching and concurrency.

Usage:
    python -m src.teachers.generate --teacher openai --prompt-variant concise \
        --input data/raw/mlsum_tr/train.jsonl --n 10000 \
        --out-dir data/synthetic/openai/concise --workers 8

Cache scheme: data/synthetic/<teacher>/<prompt>/<article_id>.json plus
manifest.jsonl. Re-runs are zero-cost as long as cache files are intact.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        return AnthropicTeacher(model=model or "claude-3-haiku-20240307", temperature=temperature)
    raise ValueError(f"Unknown teacher: {name!r}")


def _process_one(teacher, prompt, row, max_chars, out_dir):
    """Worker function — pure of any shared state; safe to run on a thread pool."""
    aid = row["id"]
    cache_path = out_dir / f"{aid}.json"
    if cache_path.exists():
        return ("cached", aid, None)
    article = (row.get("article") or "")[:max_chars]
    try:
        resp = teacher.summarize(article, prompt)
    except Exception as e:
        return ("failed", aid, str(e))
    rec = {
        "id": aid,
        "summary": resp.summary,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "teacher": teacher.name,
        "model": teacher.model,
        "prompt_variant": prompt.name,
        "temperature": teacher.temperature,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return ("called", aid, rec)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate teacher summaries with caching and concurrency.")
    p.add_argument("--teacher", required=True, choices=["openai", "anthropic"])
    p.add_argument("--model", default=None,
                   help="Override model (e.g. gpt-4o, claude-3-5-haiku-20241022).")
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"])
    p.add_argument("--input", required=True, help="JSONL with id + article fields.")
    p.add_argument("--n", type=int, default=10000, help="Maximum articles to process.")
    p.add_argument("--out-dir", required=True, help="Per-article cache directory.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-article-chars", type=int, default=3000)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent API workers. 8 is safe for OpenAI tier 1; "
                        "drop to 4 if you see 429 rate-limit warnings.")
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
        "workers": args.workers,
    })

    rows = list(read_jsonl(args.input))[: args.n]
    LOG.info("Generating up to %d summaries with %s/%s using %d workers",
              len(rows), args.teacher, prompt.name, args.workers)

    n_cached = n_called = n_failed = 0
    total_in = total_out = 0
    manifest_lock = threading.Lock()
    start = time.time()

    with open(manifest_path, "a", encoding="utf-8") as manifest, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, teacher, prompt, row,
                              args.max_article_chars, out_dir): row["id"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures),
                          desc=f"{args.teacher}/{prompt.name}"):
            kind, aid, payload = fut.result()
            if kind == "cached":
                n_cached += 1
            elif kind == "called":
                n_called += 1
                total_in += payload["input_tokens"]
                total_out += payload["output_tokens"]
                with manifest_lock:
                    manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    manifest.flush()
            else:
                n_failed += 1
                LOG.warning("Failed id=%s: %s", aid, payload)

    elapsed = time.time() - start
    LOG.info("Done in %.1fs | cached=%d called=%d failed=%d | tokens in=%d out=%d",
              elapsed, n_cached, n_called, n_failed, total_in, total_out)


if __name__ == "__main__":
    main()
