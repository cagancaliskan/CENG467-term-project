"""Generate teacher predictions on a held-out test set with caching + concurrency.

Output format mirrors src.student.infer (predictions JSONL with id/prediction/
reference/article), so src.eval.run_eval can treat the teacher as another system.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _process_one(teacher, prompt, row, max_chars, cache_dir):
    aid = row["id"]
    cp = cache_dir / f"{aid}.json"
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            return ("cached", row, json.load(f).get("summary", ""))
    article = (row.get("article") or "")[:max_chars]
    try:
        resp = teacher.summarize(article, prompt)
    except Exception as e:
        return ("failed", row, str(e))
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
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return ("called", row, resp.summary)


def main() -> None:
    p = argparse.ArgumentParser(description="Run teacher LLM as a system on a test set, with caching + concurrency.")
    p.add_argument("--teacher", required=True, choices=["openai", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None,
                    help="If set, cache per-article JSON files here. Default: outputs/predictions/_cache/<teacher>/<prompt>")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-article-chars", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8,
                    help="Concurrent API workers.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    load_dotenv()
    set_seed(args.seed)
    prompt = get_prompt(args.prompt_variant)

    if args.teacher == "openai":
        teacher = OpenAITeacher(model=args.model or "gpt-4o-mini", temperature=args.temperature)
    else:
        teacher = AnthropicTeacher(model=args.model or "claude-haiku-4-5-20251001", temperature=args.temperature)

    cache_dir = Path(args.cache_dir or f"outputs/predictions/_cache/{args.teacher}/{prompt.name}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.input))
    if args.limit:
        rows = rows[: args.limit]
    LOG.info("Generating predictions for %d articles with %s/%s workers=%d",
              len(rows), args.teacher, prompt.name, args.workers)

    n_cached = n_called = n_failed = 0
    by_id: dict[str, tuple[dict, str] | None] = {r["id"]: None for r in rows}
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, teacher, prompt, row,
                              args.max_article_chars, cache_dir): row["id"] for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures),
                          desc=f"{args.teacher}/{prompt.name}"):
            kind, row, payload = fut.result()
            if kind == "failed":
                n_failed += 1
                LOG.warning("Failed id=%s: %s", row["id"], payload)
                continue
            if kind == "cached":
                n_cached += 1
            else:
                n_called += 1
            by_id[row["id"]] = (row, payload)

    # Re-serialize in original input order so the predictions JSONL is aligned to test.jsonl.
    out_records: list[dict] = []
    for row in rows:
        slot = by_id.get(row["id"])
        if slot is None:
            continue
        _, summary = slot
        out_records.append({
            "id": row["id"],
            "prediction": (summary or "").strip(),
            "reference": row.get("reference"),
            "article": row.get("article"),
            "source": row.get("source"),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out_records)
    elapsed = time.time() - start
    LOG.info("Done in %.1fs | cached=%d called=%d failed=%d | wrote %d -> %s",
              elapsed, n_cached, n_called, n_failed, len(out_records), args.out)


if __name__ == "__main__":
    main()
