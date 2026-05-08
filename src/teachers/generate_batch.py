"""Submit a teacher generation job via Anthropic's Batch API.

50% cheaper than real-time API, typically completes in 10-60 minutes for
batches up to 10k requests. Results are written to the same per-article JSON
cache files as src.teachers.generate, so downstream code (prepare_synthetic,
infer, eval) doesn't need to know which mode produced them.

Usage:
    python -m src.teachers.generate_batch --teacher anthropic \
        --prompt-variant concise --input data/raw/mlsum_tr/train.jsonl \
        --n 10000 --out-dir data/synthetic/anthropic/concise \
        --model claude-haiku-4-5-20251001

Resume:
    If you Ctrl-C during polling, the batch keeps running on Anthropic's side.
    Re-run with --resume to pick up the same batch by its saved id.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.teachers.prompts import get_prompt
from src.utils.io import dump_run_config, read_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("teachers.generate_batch")

POLL_INTERVAL_S = 30


def _build_anthropic_requests(rows, prompt, model, temperature, max_chars, out_dir):
    """Build batch request dicts for articles not already cached."""
    needed = []
    n_skipped = 0
    seen = set()
    for row in rows:
        aid = row["id"]
        if (out_dir / f"{aid}.json").exists():
            n_skipped += 1
            continue
        if aid in seen:
            continue
        seen.add(aid)
        article = (row.get("article") or "")[:max_chars]
        needed.append({
            "custom_id": aid,
            "params": {
                "model": model,
                "max_tokens": prompt.max_output_tokens,
                "system": prompt.system,
                "messages": [
                    {"role": "user", "content": prompt.user_template.format(article=article)}
                ],
                "temperature": temperature,
            },
        })
    return needed, n_skipped


def _wait_anthropic_batch(client, batch_id):
    """Poll until processing_status == 'ended'. Logs progress every ~30s."""
    last_log = 0.0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        rc = batch.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        done = rc.succeeded + rc.errored + rc.canceled + rc.expired
        now = time.time()
        if now - last_log >= 25:
            LOG.info("batch %s | status=%s | done=%d/%d (succeeded=%d, errored=%d, processing=%d)",
                     batch_id, batch.processing_status, done, total,
                     rc.succeeded, rc.errored, rc.processing)
            last_log = now
        if batch.processing_status == "ended":
            return batch
        time.sleep(POLL_INTERVAL_S)


def _fetch_anthropic_results(client, batch_id, out_dir, prompt_name, model, temperature):
    """Stream results, write succeeded ones as per-article cache JSON."""
    n_succ = n_err = 0
    t_in = t_out = 0
    for r in client.messages.batches.results(batch_id):
        aid = r.custom_id
        if r.result.type == "succeeded":
            msg = r.result.message
            text_parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            summary = "".join(text_parts).strip()
            usage = msg.usage
            in_t = getattr(usage, "input_tokens", 0) or 0
            out_t = getattr(usage, "output_tokens", 0) or 0
            rec = {
                "id": aid,
                "summary": summary,
                "input_tokens": in_t,
                "output_tokens": out_t,
                "teacher": "anthropic",
                "model": model,
                "prompt_variant": prompt_name,
                "temperature": temperature,
                "via_batch": True,
            }
            with open(out_dir / f"{aid}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            n_succ += 1
            t_in += in_t
            t_out += out_t
        else:
            err = getattr(r.result, "error", None)
            LOG.warning("errored id=%s err=%s", aid, err)
            n_err += 1
    return n_succ, n_err, t_in, t_out


def main() -> None:
    p = argparse.ArgumentParser(description="Submit a teacher generation batch (Anthropic Batch API).")
    p.add_argument("--teacher", required=True, choices=["anthropic"],
                   help="Only Anthropic is supported in this script. OpenAI Batch API uses a different (file-based) flow; for OpenAI, use src.teachers.generate.")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--prompt-variant", default="concise", choices=["concise", "detailed"])
    p.add_argument("--input", required=True)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-article-chars", type=int, default=3000)
    p.add_argument("--resume", action="store_true",
                   help="Use the batch_id saved in _batch_id.txt instead of submitting a new batch.")
    args = p.parse_args()

    load_dotenv()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_id_path = out_dir / "_batch_id.txt"

    prompt = get_prompt(args.prompt_variant)

    from anthropic import Anthropic
    client = Anthropic()

    dump_run_config(out_dir, {
        "teacher": args.teacher,
        "model": args.model,
        "prompt_variant": prompt.name,
        "temperature": args.temperature,
        "input": args.input,
        "n_requested": args.n,
        "seed": args.seed,
        "max_article_chars": args.max_article_chars,
        "via_batch": True,
    })

    rows = list(read_jsonl(args.input))[: args.n]
    LOG.info("Loaded %d rows from %s", len(rows), args.input)

    if args.resume and batch_id_path.exists():
        batch_id = batch_id_path.read_text().strip()
        LOG.info("Resuming existing batch_id=%s", batch_id)
    else:
        requests, n_skipped = _build_anthropic_requests(
            rows, prompt, args.model, args.temperature, args.max_article_chars, out_dir
        )
        LOG.info("To generate: %d articles (already cached, skipping: %d)", len(requests), n_skipped)
        if not requests:
            LOG.info("Nothing to do — all articles already cached.")
            return

        # Quick cost estimate before submission.
        sample_input_chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests[:50])
        est_input_tokens_per_call = (sample_input_chars / max(1, min(50, len(requests)))) / 3.5
        est_output_tokens_per_call = prompt.max_output_tokens * 0.6
        # Haiku 4.5 batch pricing: $0.50 / $2.50 per M (50% off real-time).
        est_cost = len(requests) * (
            est_input_tokens_per_call * 0.50 / 1_000_000
            + est_output_tokens_per_call * 2.50 / 1_000_000
        )
        LOG.info("Estimated batch cost: ~$%.2f at Haiku 4.5 batch pricing ($0.50/$2.50 per M)", est_cost)

        LOG.info("Submitting batch with %d requests ...", len(requests))
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        batch_id_path.write_text(batch_id)
        LOG.info("Submitted batch_id=%s status=%s", batch_id, batch.processing_status)
        LOG.info("Saved batch_id to %s — re-run with --resume if interrupted.", batch_id_path)

    LOG.info("Polling every %ds (Ctrl-C is safe; use --resume later) ...", POLL_INTERVAL_S)
    final = _wait_anthropic_batch(client, batch_id)
    LOG.info("Batch ended | counts=%s", final.request_counts)

    LOG.info("Fetching results and writing per-article cache ...")
    n_succ, n_err, t_in, t_out = _fetch_anthropic_results(
        client, batch_id, out_dir, prompt.name, args.model, args.temperature
    )
    LOG.info("Done | succeeded=%d errored=%d | tokens in=%d out=%d (actual cost ~$%.2f)",
             n_succ, n_err, t_in, t_out,
             t_in * 0.50 / 1_000_000 + t_out * 2.50 / 1_000_000)

    if batch_id_path.exists():
        batch_id_path.unlink()


if __name__ == "__main__":
    main()
