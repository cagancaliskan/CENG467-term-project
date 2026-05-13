"""LLM-as-a-judge qualitative labeling for the qual_labels_blank.csv export.

For each (article, system, prediction) row in the blank CSV, ask Claude Haiku
4.5 to score five binary axes following a fixed Turkish-aware rubric. Writes
the same CSV format expected by qual_aggregate.py.

Methodology: LLM-as-a-judge (Zheng et al., 2023, "Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena"). The rubric is the same one used for human
labeling so results are comparable. The judge is Anthropic Claude Haiku 4.5;
the prompt explicitly asks for a strict reading on factuality and a lenient
reading on morphology (where automatic judgment is weakest).

Usage:
    python -m src.eval.qual_llm_judge \
        --input outputs/results/qual_labels_blank.csv \
        --output outputs/results/qual_labels_filled.csv \
        --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.teachers.anthropic_teacher import AnthropicTeacher
from src.teachers.prompts import PromptSpec
from src.utils.logging import get_logger

LOG = get_logger("eval.qual_llm_judge")

JUDGE_SYSTEM = (
    "You are a careful bilingual evaluator of Turkish news summaries. Given a "
    "Turkish source article snippet, a reference summary, and a candidate "
    "summary produced by some system, you score the candidate on FIVE binary "
    "axes (1 = good, 0 = bad). Apply the rubric strictly and consistently:\n"
    "\n"
    "1. factual_correct (STRICT): score 1 only if every named entity, number, "
    "date, percentage, and quote in the candidate is present in the source "
    "article (paraphrasing OK). Even ONE fabricated number/date/entity → 0. "
    "Default when in doubt: 0.\n"
    "\n"
    "2. completeness (LENIENT): score 1 if the candidate captures the article's "
    "main event (who/what/where). Side details only → 0. Missing the lede → 0. "
    "Default when in doubt: 1.\n"
    "\n"
    "3. fluency: score 1 if the candidate reads as natural Turkish prose. "
    "Broken syntax, dropped words, English fragments, or mid-sentence cuts → 0. "
    "Mild stylistic awkwardness is OK. Default: 1.\n"
    "\n"
    "4. morpho_correct (LENIENT — give benefit of doubt): score 1 unless you "
    "spot a clear noun-case error (missing -i accusative, wrong -de/-da "
    "locative, wrong genitive -in) or a clear wrong verb tense or a malformed "
    "agglutinated word. Default: 1.\n"
    "\n"
    "5. no_mode_collapse: score 1 unless a 3+ word phrase repeats within the "
    "candidate, OR two near-identical sentences appear, OR filler phrases stack "
    "('aynı şekilde... aynı şekilde'). Single repeated entities (e.g., a name) "
    "are OK. Default: 1.\n"
    "\n"
    "Respond with STRICT JSON of the form:\n"
    "{\"factual_correct\": 0|1, \"completeness\": 0|1, \"fluency\": 0|1, "
    "\"morpho_correct\": 0|1, \"no_mode_collapse\": 0|1, \"notes\": \"...\"}\n"
    "\n"
    "The 'notes' field should be a brief English phrase (<15 words) flagging "
    "the most notable issue, or empty string if everything is fine. Examples: "
    "'fabricated year 2023', 'missed lede', 'repeats Erdoğan sentence', "
    "'wrong case on object'. Do not explain reasoning beyond the notes field."
)

JUDGE_USER_TEMPLATE = (
    "SOURCE ARTICLE (truncated):\n{article}\n\n"
    "HUMAN REFERENCE SUMMARY:\n{reference}\n\n"
    "CANDIDATE SUMMARY (system={system}):\n{prediction}\n\n"
    "Score the candidate on the five axes. Return only the JSON object."
)


def _build_prompt() -> PromptSpec:
    """Wrap the judge prompt in a PromptSpec for AnthropicTeacher."""
    return PromptSpec(
        name="qual_judge",
        system=JUDGE_SYSTEM,
        user_template=JUDGE_USER_TEMPLATE.replace("{article}", "{article}"),  # template marker preserved
        max_output_tokens=180,
    )


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judgment(raw: str) -> dict | None:
    """Extract the first JSON object from the judge's response."""
    if not raw:
        return None
    # Try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Fallback: find first {...} block
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _judge_one(teacher: AnthropicTeacher, prompt: PromptSpec, row: dict):
    """Call the judge for one row. Returns the parsed judgment dict (or None)."""
    user_msg = prompt.user_template.format(
        article=row.get("article_snippet", ""),
        reference=row.get("reference", ""),
        system=row.get("system", ""),
        prediction=row.get("prediction", ""),
    )
    # Bypass PromptSpec interpolation since we already built the user message.
    custom_prompt = PromptSpec(
        name=prompt.name,
        system=prompt.system,
        user_template=user_msg,
        max_output_tokens=prompt.max_output_tokens,
    )

    try:
        # AnthropicTeacher calls summarize(article, prompt). We pass empty article
        # because the user_template above already contains all the content with
        # no {article} placeholder.
        # But the prompt's user_template has .format(article=...) call inside
        # teacher.summarize. We need to avoid double-formatting.
        # Workaround: use a single-format-safe template (no remaining braces).
        resp = teacher._client.messages.create(
            model=teacher.model,
            temperature=0.0,  # deterministic for judging
            max_tokens=prompt.max_output_tokens,
            system=custom_prompt.system,
            messages=[{"role": "user", "content": custom_prompt.user_template}],
        )
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return _parse_judgment("".join(text_parts).strip())
    except Exception as e:
        LOG.warning("Failed to judge id=%s sys=%s: %s", row.get("id"), row.get("system"), e)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-as-a-judge qualitative labeling.")
    p.add_argument("--input", default="outputs/results/qual_labels_blank.csv")
    p.add_argument("--output", default="outputs/results/qual_labels_filled.csv")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    load_dotenv()

    # Load all blank rows
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    LOG.info("Loaded %d rows from %s", len(rows), args.input)

    teacher = AnthropicTeacher(model=args.model, temperature=0.0)
    prompt = _build_prompt()

    # Judge each row concurrently
    results: dict[int, dict | None] = {}
    lock = threading.Lock()
    start = time.time()

    def _work(idx: int, row: dict):
        j = _judge_one(teacher, prompt, row)
        with lock:
            results[idx] = j

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_work, i, row) for i, row in enumerate(rows)]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            pass

    elapsed = time.time() - start
    n_ok = sum(1 for v in results.values() if v is not None)
    LOG.info("Done in %.1fs | parsed %d / %d judgments", elapsed, n_ok, len(rows))

    # Write filled CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    AXES = ["factual_correct", "completeness", "fluency", "morpho_correct", "no_mode_collapse"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            j = results.get(i) or {}
            for axis in AXES:
                v = j.get(axis)
                row[axis] = "" if v is None else int(bool(v))
            row["notes"] = (j.get("notes") or "").strip()
            w.writerow(row)

    LOG.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
