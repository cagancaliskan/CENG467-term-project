"""Batched generation for the trained student or any seq2seq HF model.

Supports both LoRA adapter directories (auto-detects adapter_config.json) and
plain checkpoints. Outputs predictions JSONL aligned with the input file.
"""
from __future__ import annotations

import re
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

# Strip mT5 SentencePiece sentinel tokens (e.g., <extra_id_0>) that leak
# through skip_special_tokens. Documented as a known failure mode in report §6.2.
_SENTINEL_RE = re.compile(r"<extra_id_\d+>")
from src.utils.seed import set_seed

LOG = get_logger("student.infer")


def _is_peft_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists()


def load_model(model_path: str, base_model: str = "google/mt5-small") -> tuple:
    p = Path(model_path)
    if _is_peft_dir(p):
        from peft import PeftModel
        LOG.info("Detected LoRA adapter dir; loading base=%s", base_model)
        tokenizer = AutoTokenizer.from_pretrained(p if (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists() else base_model)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, p)
    else:
        LOG.info("Loading full seq2seq model from %s", p)
        tokenizer = AutoTokenizer.from_pretrained(p)
        model = AutoModelForSeq2SeqLM.from_pretrained(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def main() -> None:
    p = argparse.ArgumentParser(description="Generate summaries with the student.")
    p.add_argument("--model-path", required=True,
                    help="Either a LoRA adapter dir (e.g. outputs/checkpoints/.../final) or a full HF model dir.")
    p.add_argument("--base-model", default="google/mt5-small",
                    help="Base model to attach the LoRA adapter to (ignored if --model-path is a full model).")
    p.add_argument("--input", required=True, help="JSONL with id + article fields.")
    p.add_argument("--out", required=True, help="Output predictions JSONL.")
    p.add_argument("--source-prefix", default="özetle: ")
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--min-target-length", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                    help="Cap number of articles (for quick smoke tests).")
    p.add_argument("--keep-sentinels", action="store_true",
                    help="Do NOT strip <extra_id_*> sentinels. Only for the artifact\n"
                          "analysis in the camera-ready revision; never for released output.")
    args = p.parse_args()

    set_seed(args.seed)

    tokenizer, model, device = load_model(args.model_path, args.base_model)

    rows = list(read_jsonl(args.input))
    if args.limit:
        rows = rows[: args.limit]
    LOG.info("Generating for %d articles", len(rows))

    out_records: list[dict] = []
    for i in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        batch = rows[i : i + args.batch_size]
        inputs = [args.source_prefix + (r.get("article") or "") for r in batch]
        enc = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=args.max_target_length,
                min_length=args.min_target_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        # Defensive: strip mT5 SentencePiece sentinel tokens that leak through
        # skip_special_tokens. See report §6.2.
        if not args.keep_sentinels:
            texts = [_SENTINEL_RE.sub("", t).strip() for t in texts]

        for row, pred in zip(batch, texts):
            out_records.append({
                "id": row["id"],
                "prediction": pred.strip(),
                "reference": row.get("reference"),
                "article": row.get("article"),
                "source": row.get("source"),
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out_records)
    LOG.info("Wrote %d predictions -> %s", len(out_records), args.out)

    # Drop a small generation-config sidecar.
    sidecar = Path(args.out).with_suffix(".gen_config.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
