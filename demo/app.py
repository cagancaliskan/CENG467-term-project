"""Gradio demo: side-by-side summaries from 4 student systems plus 2 LLM teachers.

Runs locally on T4 (with checkpoints in outputs/checkpoints/) or in a
HuggingFace Space (with the four checkpoints uploaded). Teachers are called via
API only if the corresponding env var is set; otherwise their cells are empty.

Usage (local / Colab):
    python -m demo.app
Or (HF Spaces): set this as the entry point and provide OPENAI_API_KEY /
ANTHROPIC_API_KEY as Space secrets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is importable when this file is launched directly
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.teachers.prompts import get_prompt

# ---- system registry ----
CKPT_ROOT = ROOT / "outputs" / "checkpoints"

STUDENT_SYSTEMS = {
    "B1 (mT5-small zero-shot)": None,  # no adapter
    "B2 (human-supervised)": CKPT_ROOT / "B2_human_n10000_r8" / "final",
    "S-gpt (synthetic, GPT-4o-mini)": CKPT_ROOT / "S_gpt_n10000_r8" / "final",
    "S-claude (synthetic, Claude Haiku 4.5)": CKPT_ROOT / "S_claude_n10000_r8" / "final",
}

BASE_MODEL = "google/mt5-small"
SOURCE_PREFIX = "özetle: "
MAX_INPUT = 512
MAX_OUTPUT = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load all student models once at startup ----
print(f"[demo] Loading base model on {DEVICE} ...")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
BASE = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
BASE.eval()

LOADED_STUDENTS: dict[str, torch.nn.Module] = {}
for label, adapter_path in STUDENT_SYSTEMS.items():
    if adapter_path is None:
        LOADED_STUDENTS[label] = BASE
        print(f"[demo] {label}: base (no adapter)")
    elif not adapter_path.exists():
        print(f"[demo] {label}: ADAPTER NOT FOUND at {adapter_path} — will skip")
        LOADED_STUDENTS[label] = None
    else:
        # Each PeftModel wraps a fresh copy of the base, otherwise adapters would stack.
        b = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
        m = PeftModel.from_pretrained(b, adapter_path).to(DEVICE)
        m.eval()
        LOADED_STUDENTS[label] = m
        print(f"[demo] {label}: loaded LoRA adapter from {adapter_path}")


def _student_summarize(model, article: str) -> str:
    if model is None:
        return "(adapter not available locally)"
    inputs = TOKENIZER(SOURCE_PREFIX + article, max_length=MAX_INPUT, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            min_length=16,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
    return TOKENIZER.decode(out[0], skip_special_tokens=True).strip()


# ---- teacher LLM clients (optional, only if API keys present) ----
def _maybe_openai():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from src.teachers.openai_teacher import OpenAITeacher
        return OpenAITeacher()
    except Exception as e:
        print(f"[demo] OpenAI teacher init failed: {e}")
        return None


def _maybe_anthropic():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from src.teachers.anthropic_teacher import AnthropicTeacher
        return AnthropicTeacher()
    except Exception as e:
        print(f"[demo] Anthropic teacher init failed: {e}")
        return None


OPENAI = _maybe_openai()
ANTHROPIC = _maybe_anthropic()
PROMPT = get_prompt("concise")


def _teacher_summarize(teacher, article: str) -> str:
    if teacher is None:
        return "(API key not provided — set OPENAI_API_KEY / ANTHROPIC_API_KEY)"
    article_truncated = article[:3000]
    try:
        return teacher.summarize(article_truncated, PROMPT).summary
    except Exception as e:
        return f"(error: {e})"


# ---- Gradio handler ----
def summarize_all(article: str):
    article = (article or "").strip()
    if not article:
        return [""] * 6

    out_b1 = _student_summarize(LOADED_STUDENTS["B1 (mT5-small zero-shot)"], article)
    out_b2 = _student_summarize(LOADED_STUDENTS["B2 (human-supervised)"], article)
    out_sgpt = _student_summarize(LOADED_STUDENTS["S-gpt (synthetic, GPT-4o-mini)"], article)
    out_sclaude = _student_summarize(LOADED_STUDENTS["S-claude (synthetic, Claude Haiku 4.5)"], article)
    out_b3a = _teacher_summarize(OPENAI, article)
    out_b3b = _teacher_summarize(ANTHROPIC, article)

    return out_b1, out_b2, out_sgpt, out_sclaude, out_b3a, out_b3b


EXAMPLE_ARTICLES = [
    [(
        "TESK Başkanı Bendevi Palandöken, kredi faizlerinin düşürülmesinin esnaf ve sanatkara nefes "
        "aldıracağını ifade etti. TESK Başkanı Palandöken, Halkbank aracılığı ile kullandırılan kredi "
        "faizlerinin düşürülmesiyle ilgili yazılı bir açıklama yaptı. Piyasaların daralması nedeniyle "
        "paranın dönmediğini, bu durumun kredi ödemelerine olumsuz yansıdığını belirten TESK Genel "
        "Başkanı Bendevi Palandöken, 'Ekonomide büyümenin yavaşlaması, işsizliğin artması piyasaları "
        "darlattı. Alışverişin miktarı azaldı. Haliyle para dönmeyince kredi geri ödemeleri de "
        "aksamaya başladı. Halkbank aracılığı ile esnafımızın kullandığı kredi faizlerinin yüzde 5.5'ler "
        "seviyesine indirilmesi esnaf ve sanatkarımıza nefes aldıracaktır.' dedi."
    )],
]

with gr.Blocks(title="Turkish News Summarizer — Distilled vs Teacher") as demo:
    gr.Markdown(
        "# Türkçe Haber Özetleyici / Turkish News Summarizer\n\n"
        "Compare four small distilled students against two large LLM teachers on Turkish news articles. "
        "All four students share the same 300M-parameter mT5-small backbone with LoRA adapters; only "
        "the supervision differs. Teachers are called via API and only appear if API keys are set "
        "as environment variables."
    )

    inp = gr.Textbox(label="Turkish news article", lines=10,
                     placeholder="Paste a Turkish news article here ...")
    btn = gr.Button("Özetle / Summarize", variant="primary")

    gr.Markdown("## Small students (mT5-small + LoRA)")
    with gr.Row():
        out_b1 = gr.Textbox(label="B1 — Zero-shot mT5-small", lines=4)
        out_b2 = gr.Textbox(label="B2 — Human-supervised", lines=4)
    with gr.Row():
        out_sgpt = gr.Textbox(label="S-gpt — Distilled from GPT-4o-mini", lines=4)
        out_sclaude = gr.Textbox(label="S-claude — Distilled from Claude Haiku 4.5", lines=4)

    gr.Markdown("## Teacher LLMs (zero-shot, via API)")
    with gr.Row():
        out_b3a = gr.Textbox(label="B3a — GPT-4o-mini", lines=4)
        out_b3b = gr.Textbox(label="B3b — Claude Haiku 4.5", lines=4)

    btn.click(summarize_all, inp, [out_b1, out_b2, out_sgpt, out_sclaude, out_b3a, out_b3b])

    gr.Examples(examples=EXAMPLE_ARTICLES, inputs=inp)

    gr.Markdown(
        "**Notes.** B1 (zero-shot) is a sanity baseline and tends to produce short noise. "
        "B2 was trained on human reference summaries from MLSUM-TR. S-gpt and S-claude were "
        "trained on 10,000 synthetic summaries from the respective teacher. All four students "
        "use the same hyperparameters (LoRA rank 8 on q/v projections, 3 epochs, fp32) and run "
        "in well under a second per article on a T4 GPU. See "
        "[the project repository](https://github.com/cagancaliskan/CENG467-term-project) for "
        "training details, evaluation results, and the LNCS-format paper."
    )


if __name__ == "__main__":
    demo.launch(share=True)
