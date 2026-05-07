# Project Plan — Synthetic Data Distillation for Turkish News Summarization

**Course:** CENG 467 — Natural Language Understanding and Generation, IYTE, Spring 2026
**Author:** Çağan Çalışkan
**Instructor:** Prof. Dr. Aytuğ Onan

---

## 1. One-line description

Distill abstractive summarization capabilities of large teacher LLMs (GPT-4o-mini and Claude) into a small student model (mT5-small + LoRA) for Turkish news, then compare against zero-shot, human-supervised, and teacher-LLM baselines.

## 2. Research questions

1. Does training mT5-small on synthetic teacher summaries beat training on the same number of human reference summaries from MLSUM-TR?
2. How does student quality scale with synthetic dataset size (1k → 5k → 10k)?
3. Does prompt design (concise vs. detailed) at the teacher stage propagate measurable changes to the student?
4. Does LoRA rank (4, 8, 16, 32) trade off quality vs. memory in a meaningful way at this scale?
5. Does the choice of teacher (GPT-4o-mini vs. Claude) measurably bias student behavior on Turkish summarization?

## 3. Experimental matrix

### Systems we evaluate

| ID | System | Source of supervision |
| --- | --- | --- |
| `B1` | mT5-small zero-shot | none (off-the-shelf) |
| `B2` | mT5-small + LoRA on human refs (MLSUM-TR) | human |
| `B3a` | GPT-4o-mini zero-shot | n/a (teacher itself) |
| `B3b` | Claude (Haiku) zero-shot | n/a (teacher itself) |
| `S-gpt` | mT5-small + LoRA on GPT-4o-mini synthetic | synthetic |
| `S-claude` | mT5-small + LoRA on Claude synthetic | synthetic |

### Ablations (run on the better-performing teacher only, to control compute)

| Axis | Levels |
| --- | --- |
| Synthetic size | 1k, 5k, 10k |
| Teacher prompt | concise, detailed |
| LoRA rank | 4, 8, 16, 32 |

### Evaluation

- Metrics: ROUGE-1, ROUGE-2, ROUGE-L (multilingual tokenizer), BERTScore (xlm-roberta-large baseline).
- In-domain test: MLSUM-TR test split (>=1000 articles).
- Out-of-domain test: TR-News held-out subset (>=500 articles).
- Qualitative: 30 cases hand-labeled for hallucination, omission, repetition, morphological errors.

## 4. Compute and budget

### GPU

Colab free-tier T4, 15 GB VRAM. mT5-small (~300M params) + LoRA at fp16 with batch size 4 and gradient accumulation 4 (effective batch 16) fits comfortably.

### Teacher API budget (estimate, balanced tier)

Articles truncated to 3000 chars (~750 tokens). Average summary ~80 tokens.
Per-teacher call volume: 10k concise + 1k detailed (train) + 2k MLSUM test + 1k TR-News OOD = **14,000 calls**.

| Teacher | Model | Input price | Output price | Per call | 14k calls |
| --- | --- | --- | --- | --- | --- |
| GPT-4o-mini | `gpt-4o-mini` | $0.15/M | $0.60/M | ~$0.000168 | ~$2.35 |
| Claude 3 Haiku | `claude-3-haiku-20240307` | $0.25/M | $1.25/M | ~$0.000300 | ~$4.20 |

**Combined main cost: ~$6.55. Plus ~$0.50 for pilots and retries. Grand total: ~$7.**

Lever taken to get here: switched Claude from Haiku 4.5 ($0.80/$4) to Haiku 3 ($0.25/$1.25), tightened article truncation 8000 -> 3000 chars. Real-time API (no Batch API discount used — easier debugging and faster iteration).

### Caching

Every API call cached on disk by `sha256(article_id + prompt_variant + teacher_name)`. Reruns cost zero. **This is non-negotiable** — without it a single rerun blows the budget.

## 5. Three-week timeline

### Week 1 — Data, prompts, baselines

| Day | Deliverable |
| --- | --- |
| 1 | Repo scaffold, dependencies installed in Colab |
| 2 | MLSUM-TR + TR-News loaders working, 100-article pilot subset cached |
| 3 | Both teacher prompts validated on 50 articles each, manual inspection of outputs |
| 4 | Generate 10k summaries from GPT-4o-mini (concise prompt) — cache verified |
| 5 | Generate 10k summaries from Claude (concise prompt) |
| 6 | Generate 1k summaries from each teacher with detailed prompt (for prompt ablation) |
| 7 | Run zero-shot mT5-small + zero-shot teacher baselines on test set, save predictions |

### Week 2 — Student training and main results

| Day | Deliverable |
| --- | --- |
| 8 | Train `B2` (mT5 + LoRA on human refs, 10k pairs) — main human-baseline run |
| 9 | Train `S-gpt` and `S-claude` at 10k each, LoRA rank 8 — main synthetic runs |
| 10 | Run full evaluation on 6 systems (B1, B2, B3a, B3b, S-gpt, S-claude) |
| 11 | Pick stronger teacher, train size ablation: 1k, 5k variants |
| 12 | Train LoRA-rank ablation: 4, 16, 32 (rank 8 already done) |
| 13 | Train prompt ablation: detailed-prompt 1k subset |
| 14 | Generate predictions on TR-News OOD set for all systems |

### Week 3 — Analysis and report

| Day | Deliverable |
| --- | --- |
| 15 | Compile metric tables, run BERTScore (more memory-hungry) in batched chunks |
| 16 | Hand-label 30 outputs across systems for error categories |
| 17 | Build figures: scaling curve, LoRA rank curve, prompt comparison bars |
| 18 | Draft report sections 1-4 (intro, related work, method, experiments) |
| 19 | Draft report sections 5-7 (results, error analysis, ethics, conclusion) |
| 20 | Live demo (Gradio Space or Colab notebook), polish report, push final commit |
| 21 | Buffer day for unblocking surprises |

## 6. Reproducibility checklist

- All scripts have argparse, no notebook-only state.
- Single `requirements.txt` pinned to exact versions known to work on Colab T4.
- Every experiment writes a `run_config.json` next to its outputs.
- Random seeds set deterministically in `src/utils/seed.py` — all runs use `--seed 42` by default.
- Synthetic datasets and final predictions are committed (compressed) to the repo for full reproducibility without re-running API calls.
- README has a single command per stage.

## 7. Risk register

| Risk | Mitigation |
| --- | --- |
| API quota or rate limits | Exponential backoff, on-disk cache, parallelism capped at 4 |
| MLSUM-TR licensing question | Use HuggingFace mirror with CC license, cite Scialom et al. 2020 |
| TR-News access | Fall back to author repo if HF mirror unavailable |
| Colab session timeout during training | Save checkpoints every 500 steps to Drive |
| BERTScore OOM on T4 | Batch size 8, optionally CPU fallback for final scoring pass |
| Turkish-specific tokenization underestimating ROUGE | Report stem-aware ROUGE variant alongside standard, discuss in §6 of report |

## 8. Ethics considerations (write-up scope)

- Hallucination risk on news (factual fabrication consequences).
- Bias inheritance from teacher LLMs trained primarily on English.
- Misuse: synthetic Turkish news summaries could be used to fabricate news at scale.
- Carbon cost transparency: report total GPU-hours and API calls.
