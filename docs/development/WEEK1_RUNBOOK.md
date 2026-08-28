# Week 1 Runbook — Data, Prompts, Baselines

Every command runs from the repo root. On Colab, prefix shell calls with `!`. All commands are idempotent — caching makes reruns free for teacher generation. **Stop and verify output after every day** before moving on.

Estimated total cost for Week 1: **~$5.50** (most of the project's spend happens here).
Estimated total wall-clock for Week 1 in Colab: **~6-8 hours of API time + ~30 min of GPU time**.

---

## Day 1 — Scaffold + dependencies (30 min)

Status from prior session: scaffold complete (46 files, all CLIs parse `--help`, smoke tests pass).

```bash
# In Colab (or your shell after cloning):
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add OPENAI_API_KEY and ANTHROPIC_API_KEY.
```

**Verification**

```bash
python -c "import torch, transformers, peft, openai, anthropic, datasets, rouge_score, bert_score; print('OK')"
```

Expected: `OK`. If anything errors, reinstall just the failing package — pinned versions in `requirements.txt` are known-good for Colab T4 as of May 2026.

---

## Day 2 — Download MLSUM-TR + TR-News, build 100-article pilot (45 min, $0)

```bash
# 1. MLSUM-TR — 20k train, 2k val, 2k test (capped to keep costs predictable)
python -m src.data.load_mlsum --out-dir data/raw/mlsum_tr

# 2. TR-News — 1000 articles for OOD evaluation
python -m src.data.load_trnews --out data/raw/trnews/test.jsonl --n 1000

# 3. Carve a deterministic 100-article pilot subset for Day 3
python -m src.data.make_pilot \
    --input data/raw/mlsum_tr/train.jsonl \
    --out data/raw/mlsum_tr/pilot_100.jsonl \
    --n 100
```

**Verification**

```bash
wc -l data/raw/mlsum_tr/*.jsonl data/raw/trnews/*.jsonl data/raw/mlsum_tr/pilot_100.jsonl
head -n 1 data/raw/mlsum_tr/pilot_100.jsonl | python -m json.tool | head -n 20
```

Expected counts: ~20k train, ~2k val, ~2k test, ~1k trnews, exactly 100 pilot. Each row has fields: `id`, `article`, `reference`, `topic`, `url`, `date`, `split`, `source`.

**If TR-News fails:** the script tries two HF mirrors. If both 404, fall back to a local CSV: `python -m src.data.load_trnews --local-csv path/to/trnews.csv --n 1000`. Tell me if you hit this.

---

## Day 3 — Validate both teacher prompts on 50 articles each (1 hr, ~$0.05)

This is the cheap, high-value step. We generate 4 small batches (~50 articles × 2 teachers × 2 prompts = ~200 API calls) and eyeball them before committing to the 14k-call full run.

```bash
# Subset the pilot to 50 articles for each variant. The cache uses article IDs,
# so smaller --n on the same input file just generates fewer files in the cache.
PILOT=data/raw/mlsum_tr/pilot_100.jsonl

# OpenAI: concise + detailed
python -m src.teachers.generate --teacher openai --prompt-variant concise  --input "$PILOT" --n 50 --out-dir data/synthetic/openai/concise
python -m src.teachers.generate --teacher openai --prompt-variant detailed --input "$PILOT" --n 50 --out-dir data/synthetic/openai/detailed

# Anthropic: concise + detailed
python -m src.teachers.generate --teacher anthropic --prompt-variant concise  --input "$PILOT" --n 50 --out-dir data/synthetic/anthropic/concise
python -m src.teachers.generate --teacher anthropic --prompt-variant detailed --input "$PILOT" --n 50 --out-dir data/synthetic/anthropic/detailed
```

**Inspect manually**

```bash
python -m scripts.inspect_outputs \
    --pilot data/raw/mlsum_tr/pilot_100.jsonl \
    --concise data/synthetic/openai/concise \
    --detailed data/synthetic/openai/detailed \
    --k 5

python -m scripts.inspect_outputs \
    --pilot data/raw/mlsum_tr/pilot_100.jsonl \
    --concise data/synthetic/anthropic/concise \
    --detailed data/synthetic/anthropic/detailed \
    --k 5
```

**What to check in the output**

- Are the summaries actually in Turkish (no English leakage)?
- Length stats: concise should average ~30-50 words, detailed ~50-80.
- Hallucinated numbers fraction (`frac_hallucinated_numbers`) should be **<0.1**. Higher means the prompt is leaking dates from the model's training data.
- Repetition fraction (`frac_repetition`) should be near zero. Higher means the prompt is encouraging the model to restate.
- Eyeball 3-5 side-by-side samples per teacher. Look for: missing entities, wrong dates, made-up quotes, English words.

**If any prompt looks weak, stop and tell me.** It's far cheaper to iterate on prompts at 50 articles than at 10k.

---

## Days 4-5 — Full 10k concise generation per teacher (4-6 hrs API time, ~$1.90)

Run these one at a time so you can monitor the rate-limit behavior. Each is resumable thanks to the per-article cache; if a session disconnects, just rerun.

```bash
# Day 4: GPT-4o-mini, 10k concise. Expect ~2.5 hr at default rate, ~$1.50.
bash scripts/02_generate_teacher.sh openai concise 10000

# Day 5: Claude 3 Haiku, 10k concise. Expect ~3 hr, ~$2.50.
bash scripts/02_generate_teacher.sh anthropic concise 10000
```

**Verification after each**

```bash
python -m scripts.check_cache \
    --input data/raw/mlsum_tr/train.jsonl \
    --cache-dir data/synthetic/openai/concise \
    --n 10000

python -m scripts.check_cache \
    --input data/raw/mlsum_tr/train.jsonl \
    --cache-dir data/synthetic/anthropic/concise \
    --n 10000
```

Expected: `present=10000 missing=0`. If missing > 0, the script writes `_missing_ids.txt` into the cache dir and returns exit code 2 — just rerun the generate command and it'll fill the gaps from the cached state.

**If you see persistent failures on specific articles** (>0.5% failure rate), check `outputs/logs/` and tell me — it's usually rate limit tuning rather than content moderation.

---

## Day 6 — Detailed-prompt subsets, 1k each teacher (1.5 hrs API time, ~$0.40)

```bash
# 1k each of detailed-prompt for the prompt-design ablation.
bash scripts/02_generate_teacher.sh openai detailed 1000
bash scripts/02_generate_teacher.sh anthropic detailed 1000
```

**Verification**

```bash
python -m scripts.check_cache --input data/raw/mlsum_tr/train.jsonl --cache-dir data/synthetic/openai/detailed --n 1000
python -m scripts.check_cache --input data/raw/mlsum_tr/train.jsonl --cache-dir data/synthetic/anthropic/detailed --n 1000
```

---

## Day 7 — Zero-shot baselines on test set (1.5 hr GPU + API time, ~$1.30)

We need predictions on the MLSUM-TR test set from three systems: B1 (zero-shot mT5-small), B3a (GPT-4o-mini zero-shot), B3b (Claude 3 Haiku zero-shot). B2 and the synthetic students come in Week 2.

```bash
TEST=data/raw/mlsum_tr/test.jsonl
mkdir -p outputs/predictions

# B1 — zero-shot mT5-small (downloads ~1.2 GB on first run; ~10 min on T4)
python -m src.student.infer \
    --model-path google/mt5-small \
    --input $TEST \
    --out outputs/predictions/B1_zeroshot.jsonl

# B3a — GPT-4o-mini as a zero-shot summarizer on the test set (~30 min, ~$0.30)
python -m src.student.infer_teacher \
    --teacher openai --prompt-variant concise \
    --input $TEST \
    --out outputs/predictions/B3a_gpt.jsonl

# B3b — Claude 3 Haiku zero-shot (~45 min, ~$0.55)
python -m src.student.infer_teacher \
    --teacher anthropic --prompt-variant concise \
    --input $TEST \
    --out outputs/predictions/B3b_claude.jsonl
```

**Sanity score (optional but strongly recommended)**

```bash
# Quick metric snapshot for Day 7. BERTScore is the slow bit; --metrics rouge errors
# is fast enough to confirm the predictions look sane.
python -m src.eval.run_eval \
    --pred B1=outputs/predictions/B1_zeroshot.jsonl \
    --pred B3a=outputs/predictions/B3a_gpt.jsonl \
    --pred B3b=outputs/predictions/B3b_claude.jsonl \
    --metrics rouge errors \
    --out-json outputs/results/week1_baselines.json
```

**What to expect**

- B1 (zero-shot mT5-small) ROUGE-1 around **0.10-0.15** — it's not really trained for Turkish summarization, so this should look bad. That's the point.
- B3a / B3b ROUGE-1 around **0.25-0.35**. These are your strong-baseline numbers and should beat the human-supervised B2 and possibly the student S-* in Week 2 (LLMs are a hard ceiling).

If anything is wildly off — e.g. B1 produces empty strings, or B3a/B3b score below B1 — stop and tell me before Week 2.

---

## Week 1 completion checklist

Tick each box only after the verification step passes:

- [ ] Day 1 — `pip install -r requirements.txt` succeeds, `.env` populated.
- [ ] Day 2 — `data/raw/mlsum_tr/{train,validation,test}.jsonl` exist with expected counts; `data/raw/trnews/test.jsonl` exists; `pilot_100.jsonl` has exactly 100 rows.
- [ ] Day 3 — All four pilot caches populated (~50 files each). `inspect_outputs.py` shows reasonable Turkish summaries, `frac_hallucinated_numbers` < 0.1.
- [ ] Day 4 — `data/synthetic/openai/concise/` has 10000 `.json` files. `check_cache.py` prints `missing=0`.
- [ ] Day 5 — Same for `data/synthetic/anthropic/concise/`.
- [ ] Day 6 — Same for both `*/detailed/` directories at 1000 files each.
- [ ] Day 7 — Three baseline prediction files exist. Quick `run_eval.py` shows B1 ≪ B3a, B3b.

When all seven are ticked, ping me and I'll move on to Week 2 (student training).
