# Pre-registration — camera-ready revision analyses

**Committed before any multi-seed training run or judge run was executed.**
Git history is the timestamp. Nothing below may be revised after seeing results;
if a decision here turns out to be wrong, the paper says so rather than changing it.

Paper: *Synthetic Data Distillation from Large Language Models for Turkish
Abstractive News Summarization* — IDAP 2026 camera-ready revision.

---

## 1. Primary outcome measure

**ROUGE-1 F1, `standard` tokenization** (Unicode lowercase, punctuation stripped),
as defined in `src/eval/rouge_tr.py`.

All confirmatory tests below are performed on this measure only. BERTScore,
stem-ROUGE, ROUGE precision/recall, and every faithfulness metric are reported
as **secondary / descriptive**. No equivalence or superiority claim in the paper
will rest on a secondary measure.

## 2. Equivalence margin

**δ = 0.010 ROUGE-1 F1.**

Provenance — derived from the **already-published v1 numbers**, before any new
run: v1 Table I reports B3a − B3b = 0.278 − 0.276 = **0.002** and
S-gpt − S-claude = 0.261 − 0.259 = **0.002**. δ = 0.010 is five times those
observed gaps and corresponds to the one-ROUGE-point threshold conventionally
treated as the floor of practical significance in summarization.

δ is **not** derived from the seed variance measured in this revision, because
choosing a margin after seeing the variance is not pre-registration.

## 3. Confirmatory hypotheses

| ID | Claim in the paper | Test | Decision rule |
|---|---|---|---|
| H1 | "the two teachers are essentially indistinguishable" | TOST, paired, B3a vs B3b | equivalence declared only if `p_tost < 0.05`, i.e. the 90% CI of the difference lies inside ±δ |
| H2 | "the choice of teacher LLM does not measurably affect downstream student quality" | TOST, paired, S-gpt vs S-claude | as H1 |
| H3 | "S-gpt is statistically tied with B2 on BERTScore F1" | paired approximate randomization, S-gpt vs B2 | *descriptive only* — BERTScore is a secondary measure; the word "tied" is replaced by the measured CI |
| H4 | "rank 8 is the Pareto-efficient operating point" | paired randomization on R1 for r8 vs r16 and r8 vs r32, plus Wilson CIs on the hallucination rate | if the r8/r16 R1 CIs overlap, the claim is restated as "r8 and r16 are statistically indistinguishable; r8 is preferred on parameter count" |

If TOST fails for H1 or H2, the paper states: *"no evidence of a difference, but
equivalence was not established"* — it does not report a null result as equivalence.

## 4. Uncertainty reporting

Two sources are reported separately and never pooled into one number:

1. **Test-set sampling variance** — paired BCa bootstrap, 10 000 resamples, over
   the fixed test articles. Applies to all systems, including the API teachers.
2. **Training seed variance** — Student *t* interval over seeds, df = m−1
   (m = 3 → multiplier 4.303). A bare standard deviation over three points is
   not reported, because it reads as far tighter than it is.

Significance tests run on per-example scores **averaged over the three seeds**,
so no p-value is conditional on a single training run.

Multiple comparisons within each table are corrected with **Holm–Bonferroni**.

## 5. Seeds

Seeds **42** (already run), **1337**, **2024**. Only the training seed varies —
LoRA initialisation and data shuffling. The training subset is held fixed, so the
measured variance is optimisation noise, not data resampling.

Systems retrained across all three seeds: **B2-human, S-gpt, S-claude** (every
trained system in Table I), plus **S-gpt at LoRA rank 16 and 32** for H4.
Ablations over synthetic size and prompt variant remain single-seed; this is
declared as a limitation rather than silently omitted.

## 6. LLM-as-a-judge protocol

- Sample: **60 articles × 6 systems**, stratified by source length, drawn before
  any judging.
- **Blinded**: system identity is replaced by a per-article random letter A–F.
  (The v1 protocol passed the literal system name into the prompt.)
- Context: **3 000 characters** of source article — the same text the teachers
  received. (The v1 protocol showed the judge 600 characters while asking for a
  strict factuality verdict.)
- Judges: **two independent models, neither from Anthropic**, because the
  Claude Haiku 4.5 teacher generated B3b's summaries and supervised S-claude.
  Every prompt and raw response is written to disk.
- **Exclusion rule, fixed in advance:** any judge whose agreement with the human
  annotations is **Cohen κ < 0.40** on the overlapping subset is excluded from
  the headline table and reported separately. Judges are not silently averaged.
- Self-preference is estimated per article, with a CI, not per system.

## 7. Human annotation

- **40 articles × 5 systems (B2, B3a, B3b, S-gpt, S-claude) = 200 rows.**
  B1 is excluded: no claim in the paper depends on it.
- Blinded and grouped by article; system order randomised within each article,
  article order randomised.
- Axes: `factual_correct`, `entity_hallucination`, `misattribution`,
  `salient_omission`.
- **Power, acknowledged in advance:** at n = 40 per system and a base rate near
  0.10, the Wilson interval half-width is ≈ 0.10 and the minimum detectable
  difference at 80% power is ≈ 0.19. Differences smaller than that will **not**
  be described as "no difference"; the paper reports the MDE alongside the
  result.

## 8. Declared limitation of blinding

Teacher summaries run 2.1–3.1× the reference length and student summaries
0.74–1.43×. Anonymous labels do not remove that cue: a judge or annotator can
infer the system family from length alone. This residual confound is stated in
the paper; a length-stratified sub-analysis is reported alongside the raw
comparison.
