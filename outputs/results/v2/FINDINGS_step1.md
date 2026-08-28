# Camera-ready revision — Step 1 findings (sentinel artifact)

Source: `artifact_impact_{main,ood,abl}.md`, `sentinel_incidence.json`.
Every number below is from the *cleaned* predictions unless marked "published".

## 1. The artifact is deterministic, not degenerate generation

| file | rows | rows w/ sentinel | sentinels | per row | emptied |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2_human | 2000 | 100.0% | 2000 | **1.000** | 0 |
| S_gpt | 2000 | 100.0% | 2000 | **1.000** | 0 |
| S_claude | 2000 | 100.0% | 2000 | **1.000** | 0 |
| S_gpt_r4 / r16 / r32 | 2000 | 100.0% | 2000 | **1.000** | 0 |
| S_gpt_n5k / detailed_1k | 2000 | 100.0% | 2000–2001 | 1.000 | 0 |
| B1_zeroshot | 2000 | 100.0% | 2143 | **1.072** | 0 |
| B3a_gpt / B3b_claude | 2000 | 0.0% | 0 | 0 | 0 |

mT5 is pretrained on span corruption alone, so every training target begins with
`<extra_id_0>` and the decoder reproduces that leading token on every sequence.
The released inference code did not strip it. This is a one-token, fixed-position
decoding bug affecting 100% of small-model outputs — **not** intermittent
degenerate generation. No prediction was emptied by cleaning, so no
empty-prediction sensitivity analysis is required.

The one genuine degeneracy signal: zero-shot B1 emits 1.07 sentinels per row,
i.e. ~7% of its outputs carry a second sentinel beyond the structural one. The
trained students emit exactly one. Training removes the extra span-corruption
behaviour but not the leading token.

## 2. Mechanism, confirmed on real data

Recall change is **exactly 0.0000 for every system and every ROUGE variant**;
precision rises 1.4–2.9 points. Sentinels add spurious prediction tokens, which
can only dilute precision. The two API teachers show **0.0000 change on every
metric** — an internal control that arose for free.

## 3. Corrected Table I (MLSUM-TR test, n=2000)

| System | R1 std | R1 stem | BERTScore F1 | halluc# | lr | extract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 zero-shot mT5 | 0.0942 | 0.1144 | 0.8508 | 0.0180 | 0.27 | 0.645 |
| B2 human-supervised | 0.2770 | 0.3148 | **0.8911** | 0.0035 | 0.70 | 0.995 |
| B3a GPT-4o-mini | **0.2778** | **0.3319** | 0.8908 | 0.0210 | 2.12 | 0.723 |
| B3b Claude Haiku 4.5 | 0.2756 | 0.3314 | 0.8882 | 0.0390 | 2.19 | 0.728 |
| S-gpt (synthetic) | 0.2669 | 0.3072 | 0.8886 | 0.0045 | 0.82 | 0.988 |
| S-claude (synthetic) | 0.2640 | 0.3053 | 0.8870 | 0.0040 | 0.94 | 0.984 |

## 4. Corrected Table II (TR-News OOD, n=1000)

| System | R1 std | R1 stem | BERTScore F1 | halluc# | lr |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 zero-shot mT5 | 0.0960 | 0.1187 | 0.8537 | 0.0210 | 0.38 |
| B2 human-supervised | **0.2790** | **0.3251** | **0.8934** | 0.0020 | 1.00 |
| B3a GPT-4o-mini | 0.2463 | 0.2960 | 0.8891 | 0.0200 | 3.04 |
| B3b Claude Haiku 4.5 | 0.2481 | 0.3001 | 0.8871 | 0.0350 | 3.07 |
| S-gpt (synthetic) | 0.2671 | 0.3135 | 0.8911 | 0.0010 | 1.10 |
| S-claude (synthetic) | 0.2595 | 0.3043 | 0.8880 | 0.0060 | 1.37 |

## 5. Claims that change

| Location | Published | Corrected |
| --- | --- | --- |
| Abstract | "within **1.7** ROUGE-1 points of its teacher in-domain" | **1.1** points (0.2778 − 0.2669) |
| Abstract | "outperforms the teacher by **1.4** points out-of-domain" | **2.1** points (0.2671 − 0.2463) |
| Sec. IV-B | "**91%** of teacher quality" (R1 stem) | **92.6%** (0.3072 / 0.3319) |
| **Fig. 1(b) caption** | "the teachers lead by **~0.02** [BERTScore] over the small students" | gap is **0.0022**; B2 (0.8911) edges past B3a (0.8908). The reported semantic-quality gap was ~90% artifact. |
| Sec. IV-C | student beats teacher OOD on ROUGE only | student also beats teacher on **BERTScore** OOD (0.8911 vs 0.8891) |
| Sec. V (LoRA) | "at rank 32 the student **matches** the GPT teacher on standard ROUGE-1" | r32 = 0.2901 vs B3a 0.2778 — **exceeds** it by 1.2 points |
| Sec. VI-B | "the lexical metrics in Tables I–II are **unaffected**, because ROUGE's tokenizer already discards angle-bracketed sentinels as punctuation" | **False.** `_PUNCT_RE` strips only `<` and `>`; `extra_id_0` survives as a word token (`extra` under the 5-char stemmer). Delete and replace with the measured impact. |
| Sec. VII-A | "extractive overlap ≈0.99" | now consistent with Sec. IV–VI (0.984–0.995). Sec. VII was the only section that stripped sentinels, so it was right all along; the internal inconsistency is resolved in its favour. |

## 6. Claims that do NOT change

`halluc#` is **identical to four decimals** for every system in every table
(Δ = 0.0000): sentinels are not numbers. The "five times fewer hallucinations"
claim is untouched by the artifact and stands or falls on its own (the
length-normalisation work addresses that separately).

Also unchanged: the teacher-choice null result, the diminishing-returns shape of
the size ablation (1k→5k +7.4, 5k→10k +3.1 R1 points), the prompt ablation
(+1.44 R1, +1.7 pp hallucinated numbers), and the rank-32 hallucination spike
(0.0030 → 0.0115, a factor of 3.8).

## 7. Reproduction check

Scoring the *archived* predictions reproduces every published number in
Tables I–II and Sec. V to three decimals (B1 0.0912 vs 0.091; B3a 0.2778 vs
0.278; S-gpt 0.2612 vs 0.261; halluc# 0.0390 vs 0.039; lr 2.1179 vs 2.12; LoRA
0.2434 / 0.2612 / 0.2755 / 0.2853 vs 0.243 / 0.261 / 0.276 / 0.285). The
dirty→clean deltas are therefore attributable to the artifact alone.

All deltas are significant: paired BCa bootstrap 95% CIs (10,000 resamples) over
the 2,000 test articles exclude zero for every affected metric, and are exactly
[0, 0] for the two teachers.
