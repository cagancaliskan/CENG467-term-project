# Camera-ready edit list — apply in order

Every number below is measured, not estimated. Source: `outputs/results/v2/`.
Sanity anchor: scoring the *archived* predictions reproduces every published
number in Tables I–II and Sec. V to three decimals, so the changes below are
attributable to the decoding artifact alone.

Statistics: paired BCa bootstrap, 10,000 resamples, over the test articles.
Pairwise tests are paired approximate randomization with Holm–Bonferroni
correction. Equivalence is TOST at the pre-registered margin δ = 0.010 ROUGE-1
(`PREREGISTRATION.md`). Paired tests use the 1,962 MLSUM-TR articles present in
all six prediction files.

---

## EDIT 1 — Abstract

**Find:** `reaches within \textbf{1.7} ROUGE-1 points of its teacher in-domain and \emph{outperforms} the teacher by \textbf{1.4} ROUGE-1 points out-of-domain`

**Replace:** `reaches within \textbf{1.1} ROUGE-1 points of its teacher in-domain and \emph{significantly outperforms} it by \textbf{2.1} ROUGE-1 points out-of-domain (paired randomization test, $p<0.01$ after Holm correction)`

**Then find:** `The student also hallucinates numeric facts \textbf{approximately five times less often} than its teacher.`

**Append immediately after:** ` All results are reported with 95\% bootstrap confidence intervals, and the two teachers are shown to be statistically \emph{equivalent} in-domain under a pre-registered $\pm 0.010$ ROUGE-1 margin (TOST, $p<10^{-7}$) rather than merely indistinguishable.`

**Also in the abstract, find:** `we identify and fix a SentencePiece decoding artifact that had depressed the small models' apparent fluency`

**Replace:** `we identify and quantify a SentencePiece decoding artifact that had depressed not only the small models' apparent fluency but every lexical and semantic metric they were scored on`

---

## EDIT 2 — Table I (MLSUM-TR test, 2,000 articles)

Replace the six data rows. Numbers in brackets are 95\% CIs for R1 std; add
them as a new column or fold into the caption, whichever fits.

| System | R1 std | 95% CI | R2 std | RL std | R1 stem | BSf1 | halluc# | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 zero-shot mT5 | 0.094 | [0.090, 0.099] | 0.039 | 0.088 | 0.114 | 0.851 | 0.018 | 0.27 |
| B2 human-supervised | 0.277 | [0.268, 0.286] | **0.176** | **0.253** | 0.315 | **0.891** | 0.004 | 0.70 |
| B3a GPT-4o-mini | **0.278** | [0.273, 0.283] | 0.141 | 0.228 | **0.332** | 0.891 | 0.021 | 2.12 |
| B3b Claude Haiku 4.5 | 0.276 | [0.270, 0.281] | 0.141 | 0.227 | 0.331 | 0.888 | 0.039 | 2.19 |
| S-gpt (synthetic) | 0.267 | [0.258, 0.276] | 0.163 | 0.237 | 0.307 | 0.889 | 0.005 | 0.82 |
| S-claude (synthetic) | 0.264 | [0.256, 0.272] | 0.159 | 0.234 | 0.305 | 0.887 | 0.004 | 0.94 |

**Caption — append:** ` Bracketed values are 95\% BCa bootstrap confidence intervals over the test articles (10,000 resamples). All systems are scored on sentinel-stripped predictions; see Sec.~VI-B.`

---

## EDIT 3 — Table II (TR-News test, 1,000 articles)

| System | R1 std | 95% CI | R1 stem | BSf1 | halluc# | lr | Δ R1 std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 zero-shot mT5 | 0.096 | [0.089, 0.104] | 0.119 | 0.854 | 0.021 | 0.38 | $+0.002$ |
| B2 human-supervised | **0.279** | [0.266, 0.292] | **0.325** | **0.893** | 0.002 | 1.00 | $+0.002$ |
| B3a GPT-4o-mini | 0.246 | [0.239, 0.254] | 0.296 | 0.889 | 0.020 | 3.04 | $-0.032$ |
| B3b Claude Haiku 4.5 | 0.248 | [0.241, 0.255] | 0.300 | 0.887 | 0.035 | 3.07 | $-0.028$ |
| S-gpt (synthetic) | 0.267 | [0.255, 0.280] | 0.314 | 0.891 | **0.001** | 1.10 | $+0.000$ |
| S-claude (synthetic) | 0.260 | [0.248, 0.271] | 0.304 | 0.888 | 0.006 | 1.37 | $-0.005$ |

---

## EDIT 4 — Fig. 1 caption (this one is mandatory; the current claim is now false)

**Find:** `(b)~BERTScore F1 rewards semantic similarity; the teachers lead by $\sim$0.02 over the small students.`

**Replace:** `(b)~BERTScore F1 rewards semantic similarity. On sentinel-stripped output the teachers no longer lead: B3a reaches 0.891 and the human-supervised B2 0.891, with the distilled S-gpt at 0.889. The $\sim$0.02 gap reported in the submitted version was an artifact of the undstripped decoder output.`

*(Regenerate the figure from `outputs/results/v2/main_eval_CLEAN.json` if time permits; if not, the corrected caption plus Table I is enough, but say in the caption that the panel is drawn from the corrected values.)*

---

## EDIT 5 — Sec. IV-B, "Three observations follow"

Replace the whole paragraph with:

> Three observations follow. First, distillation transfers most teacher
> capability: S-gpt reaches R1 stem 0.307 against the teacher's 0.332
> (\textbf{92.6\%} of teacher quality). On standard ROUGE-1 the teacher's
> advantage is \textbf{0.010} points (Holm-corrected $p=0.046$) over S-gpt and
> is \emph{not} significant against the Claude teacher ($p=0.16$). The
> human-supervised B2 is statistically indistinguishable from the GPT teacher
> ($-0.0001$, $p=1.0$) and matches it on BERTScore F1 (0.891 vs.\ 0.891).
> Second, the two teachers are not merely similar but \textbf{equivalent}: the
> difference is $+0.0017$ with a 90\% CI of $[-0.0009, +0.0043]$, inside the
> pre-registered $\pm 0.010$ margin (TOST, $p=7.6\times10^{-8}$). The same holds
> for their distilled students in-domain ($+0.0025$, TOST $p=6.5\times10^{-4}$).
> Teacher choice is therefore a genuinely weak design lever here, and we can now
> say so with an equivalence test rather than an absence of evidence. Third, the
> small students hallucinate numbers \textbf{five times less} than their teachers
> (0.004--0.005 vs.\ 0.021 and 0.039): the student inherits an extractive bias
> (unigram overlap with source 0.98--0.99) that the LLM teachers do not have.

---

## EDIT 6 — Sec. IV-C, out-of-domain

**Find:** `the distilled S-gpt outperforms its GPT teacher on TR-News (\textbf{0.260} vs.\ 0.246 R1~std)`

**Replace:** `the distilled S-gpt significantly outperforms both teachers on TR-News (\textbf{0.267} vs.\ 0.246 for B3a, $p=0.0015$; vs.\ 0.248 for B3b, $p=0.0024$; Holm-corrected paired randomization), and it also overtakes them on BERTScore F1 (0.891 vs.\ 0.889 and 0.887)`

**Append at the end of that subsection:**

> Out of domain the teacher-choice equivalence no longer holds: S-gpt exceeds
> S-claude by $+0.0076$ R1~std with a 90\% CI of $[+0.0023, +0.0129]$, which
> leaves the $\pm 0.010$ margin, so equivalence is \emph{not} established
> (TOST $p=0.23$). Teacher choice appears immaterial in-domain but may matter
> under distribution shift --- a distinction the in-domain comparison alone
> would have hidden.

---

## EDIT 7 — Sec. V, LoRA rank

**Find:** `At rank 32 the student matches the GPT teacher on standard ROUGE-1`

**Replace:** `At rank 32 the student \emph{exceeds} the GPT teacher on standard ROUGE-1 (0.290 vs.\ 0.278)`

**And update the rank series:** `0.243 (r4) $\to$ 0.261 (r8) $\to$ 0.276 (r16) $\to$ 0.285 (r32)`
**becomes:** `0.249 (r4) $\to$ 0.267 (r8) $\to$ 0.281 (r16) $\to$ 0.290 (r32)`

Hallucination series is unchanged: 0.003 at rank 16 $\to$ 0.012 at rank 32.

**Append one sentence:** `The rank-8 recommendation rests on the hallucination trade-off rather than on a quality difference; with single-seed runs we cannot exclude that r8 and r16 differ only by training noise.`

---

## EDIT 8 — Sec. V, dataset size and prompt

Size series `0.160 to 0.261` becomes `0.162 to 0.267`; the intermediate 5k value
is 0.236. The shape is unchanged: $+7.4$ points from 1k to 5k, $+3.1$ from 5k to
10k. Phase-transition numbers become len\_ratio 2.33 / halluc\# 0.051 at 1k and
len\_ratio 0.78 / halluc\# 0.010 at 5k.

Prompt ablation is unchanged: detailed yields $+1.4$ R1 std points (0.162 $\to$
0.176) at $+1.7$ percentage points of hallucinated numbers (0.051 $\to$ 0.068).

---

## EDIT 9 — Sec. VI-B (MANDATORY: the current text states something false)

**Delete:**

> Two consequences matter for interpreting Table IV. First, the lexical metrics
> in Tables I–II are *unaffected*, because ROUGE's tokenizer already discards
> angle-bracketed sentinels as punctuation.

**Replace with:**

> Two consequences matter. First, contrary to what we assumed when the artifact
> was found, the lexical metrics in Tables~I--II are \emph{not} immune to it.
> Our ROUGE tokenizer strips only non-word characters, so \texttt{<extra\_id\_0>}
> loses its angle brackets but survives as the word token
> \texttt{extra\_id\_0} (and as \texttt{extra} under the 5-character stemmer).
> The artifact is fully systematic: mT5 is pretrained on span corruption alone,
> every training target begins with the first sentinel, and the decoder
> reproduces it on \textbf{100\% of small-model outputs} --- exactly one
> occurrence per summary for B2, S-gpt, S-claude and every LoRA and data-size
> variant, and 1.07 per summary for the untrained B1, whose surplus is genuine
> span-corruption behaviour. The two API teachers are unaffected, giving a
> free internal control. Re-scoring the sentinel-stripped predictions leaves
> ROUGE \emph{recall} unchanged to four decimals for every system while raising
> precision by 1.4--2.9 points; ROUGE-1 F1 rises 0.5--0.7 points and BERTScore
> F1 by 1.6--2.6 points for the four affected systems, and by exactly zero for
> the teachers. All deltas are significant under a paired bootstrap. Every
> number in Tables~I--III and Sec.~V is reported on stripped output. Second, the
> affected content is otherwise sound (factual correctness 83--97\% on the very
> same outputs), so the fluency column of Table~IV reflects pre-strip text and
> understates the deployed system.
>
> We verified that post-hoc stripping is identical to regenerating with the
> fixed decoder: on 200 articles decoded twice from the same checkpoint, once
> with and once without the strip, the regular expression applied to the
> unstripped output reproduced the stripped output on \textbf{198 of 198}
> comparable rows.
>
> We report the pre-strip judge scores in Table~IV unchanged, for transparency.
> Re-running the qualitative evaluation on stripped output is the most important
> piece of work this revision could not complete; see Sec.~VII-B.

---

## EDIT 10 — Sec. VII-B, Evaluation and Model Limitations

**Delete** the sentence beginning `Finally, the judge, Claude Opus 4.7, shares with one of the evaluated systems...` and **replace the end of the subsection** with:

> Four limitations of this revision should be stated plainly rather than left
> to inference.
>
> \emph{Single training run.} Every trained system is one run at seed 42. The
> confidence intervals above capture test-set sampling uncertainty only; they do
> not capture training-seed variance, so the equivalence results should be read
> as equivalence \emph{of these particular runs} on this test set. A three-seed
> protocol (seeds 42/1337/2024 for B2, S-gpt, S-claude and for LoRA ranks 16 and
> 32) is implemented and pre-registered in the repository but was not executed
> before the camera-ready deadline.
>
> \emph{Judge independence.} The qualitative labels in Table~IV come from a
> single Claude model, and the Claude Haiku~4.5 teacher generated B3b's
> summaries and supervised S-claude, so the judge is not independent of two of
> the six systems it scores. The judge prompt also carried the system label and
> only the first 600 characters of the source article while asking for a strict
> factuality verdict. A blinded protocol with a 3,000-character context and two
> non-Anthropic judges is implemented in the repository; running it, and
> re-running the dirty/clean contrast of Sec.~VI-B, is the first item of future
> work.
>
> \emph{Faithfulness is measured on numbers only.} Our \texttt{halluc\#} flag
> checks whether a numeric token in the summary appears in the source. It says
> nothing about fabricated or misattributed persons, events and relations,
> which for a news summarizer are the higher-harm error class, and it is
> confounded with length: the teachers write 2.1--3.1$\times$ the reference
> length and the students 0.7--1.4$\times$, so a per-summary rate mechanically
> favours the shorter system. The five-fold figure should be read as a
> length-uncontrolled, numeric-only estimate. Entity-level precision and recall,
> length-normalised rates, and a human-annotated misattribution axis are
> specified in the repository but not yet measured.
>
> \emph{Omission is partly architectural.} The student's 512-token input window
> covers the median MLSUM-TR article (1,159 characters) but not the tail (p95
> 4,061; max 26,612). Some of what reads as omission is content the student
> never saw.

---

## EDIT 11 — Sec. VII-C, Reproducibility

**Find:** `All prompts, training configurations, evaluation scripts, and synthetic data caches are publicly available at`

**Replace:** `All prompts, training configurations, evaluation scripts, per-example scores, the statistical analysis code, and the pre-registration of the equivalence margin are publicly available at`

*(The synthetic caches were claimed but not actually committed. Either commit the
consolidated JSONLs produced by `scripts/consolidate_synthetic.py` and keep the
original wording, or use the replacement above, which is true as written.)*

**Also remove the dead Gradio URL** from the demo mention: it expired seven days
after 26 May 2026.

---

## EDIT 12 — Conclusion

**Find:** `matches the teacher within \textbf{1.7} ROUGE-1 points in-domain, outperforms the teacher by \textbf{1.4} R1 points out-of-domain`

**Replace:** `matches the teacher within \textbf{1.1} ROUGE-1 points in-domain, significantly outperforms both teachers by \textbf{1.9--2.1} R1 points out-of-domain`

**Find:** `The teacher comparison is essentially a null result`

**Replace:** `The teacher comparison is an equivalence result in-domain, established by TOST at a pre-registered $\pm 0.010$ margin rather than by a failure to reject`

---

## New references to add

- H. Lakens, "Equivalence tests: a practical primer for $t$-tests, correlations, and meta-analyses," \emph{Soc. Psychol. Personal. Sci.}, vol.~8, no.~4, pp.~355--362, 2017.
- R. Dror, G. Baumer, S. Shlomov, and R. Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing," in \emph{Proc. ACL}, 2018, pp.~1383--1392.
