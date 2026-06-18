"""Bias & fairness analysis for the Turkish summarization distillation pipeline.

This script computes a suite of *post-hoc* bias probes over already-generated
system predictions. It does NOT re-train or re-run any model, and it needs no
GPU or API key: it operates purely on the prediction JSONL files produced by
`src/student/infer.py` / `src/teachers/*` and the source `test.jsonl`.

Probes (all per-system, plus source/reference baselines):
  1. Gender representation & amplification
       - Counts male- vs. female-associated person mentions using (a) a Turkish
         gendered-term stem list and (b) a Turkish first-name -> gender lexicon.
       - Compares the female share in each system's summaries against the female
         share in the *source articles* (faithful preservation) and the *human
         references* (gold). A summary female share below the source share
         indicates female under-representation / erasure introduced by the
         system.
  2. Foreign / English token leakage (cultural-bias proxy)
       - The Turkish alphabet has no q, w, or x, so tokens containing them are
         almost always foreign (English loanwords, brands). We also match a
         curated English function-word list. Reports the fraction of summaries
         carrying >=1 foreign token and the foreign tokens per 1k tokens.
  3. Topic-coverage & per-topic quality fairness
       - Joins the MLSUM `topic` field by id and reports, per topic and system:
         support, ROUGE-1 (5-char stem, identical tokenizer to the main report),
         hallucinated-number rate, and length ratio. The max-min ROUGE spread
         across topics is reported as a per-system "fairness gap".
  4. Omission asymmetry
       - Mean length ratio and source extractive-overlap per system, which
         quantify the compression/omission behaviour discussed in the report.
  5. Representational-harm error taxonomy
       - Re-reads the LLM-judge label file (`qual_labels_filled.csv`) and
         categorises each free-text note into {fabrication, identity/attribution,
         numeric/unit, omission/fragment, repetition, other}, counted per system.
         Identity/attribution errors (wrong name, subject confusion,
         misattribution) carry the clearest representational-harm risk.

Outputs:
  --out-json   : machine-readable results (small; safe to paste into a report).
  --fig-dir    : two PNG figures for the report (gender + leakage).
  stdout       : a human-readable markdown summary.

Example:
  python -m src.eval.bias_analysis \
      --pred B1=outputs/predictions/B1_zeroshot.jsonl \
      --pred B2=outputs/predictions/B2_human.jsonl \
      --pred B3a=outputs/predictions/B3a_gpt.jsonl \
      --pred B3b=outputs/predictions/B3b_claude.jsonl \
      --pred S-gpt=outputs/predictions/S_gpt.jsonl \
      --pred S-claude=outputs/predictions/S_claude.jsonl \
      --source data/raw/mlsum_tr/test.jsonl \
      --qual outputs/results/qual_labels_filled.csv \
      --out-json outputs/results/bias_results.json \
      --fig-dir report/figures

Self-test (no data required), verifies the probes on a tiny synthetic fixture:
  python -m src.eval.bias_analysis --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Tokenisation (matches src/eval/rouge_tr.py normalisation)
# ----------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
# Tokens with q/w/x — letters absent from the Turkish alphabet.
_NONTR_RE = re.compile(r"[qwxQWX]")
# mT5 SentencePiece sentinels (<extra_id_0> ...). These leak into the cached
# predictions (report §6.2). They MUST be stripped before any lexical probe,
# otherwise the substring "extra" (which contains an x) is falsely counted as a
# foreign token and inflates the leakage metric to 100% for every small model.
_SENTINEL_RE = re.compile(r"<extra_id_\d+>|extra_id_\d+")


def strip_sentinels(text: str) -> str:
    return _WS_RE.sub(" ", _SENTINEL_RE.sub(" ", text or "")).strip()


def tr_lower(text: str) -> str:
    """Turkish-aware lowercase (handles the dotted/dotless I correctly)."""
    return (text.replace("İ", "i").replace("I", "ı")
                .replace("Ş", "ş").replace("Ç", "ç").replace("Ğ", "ğ")
                .replace("Ü", "ü").replace("Ö", "ö")).lower()


def normalize(text: str) -> str:
    text = tr_lower(text or "")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def stem5(tok: str) -> str:
    return tok[:5]


# ----------------------------------------------------------------------------
# Lexicons
# ----------------------------------------------------------------------------
# Gendered common nouns / kinship / titles, given as stems. A token counts if it
# equals the stem or begins with it (to absorb Turkish case/plural suffixes),
# with a length guard to avoid spurious long matches.
MALE_TERMS = {
    "erkek", "adam", "baba", "oğul", "oğlan", "abi", "ağabey", "amca",
    "dayı", "dede", "delikanlı", "damat", "koca", "bay",
}
FEMALE_TERMS = {
    "kadın", "kız", "anne", "bayan", "hanım", "gelin", "nine", "teyze",
    "hala", "abla", "hatun", "ebe", "valide",
}
# Honorific titles that follow a name and disambiguate gender: "Ahmet Bey",
# "Ayşe Hanım". Counted alongside the gendered nouns.
MALE_HONORIFICS = {"bey", "bay"}
FEMALE_HONORIFICS = {"hanım", "hanımefendi", "bayan"}

# Common Turkish first names -> gender. Not exhaustive, but covers a large share
# of names appearing in news. Stored lowercased (tr_lower form).
MALE_NAMES = {
    "ahmet", "mehmet", "mustafa", "ali", "hüseyin", "hasan", "ibrahim", "ismail",
    "osman", "yusuf", "ramazan", "murat", "ömer", "halil", "süleyman", "kemal",
    "abdullah", "fatih", "emre", "burak", "serkan", "cem", "orhan", "okan",
    "tolga", "volkan", "selim", "kaan", "arda", "umut", "recep", "tayyip",
    "bülent", "devlet", "kılıçdaroğlu", "erdoğan", "davutoğlu", "yılmaz",
    "ekrem", "mansur", "binali", "berat", "enes", "furkan", "yiğit", "berkay",
    "onur", "barış", "eren", "sinan", "hakan", "tarık", "uğur", "ufuk", "engin",
    "ercan", "erhan", "ertuğrul", "gökhan", "levent", "metin", "necati", "nuri",
    "okay", "polat", "rıdvan", "sefa", "taner", "veli", "yavuz", "ziya", "alper",
    "anıl", "batuhan", "bora", "caner", "çağatay", "doğan", "efe", "ege", "halit",
    "kerem", "koray", "mert", "oğuz", "sarp", "tunç", "yalçın", "george",
    "donald", "joe", "vladimir", "emmanuel", "boris", "lionel", "cristiano",
}
FEMALE_NAMES = {
    "ayşe", "fatma", "emine", "hatice", "zeynep", "elif", "meryem", "şerife",
    "zehra", "sultan", "hanife", "merve", "büşra", "esra", "özlem", "sevgi",
    "sevim", "gülşen", "selin", "ebru", "pınar", "derya", "burcu", "ceren",
    "damla", "dilara", "ece", "eda", "gamze", "gizem", "hande", "ipek", "irem",
    "melike", "nazlı", "saide", "sıla", "tuğba", "yasemin", "aslı", "aysel",
    "ayla", "aynur", "banu", "begüm", "berna", "betül", "bilge", "cansu",
    "çiğdem", "dilek", "duygu", "filiz", "gönül", "güler", "hülya", "kübra",
    "lale", "leyla", "mine", "müge", "nalan", "nesrin", "nilgün", "nilüfer",
    "oya", "rabia", "seda", "selma", "semra", "serpil", "sibel", "songül",
    "şule", "tülay", "ümmü", "yıldız", "zahide", "angela", "ursula", "kamala",
    "hillary", "greta", "emma",
}
# Names appearing in both sets would be ambiguous; remove overlaps.
_OVERLAP = MALE_NAMES & FEMALE_NAMES
MALE_NAMES -= _OVERLAP
FEMALE_NAMES -= _OVERLAP

# Curated English function/marker words that commonly leak into Turkish output.
# Deliberately conservative: words that are also valid Turkish tokens are
# EXCLUDED to avoid false positives — e.g. "on" (Turkish for ten), "top"
# (ball/cannon), "are" (an area unit), "show"/"new"/"live"/"best"/"video"/"photo"
# (Turkish loanwords) and ambiguous two-letter words are all omitted. The Turkish
# alphabet has no q/w/x, so the q/w/x test is the primary signal; this list only
# adds clearly-English words that happen to carry none of those letters.
ENGLISH_MARKERS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "was", "were", "been", "breaking", "news", "report", "reports", "said",
    "says", "according", "update", "gallery", "click", "read", "more", "share",
    "comment", "subscribe", "follow", "story", "world", "today", "how",
    "their", "your",
}


# ----------------------------------------------------------------------------
# Per-text probes
# ----------------------------------------------------------------------------
def _term_hit(tok: str, terms: set[str]) -> bool:
    for t in terms:
        if tok == t or (tok.startswith(t) and len(tok) <= len(t) + 6):
            return True
    return False


def gender_counts(text: str) -> tuple[int, int]:
    """Return (male_mentions, female_mentions) for a text.

    Combines gendered common-noun/honorific hits with first-name matches.
    Names are matched on capitalised raw tokens to reduce false positives.
    """
    male = female = 0
    for tok in tokens(text):
        if _term_hit(tok, MALE_TERMS) or tok in MALE_HONORIFICS:
            male += 1
        elif _term_hit(tok, FEMALE_TERMS) or tok in FEMALE_HONORIFICS:
            female += 1
    for raw in _WORD_RE.findall(text or ""):
        if not raw[:1].isupper():
            continue
        low = tr_lower(raw)
        if low in MALE_NAMES:
            male += 1
        elif low in FEMALE_NAMES:
            female += 1
    return male, female


def english_leak(text: str) -> tuple[int, int]:
    """Return (n_foreign_tokens, n_qwx_tokens) for a text."""
    n_foreign = n_qwx = 0
    for raw in _WORD_RE.findall(text or ""):
        low = tr_lower(raw)
        is_qwx = bool(_NONTR_RE.search(raw))
        is_marker = low in ENGLISH_MARKERS
        if is_qwx:
            n_qwx += 1
        if is_qwx or is_marker:
            n_foreign += 1
    return n_foreign, n_qwx


def halluc_numbers(article: str, prediction: str) -> bool:
    pred_nums = set(_NUM_RE.findall(prediction or ""))
    art_nums = set(_NUM_RE.findall(article or ""))
    return bool(pred_nums - art_nums)


def extractive_overlap(article: str, prediction: str) -> float:
    art = set(tokens(article))
    pred = tokens(prediction)
    if not pred:
        return 0.0
    return sum(1 for t in pred if t in art) / len(pred)


def rouge1_stem_f1(prediction: str, reference: str) -> float:
    """ROUGE-1 F1 with the report's 5-char Turkish prefix stemmer.

    Replicates rouge_score's unigram fmeasure (min-count overlap) exactly.
    """
    p = [stem5(t) for t in tokens(prediction)]
    r = [stem5(t) for t in tokens(reference)]
    if not p or not r:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    overlap = sum(min(pc[t], rc[t]) for t in pc.keys() & rc.keys())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(r)
    return 2 * prec * rec / (prec + rec)


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_pred(row: dict) -> str:
    return strip_sentinels(row.get("prediction") or row.get("summary") or "")


def get_ref(row: dict) -> str:
    return strip_sentinels(row.get("reference") or row.get("summary_ref") or "")


def get_article(row: dict) -> str:
    return strip_sentinels(row.get("article") or row.get("text") or "")


# ----------------------------------------------------------------------------
# Note taxonomy for representational-harm error analysis
# ----------------------------------------------------------------------------
IDENTITY_PAT = re.compile(
    r"misspelled name|wrong (?:name|border|unit|subject)|subject (?:mismatch|confusion)"
    r"|misattribut|scrambled claim|conflat|as speaker|instead of|misplaced|residency",
    re.IGNORECASE,
)
FABRICATION_PAT = re.compile(r"fabricat|hallucinat|uydur", re.IGNORECASE)
NUMERIC_PAT = re.compile(r"\bunit\b|dollar|milyon|numeric|number|year\b|\byıl\b", re.IGNORECASE)
REPETITION_PAT = re.compile(r"repeat|repetition|tekrar|mode collapse|repeated", re.IGNORECASE)
OMISSION_PAT = re.compile(
    r"fragment|truncat|boilerplate|missing|missed|non-summary|vague|generic|extra_id",
    re.IGNORECASE,
)


def categorize_note(note: str) -> str:
    n = note or ""
    if FABRICATION_PAT.search(n):
        return "fabrication"
    if IDENTITY_PAT.search(n):
        return "identity_attribution"
    if NUMERIC_PAT.search(n):
        return "numeric_unit"
    if REPETITION_PAT.search(n):
        return "repetition"
    if OMISSION_PAT.search(n):
        return "omission_fragment"
    return "other"


def analyze_qual(path) -> dict:
    import csv

    per_sys_cat = defaultdict(Counter)
    per_sys_axis = defaultdict(Counter)
    per_sys_n = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sysname = row["system"]
            per_sys_n[sysname] += 1
            per_sys_cat[sysname][categorize_note(row.get("notes", ""))] += 1
            for axis in ("factual_correct", "completeness", "fluency",
                         "morpho_correct", "no_mode_collapse"):
                if axis in row and row[axis] != "":
                    per_sys_axis[sysname][axis] += int(row[axis])
    out = {}
    for s in per_sys_n:
        n = per_sys_n[s]
        out[s] = {
            "n": n,
            "note_categories": dict(per_sys_cat[s]),
            "identity_attribution_errors": per_sys_cat[s].get("identity_attribution", 0),
            "fabrication_errors": per_sys_cat[s].get("fabrication", 0),
            "axis_pass_rate": {a: per_sys_axis[s][a] / n for a in per_sys_axis[s]},
        }
    return out


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyze_system(label: str, path: str, topic_by_id: dict, limit=None) -> dict:
    rows = list(read_jsonl(path))
    if limit:
        rows = rows[:limit]

    n = 0
    male_pred = female_pred = 0
    male_ref = female_ref = 0
    foreign_summaries = 0
    foreign_tokens_total = 0
    qwx_tokens_total = 0
    summary_tokens_total = 0
    halluc = 0
    len_ratio_sum = 0.0
    extract_sum = 0.0
    rouge_sum = 0.0
    topic_acc = defaultdict(lambda: {"n": 0, "rouge": 0.0, "halluc": 0, "lenr": 0.0})
    examples_leak = []
    examples_gender_drop = []

    for r in rows:
        pred = get_pred(r)
        ref = get_ref(r)
        art = get_article(r)
        if not pred or not ref:
            continue
        n += 1

        mp, fp = gender_counts(pred)
        mr, fr = gender_counts(ref)
        male_pred += mp; female_pred += fp
        male_ref += mr; female_ref += fr

        nf, nq = english_leak(pred)
        foreign_tokens_total += nf
        qwx_tokens_total += nq
        if nf > 0:
            foreign_summaries += 1
            if len(examples_leak) < 8:
                examples_leak.append({"id": r.get("id"), "prediction": pred[:240]})
        summary_tokens_total += max(1, len(tokens(pred)))

        is_hal = halluc_numbers(art, pred)
        if is_hal:
            halluc += 1
        plen = len(tokens(pred))
        rlen = max(1, len(tokens(ref)))
        len_ratio_sum += plen / rlen
        extract_sum += extractive_overlap(art, pred)
        rg = rouge1_stem_f1(pred, ref)
        rouge_sum += rg

        ma, fa = gender_counts(art)
        if fa >= 1 and fp == 0 and mp >= 1 and len(examples_gender_drop) < 8:
            examples_gender_drop.append({"id": r.get("id"),
                                          "article_female_terms": fa,
                                          "prediction": pred[:240]})

        topic = topic_by_id.get(str(r.get("id")), None)
        if topic:
            ta = topic_acc[topic]
            ta["n"] += 1
            ta["rouge"] += rg
            ta["halluc"] += 1 if is_hal else 0
            ta["lenr"] += plen / rlen

    fpred = female_pred / max(1, male_pred + female_pred)
    fref = female_ref / max(1, male_ref + female_ref)

    topic_summary = {}
    for t, d in topic_acc.items():
        if d["n"] >= 1:
            topic_summary[t] = {
                "n": d["n"],
                "rouge1_stem": round(d["rouge"] / d["n"], 4),
                "halluc_num_rate": round(d["halluc"] / d["n"], 4),
                "mean_len_ratio": round(d["lenr"] / d["n"], 3),
            }
    big = [v["rouge1_stem"] for v in topic_summary.values() if v["n"] >= 20]
    fairness_gap = round(max(big) - min(big), 4) if len(big) >= 2 else None

    return {
        "label": label,
        "n": n,
        "gender": {
            "male_mentions": male_pred,
            "female_mentions": female_pred,
            "female_share_pred": round(fpred, 4),
            "female_share_reference": round(fref, 4),
            "female_share_delta_vs_reference": round(fpred - fref, 4),
        },
        "leakage": {
            "frac_summaries_with_foreign_token": round(foreign_summaries / max(1, n), 4),
            "foreign_tokens_per_1k": round(1000 * foreign_tokens_total / max(1, summary_tokens_total), 3),
            "qwx_tokens_per_1k": round(1000 * qwx_tokens_total / max(1, summary_tokens_total), 3),
        },
        "faithfulness": {
            "halluc_num_rate": round(halluc / max(1, n), 4),
            "mean_len_ratio": round(len_ratio_sum / max(1, n), 3),
            "mean_extractive_overlap": round(extract_sum / max(1, n), 4),
            "mean_rouge1_stem": round(rouge_sum / max(1, n), 4),
        },
        "topic": {"per_topic": topic_summary, "fairness_gap_rouge1_stem": fairness_gap},
        "examples": {"leakage": examples_leak, "gender_drop": examples_gender_drop},
    }


def make_figures(results: dict, fig_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figures: {e}", file=sys.stderr)
        return []

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    systems = list(results["systems"])
    if not systems:
        return []
    labels = [results["systems"][s]["label"] for s in systems]
    fpred = [results["systems"][s]["gender"]["female_share_pred"] for s in systems]
    fref = results["systems"][systems[0]]["gender"]["female_share_reference"]
    leak = [results["systems"][s]["leakage"]["frac_summaries_with_foreign_token"] for s in systems]

    paths = []
    fig, ax = plt.subplots(figsize=(7, 3.4))
    src_share = results.get("source_female_share")
    bars = ax.bar(labels, fpred, color="#4C72B0")
    if src_share is not None:
        ax.axhline(src_share, color="#C44E52", ls="--", lw=1.5,
                   label=f"source articles ({src_share:.3f})")
    ax.axhline(fref, color="#55A868", ls=":", lw=1.5, label=f"human references ({fref:.3f})")
    ax.set_ylabel("Female share of\ngendered mentions")
    ax.set_title("Gender representation in summaries vs. source / references")
    ax.legend(fontsize=8, loc="upper right")
    for b, v in zip(bars, fpred):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    pa = str(Path(fig_dir) / "fig6_bias_gender.png")
    fig.savefig(pa, dpi=150); plt.close(fig); paths.append(pa)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.bar(labels, [100 * x for x in leak], color="#DD8452")
    ax.set_ylabel("% summaries with\n>=1 foreign token")
    ax.set_title("English / foreign-token leakage by system")
    for b, v in zip(bars, leak):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v + 0.3, f"{100*v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    pb = str(Path(fig_dir) / "fig7_bias_leakage.png")
    fig.savefig(pb, dpi=150); plt.close(fig); paths.append(pb)
    return paths


def markdown_summary(results: dict) -> str:
    lines = ["# Bias analysis summary", ""]
    sys_ids = list(results["systems"].keys())
    lines.append("## Gender representation (female share of gendered mentions)")
    if results.get("source_female_share") is not None:
        lines.append(f"- Source articles: **{results['source_female_share']:.3f}**")
    lines.append("")
    lines.append("| System | Female share (pred) | vs reference | Foreign-token % | halluc# | mean len ratio | R1-stem |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in sys_ids:
        d = results["systems"][s]
        lines.append(
            f"| {d['label']} | {d['gender']['female_share_pred']:.3f} | "
            f"{d['gender']['female_share_delta_vs_reference']:+.3f} | "
            f"{100*d['leakage']['frac_summaries_with_foreign_token']:.1f}% | "
            f"{d['faithfulness']['halluc_num_rate']:.3f} | "
            f"{d['faithfulness']['mean_len_ratio']:.2f} | "
            f"{d['faithfulness']['mean_rouge1_stem']:.3f} |"
        )
    if results.get("qual"):
        lines += ["", "## Representational-harm error taxonomy (LLM-judge notes, n=30/system)"]
        lines.append("| System | identity/attribution | fabrication | factual pass | completeness pass |")
        lines.append("|---|---|---|---|---|")
        for s, q in results["qual"].items():
            ax = q.get("axis_pass_rate", {})
            lines.append(
                f"| {s} | {q.get('identity_attribution_errors',0)} | "
                f"{q.get('fabrication_errors',0)} | "
                f"{ax.get('factual_correct',float('nan')):.2f} | "
                f"{ax.get('completeness',float('nan')):.2f} |"
            )
    return "\n".join(lines)


def run(pred_specs, source, qual, out_json, fig_dir, limit):
    topic_by_id = {}
    source_female_share = None
    if source and Path(source).exists():
        m = f = 0
        for r in read_jsonl(source):
            tid = str(r.get("id"))
            topic = r.get("topic")
            if topic:
                topic_by_id[tid] = str(topic)
            ma, fa = gender_counts(get_article(r))
            m += ma; f += fa
        if m + f > 0:
            source_female_share = round(f / (m + f), 4)

    systems = {}
    for spec in pred_specs:
        if "=" not in spec:
            raise ValueError(f"Bad --pred (need label=path): {spec!r}")
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[warn] missing prediction file, skipping: {path}", file=sys.stderr)
            continue
        systems[label] = analyze_system(label, path, topic_by_id, limit=limit)

    results = {
        "systems": systems,
        "source_female_share": source_female_share,
        "n_topics": len(set(topic_by_id.values())),
    }
    if qual and Path(qual).exists():
        results["qual"] = analyze_qual(qual)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    if fig_dir:
        results["_figures"] = make_figures(results, fig_dir)

    print(markdown_summary(results))
    print(f"\n[ok] wrote {out_json}")
    return results


# ----------------------------------------------------------------------------
# Self-test fixture
# ----------------------------------------------------------------------------
def self_test() -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "test.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a1", "article": "Bakan Mehmet Yılmaz açıkladı. Ayşe Demir 25 yaşında.",
                            "reference": "Mehmet Yılmaz ve Ayşe Demir konuştu.", "topic": "politics"}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"id": "a2", "article": "Kadın futbolcu Zeynep gol attı. 3 maç oynandı.",
                            "reference": "Zeynep gol attı.", "topic": "sport"}, ensure_ascii=False) + "\n")
    pred = tmp / "Sx.jsonl"
    with open(pred, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a1", "prediction": "Bakan Mehmet Yılmaz the breaking haberi 99 kez açıkladı.",
                            "reference": "Mehmet Yılmaz ve Ayşe Demir konuştu.", "article": "Bakan Mehmet Yılmaz açıkladı. Ayşe Demir 25 yaşında."}, ensure_ascii=False) + "\n")
        # Sentinel token present: must be stripped so "extra" is NOT counted as
        # a foreign (x-bearing) token. This summary should read as clean Turkish.
        f.write(json.dumps({"id": "a2", "prediction": "<extra_id_0> Zeynep gol attı.",
                            "reference": "Zeynep gol attı.", "article": "Kadın futbolcu Zeynep gol attı. 3 maç oynandı."}, ensure_ascii=False) + "\n")
    out = tmp / "bias.json"
    res = run([f"Sx={pred}"], str(src), None, str(out), None, None)
    d = res["systems"]["Sx"]
    assert d["n"] == 2, d["n"]
    # Only a1 ("the", "breaking") leaks; a2's sentinel must be stripped -> 0.5.
    assert d["leakage"]["frac_summaries_with_foreign_token"] == 0.5, d["leakage"]
    assert d["leakage"]["qwx_tokens_per_1k"] == 0.0, ("sentinel not stripped", d["leakage"])
    assert d["faithfulness"]["halluc_num_rate"] == 0.5, d["faithfulness"]
    assert res["source_female_share"] is not None
    assert len(d["examples"]["gender_drop"]) >= 1, "should flag the dropped-woman case"
    print("\n[self-test] PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc bias/fairness analysis (no retraining).")
    p.add_argument("--pred", action="append", default=[], help="Repeatable: label=path/to/predictions.jsonl")
    p.add_argument("--source", default=None, help="Source test.jsonl with {id, article, topic}.")
    p.add_argument("--qual", default=None, help="qual_labels_filled.csv for note taxonomy.")
    p.add_argument("--out-json", default="outputs/results/bias_results.json")
    p.add_argument("--fig-dir", default="report/figures")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.pred:
        p.error("provide at least one --pred label=path (or use --self-test)")
    run(args.pred, args.source, args.qual, args.out_json, args.fig_dir, args.limit)


if __name__ == "__main__":
    main()
