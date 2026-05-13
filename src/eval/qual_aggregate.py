"""Aggregate hand-labeled qual-analysis CSV into per-system summary stats.

Reads the labeled CSV produced by qual_export.py (after the user fills in the
five binary columns) and emits:
    outputs/results/qual_summary.json   — per-system mean and per-axis breakdown
    outputs/results/qual_summary.md     — Markdown table ready for report

Five axes scored 0/1:
    factual_correct, completeness, fluency, morpho_correct, no_mode_collapse
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.utils.logging import get_logger

LOG = get_logger("eval.qual_aggregate")

AXES = ["factual_correct", "completeness", "fluency", "morpho_correct", "no_mode_collapse"]


def _parse_label(x: str) -> int | None:
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    try:
        v = int(float(x))
        if v in (0, 1):
            return v
    except ValueError:
        pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate hand-labeled qual-analysis CSV.")
    p.add_argument("--input", required=True, help="Labeled CSV from qual_export.py output.")
    p.add_argument("--out-json", default="outputs/results/qual_summary.json")
    p.add_argument("--out-md", default="outputs/results/qual_summary.md")
    args = p.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    LOG.info("Loaded %d rows from %s", len(rows), args.input)

    # Group by system
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sys[row["system"]].append(row)

    summary: dict[str, dict] = {}
    n_labeled = 0
    for sys_label, sys_rows in by_sys.items():
        axes_means: dict[str, float | None] = {}
        per_axis_counts: dict[str, dict[str, int]] = {}
        for axis in AXES:
            labels = [_parse_label(r.get(axis, "")) for r in sys_rows]
            valid = [v for v in labels if v is not None]
            n_labeled += len(valid)
            n_pos = sum(valid)
            mean = (n_pos / len(valid)) if valid else None
            axes_means[axis] = mean
            per_axis_counts[axis] = {"n_labeled": len(valid), "n_pass": n_pos, "n_fail": len(valid) - n_pos}

        # Overall pass-rate: mean across the five axes (uniform weights)
        valid_means = [v for v in axes_means.values() if v is not None]
        overall = statistics.mean(valid_means) if valid_means else None

        summary[sys_label] = {
            "n_examples_labeled": len(sys_rows),
            "axes_pass_rate": axes_means,
            "axes_counts": per_axis_counts,
            "overall_pass_rate": overall,
        }

    LOG.info("Total binary labels parsed: %d", n_labeled)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOG.info("Wrote %s", args.out_json)

    # Markdown table
    md_lines = ["# Qualitative analysis summary", "",
                "Pass rates per system across 5 axes, 30 articles × 6 systems = 180 labeled rows.",
                ""]
    md_lines.append("| System | Factual | Complete | Fluency | Morpho | NoCollapse | Overall |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sys_label in ["B1", "B2", "B3a", "B3b", "S-gpt", "S-claude"]:
        if sys_label not in summary:
            continue
        s = summary[sys_label]
        cells = [sys_label]
        for axis in AXES:
            v = s["axes_pass_rate"].get(axis)
            cells.append("—" if v is None else f"{v:.2f}")
        ov = s["overall_pass_rate"]
        cells.append("—" if ov is None else f"**{ov:.2f}**")
        md_lines.append("| " + " | ".join(cells) + " |")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    LOG.info("Wrote %s", args.out_md)


if __name__ == "__main__":
    main()
