#!/usr/bin/env bash
# Usage: scripts/02_generate_teacher.sh <openai|anthropic> <concise|detailed> <N>
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER="${1:-openai}"
PROMPT="${2:-concise}"
N="${3:-10000}"
python -m src.teachers.generate \
  --teacher "$TEACHER" \
  --prompt-variant "$PROMPT" \
  --input data/raw/mlsum_tr/train.jsonl \
  --n "$N" \
  --out-dir "data/synthetic/${TEACHER}/${PROMPT}"
