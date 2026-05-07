#!/usr/bin/env bash
# Convenience runner. Assumes scripts/01 and scripts/02 have been run for both
# teachers + both prompt variants.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick the stronger teacher here once you know it. Default: openai.
TEACHER="${TEACHER:-openai}"

# Size ablation (rank=8, prompt=concise)
for N in 1000 5000 10000; do
  bash scripts/03_train_student.sh \
    --teacher "$TEACHER" --prompt concise --size "$N" --lora-rank 8 \
    --run-name "ablation_size/${TEACHER}_n${N}"
done

# LoRA-rank ablation (size=10k, prompt=concise)
for R in 4 8 16 32; do
  bash scripts/03_train_student.sh \
    --teacher "$TEACHER" --prompt concise --size 10000 --lora-rank "$R" \
    --run-name "ablation_lora/${TEACHER}_r${R}"
done

# Prompt ablation (size=1k for fairness with detailed-only smaller cache)
for P in concise detailed; do
  bash scripts/03_train_student.sh \
    --teacher "$TEACHER" --prompt "$P" --size 1000 --lora-rank 8 \
    --run-name "ablation_prompt/${TEACHER}_${P}_n1k"
done
