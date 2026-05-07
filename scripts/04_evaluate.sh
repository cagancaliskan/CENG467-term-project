#!/usr/bin/env bash
# Run inference for every system on the test set, then score them all.
# Assumes you've trained the relevant checkpoints. Skips inference if the
# predictions JSONL already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

TEST="data/raw/mlsum_tr/test.jsonl"
mkdir -p outputs/predictions outputs/results

run_student() {
  local label="$1"
  local model_dir="$2"
  local out="outputs/predictions/${label}.jsonl"
  if [[ -f "$out" ]]; then echo "[skip] $label (exists)"; return; fi
  echo "[infer] $label"
  python -m src.student.infer \
    --model-path "$model_dir" \
    --input "$TEST" \
    --out "$out"
}

run_teacher() {
  local label="$1"
  local provider="$2"
  local out="outputs/predictions/${label}.jsonl"
  if [[ -f "$out" ]]; then echo "[skip] $label (exists)"; return; fi
  echo "[infer] $label"
  python -m src.student.infer_teacher \
    --teacher "$provider" \
    --prompt-variant concise \
    --input "$TEST" \
    --out "$out"
}

# Baselines
run_student B1_zeroshot google/mt5-small
run_teacher B3a_gpt openai
run_teacher B3b_claude anthropic
run_student B2_human outputs/checkpoints/human_concise_n10000_r8/final
# Synthetic
run_student S_gpt outputs/checkpoints/openai_concise_n10000_r8/final
run_student S_claude outputs/checkpoints/anthropic_concise_n10000_r8/final

PRED_ARGS=(
  --pred "B1=outputs/predictions/B1_zeroshot.jsonl"
  --pred "B2=outputs/predictions/B2_human.jsonl"
  --pred "B3a=outputs/predictions/B3a_gpt.jsonl"
  --pred "B3b=outputs/predictions/B3b_claude.jsonl"
  --pred "S-gpt=outputs/predictions/S_gpt.jsonl"
  --pred "S-claude=outputs/predictions/S_claude.jsonl"
)

python -m src.eval.run_eval \
  "${PRED_ARGS[@]}" \
  --out-json outputs/results/main.json \
  --out-jsonl outputs/results/main_per_example.jsonl
