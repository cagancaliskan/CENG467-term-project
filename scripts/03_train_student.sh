#!/usr/bin/env bash
# Train one student variant. Forwards all flags after -- to train.py.
# Example:
#   scripts/03_train_student.sh \
#       --teacher openai --prompt concise --size 10000 --lora-rank 8
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=openai
PROMPT=concise
SIZE=10000
LORA_RANK=8
RUN_NAME=""

# Lightweight flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher) TEACHER="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --lora-rank) LORA_RANK="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --) shift; break;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${TEACHER}_${PROMPT}_n${SIZE}_r${LORA_RANK}"
fi

PROC_DIR="data/processed/${RUN_NAME}"
CKPT_DIR="outputs/checkpoints/${RUN_NAME}"

python -m src.data.prepare_synthetic \
  --teacher "$TEACHER" \
  --prompt-variant "$PROMPT" \
  --size "$SIZE" \
  --out-dir "$PROC_DIR"

python -m src.student.train \
  --train-file "$PROC_DIR/train.jsonl" \
  --val-file "$PROC_DIR/validation.jsonl" \
  --output-dir "$CKPT_DIR" \
  --lora-rank "$LORA_RANK" \
  "$@"
