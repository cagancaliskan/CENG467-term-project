#!/usr/bin/env bash
# Download and clean MLSUM-TR + TR-News.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.load_mlsum --out-dir data/raw/mlsum_tr "$@"
python -m src.data.load_trnews --out data/raw/trnews/test.jsonl --n 1000 || \
  echo "TR-News HF mirrors unavailable — pass --local-csv when you have the CSV."
