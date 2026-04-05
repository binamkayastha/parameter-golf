#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {primary_4090|fallback_4090} [extra env assignments or command args via environment]"
  exit 1
fi

preset="$1"
shift || true

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-400}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"

case "$preset" in
  primary_4090)
    export RUN_ID="${RUN_ID:-primary_4090}"
    export MLP_ACT="${MLP_ACT:-swiglu}"
    export MLP_HIDDEN="${MLP_HIDDEN:-640}"
    ;;
  fallback_4090)
    export RUN_ID="${RUN_ID:-fallback_4090}"
    export MLP_ACT="${MLP_ACT:-relu2}"
    export MLP_HIDDEN="${MLP_HIDDEN:-0}"
    ;;
  *)
    echo "unknown preset: $preset"
    echo "expected one of: primary_4090, fallback_4090"
    exit 1
    ;;
esac

exec torchrun --standalone --nproc_per_node=1 train_gpt.py "$@"
