# 1x4090 Improvement Plan

## Objective

Improve top-level `train_gpt.py` on a single RTX 4090, prioritizing lower final `val_loss` first and keeping any `val_bpb` gain that comes with it, while staying under the 16MB artifact cap.

## Hardware And Constraints

- GPU: 1x NVIDIA GeForce RTX 4090, 24GB VRAM
- Training target: single-GPU `torchrun --standalone --nproc_per_node=1`
- Artifact target: under 16,000,000 total bytes including code
- Active script: root `train_gpt.py`
- Current repo caveat: default SP1024 dataset/tokenizer paths are not present locally yet, so full planned runs are blocked until data is available

## Baseline

- Baseline code path: root `train_gpt.py` with ReLU-squared MLP, derived grad accumulation, no EMA
- Repo notes indicate a 1-GPU baseline around `val_bpb ~= 1.405` in `JOURNAL.md`
- Current implementation now supports:
  - `GRAD_ACCUM_STEPS`
  - `MLP_HIDDEN`
  - `MLP_ACT={relu2,swiglu}`
  - `EMA_ENABLED`
  - `EMA_DECAY`

## Primary Approach

Use a throughput-oriented 1-GPU configuration plus a modest MLP change:

```bash
RUN_ID=primary_4090 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=131072 \
GRAD_ACCUM_STEPS=2 \
MLP_ACT=swiglu \
MLP_HIDDEN=640 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
VAL_LOSS_EVERY=400 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Fallback Approach

Keep the throughput change and EMA, but remove the activation/hidden-size risk:

```bash
RUN_ID=fallback_4090 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=131072 \
GRAD_ACCUM_STEPS=2 \
MLP_ACT=relu2 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
VAL_LOSS_EVERY=400 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Rejected / Defer

Do not spend this 2-hour window on:

- XSA
- bigram hash
- smear gate
- longer context
- recurrence
- weight decay changes
- quantization-format rewrites
- Hopper-only / FlashAttention-3 paths

## Run Log

| Date | Run ID | Config Summary | final val_loss | final val_bpb | Artifact Bytes | Peak Memory | Steps | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-05 | code_update_only | Added grad accum override, configurable MLP act/hidden, EMA, launcher, plan tracker | pending | pending | pending | pending | pending | waiting on local dataset/tokenizer |

## Decision Rules

- Keep a change only if final `val_loss` improves and the artifact remains legal.
- Prefer the primary branch if it is stable and under budget, even if the gain is modest.
- If the primary branch regresses or blows the size cap, revert only the MLP change and rerun the fallback branch.
- If both branches regress, stop adding architecture features and search only around batch size and accumulation.

## Next Actions

- [x] Add configurable grad accumulation to root `train_gpt.py`
- [x] Add `relu2` / `swiglu` MLP switch
- [x] Add optional EMA for final eval/export
- [x] Add `run_cuda.sh` with `primary_4090` and `fallback_4090`
- [x] Create `PLAN.md`
- [ ] Restore or download the SP1024 dataset locally
- [ ] Restore or download the SP1024 tokenizer locally
- [ ] Smoke-test `run_cuda.sh`
- [ ] Run `primary_4090`
- [ ] Compare final metrics and artifact size
- [ ] Run `fallback_4090` if needed
