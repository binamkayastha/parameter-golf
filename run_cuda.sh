#!/bin/bash
# Single-GPU training for RTX 4090 (RunPod)
# Optimized from the default 8xH100 setup:
#   --nproc_per_node=1  (single GPU; grad_accum_steps auto-becomes 8)
#   OMP_NUM_THREADS=1   (prevents CPU thread oversubscription with torchrun)

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 train_gpt.py "$@"
