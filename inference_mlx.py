#!/usr/bin/env python3
"""Simple inference script for MLX GPT models."""
import argparse
import pickle
import zlib
import json
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from pathlib import Path
from train_gpt_mlx import (
    GPT, Hyperparameters, tree_unflatten, tree_flatten,
    dequantize_state_dict_int8
)

def load_model(model_path: str, tokenizer_path: str):
    args = Hyperparameters()
    
    model_path = Path(model_path)
    if model_path.suffix == ".ptz":
        with open(model_path, "rb") as f:
            quant_blob = f.read()
        quant_obj = pickle.loads(zlib.decompress(quant_blob))
        flat_state = dequantize_state_dict_int8(quant_obj)
    elif model_path.suffix == ".npz":
        flat_state = mx.load(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")
    
    flat_state_keys = set(flat_state.keys())
    
    has_smear_gate = any("smear_proj" in k for k in flat_state_keys)
    has_bigram_hash = any("bigram_hash" in k for k in flat_state_keys)
    has_partial_rope = any("rope_partial" in k for k in flat_state_keys)
    has_value_residual = any("v_residual" in k for k in flat_state_keys)
    
    bigram_hash_buckets = args.bigram_hash_buckets
    if has_bigram_hash and bigram_hash_buckets == 0:
        bigram_hash_buckets = 4096
    
    partial_rope_dims = args.partial_rope_dims
    if has_partial_rope and partial_rope_dims == 0:
        partial_rope_dims = 16
    
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        smear_gate=has_smear_gate or args.smear_gate,
        bigram_hash_buckets=bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
        partial_rope_dims=partial_rope_dims,
        ln_scale=args.ln_scale,
        value_residual=has_value_residual or args.value_residual,
    )
    
    model.update(tree_unflatten(list(flat_state.items())))
    model.eval()
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return model, sp, args

def sample_top_p(logits: mx.array, p: float = 0.9) -> int:
    probs = mx.softmax(logits, axis=-1)
    probs_np = np.array(probs.tolist(), dtype=np.float32)
    sorted_indices = np.argsort(probs_np)[::-1]
    sorted_probs = probs_np[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    mask = cumsum <= p
    keep_indices = sorted_indices[mask]
    if len(keep_indices) == 0:
        keep_indices = sorted_indices[:1]
    keep_probs = sorted_probs[keep_indices]
    keep_probs = keep_probs / keep_probs.sum()
    chosen = np.random.choice(len(keep_indices), p=keep_probs)
    return int(keep_indices[chosen])

def generate(model, sp, prompt: str, max_tokens: int = 100, temp: float = 0.0, top_p: float = 0.0):
    tokens = np.array(sp.encode(prompt), dtype=np.int32)
    x = mx.array(tokens.reshape(1, -1), dtype=mx.int32)
    
    for _ in range(max_tokens):
        hidden = model(x)
        logits_proj = hidden @ model.tok_emb.weight.astype(hidden.dtype).T
        logits = model.softcap(logits_proj)
        next_token_logits = logits[0, -1]
        
        if temp > 0:
            next_token_logits = next_token_logits / temp
        
        if top_p > 0:
            next_token = sample_top_p(next_token_logits, top_p)
        elif temp > 0:
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = int(mx.argmax(probs).item())
        else:
            next_token = int(mx.argmax(next_token_logits).item())
        
        if next_token == sp.eos_id():
            break
        
        x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)
    
    return sp.decode(x[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on MLX GPT model")
    parser.add_argument("model", help="Path to model (.npz or .int8.ptz)")
    parser.add_argument("--prompt", "-p", default="The apple is", help="Prompt text")
    parser.add_argument("--max-tokens", "-n", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", "-t", type=float, default=0.0, help="Temperature (0 = greedy)")
    parser.add_argument("--top-p", "-pP", type=float, default=0.0, help="Top-p sampling threshold (0 = disabled)")
    args = parser.parse_args()
    
    tokenizer_path = Hyperparameters().tokenizer_path
    print(f"Loading model from {args.model}...")
    model, sp, _ = load_model(args.model, tokenizer_path)
    
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model: {n_params:,} parameters")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)
    output = generate(model, sp, args.prompt, args.max_tokens, args.temp, args.top_p)
    print(output)
    print("-" * 40)
