#!/usr/bin/env python3
"""Simple inference script for MLX GPT models."""
import argparse
import sys
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece as spm
from pathlib import Path
from train_gpt_mlx import GPT, Hyperparameters, tree_unflatten, tree_flatten

def load_model(model_path: str, tokenizer_path: str):
    args = Hyperparameters()
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
    )
    
    flat_state = mx.load(model_path)
    model.update(tree_unflatten(list(flat_state.items())))
    model.eval()
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return model, sp, args

def generate(model, sp, prompt: str, max_tokens: int = 100, temp: float = 0.0):
    tokens = np.array(sp.encode(prompt), dtype=np.int32)
    x = mx.array(tokens.reshape(1, -1), dtype=mx.int32)
    
    for _ in range(max_tokens):
        hidden = model(x)
        logits_proj = hidden @ model.tok_emb.weight.astype(hidden.dtype).T
        logits = model.softcap(logits_proj)
        next_token_logits = logits[0, -1]
        
        if temp > 0:
            next_token_logits = next_token_logits - mx.max(next_token_logits)
            exp_logits = mx.exp(next_token_logits / temp)
            probs = exp_logits / mx.sum(exp_logits)
            next_token = mx.argmax(probs).item()
        else:
            next_token = int(mx.argmax(next_token_logits).item())
        
        if next_token == sp.eos_id():
            break
        
        x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)
    
    return sp.decode(x[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on MLX GPT model")
    parser.add_argument("model", help="Path to model .npz file")
    parser.add_argument("--prompt", "-p", default="The apple is", help="Prompt text")
    parser.add_argument("--max-tokens", "-n", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", "-t", type=float, default=0.0, help="Temperature (0 = greedy)")
    args = parser.parse_args()
    
    tokenizer_path = Hyperparameters().tokenizer_path
    print(f"Loading model from {args.model}...")
    model, sp, _ = load_model(args.model, tokenizer_path)
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)
    output = generate(model, sp, args.prompt, args.max_tokens, args.temp)
    print(output)
    print("-" * 40)
