#!/usr/bin/env python3
"""Fresh inference script that directly uses the saved model."""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

# =============================================================================
# Model architecture copied from train_gpt.py
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            scale=self.scale,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim, dtype=torch.float32), torch.zeros(dim, dtype=torch.float32)]))

    def forward(self, x, x0):
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale[None, None, :] * attn_out
        x = x + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids, target_ids=None):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))

        if target_ids is not None:
            targets = target_ids.reshape(-1)
            if self.tie_embeddings:
                logits_proj = F.linear(x, self.tok_emb.weight)
            else:
                logits_proj = self.lm_head(x)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            return F.cross_entropy(logits.float(), targets, reduction="mean")
        else:
            if self.tie_embeddings:
                logits_proj = F.linear(x, self.tok_emb.weight)
            else:
                logits_proj = self.lm_head(x)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .pt model file")
    parser.add_argument("-p", "--prompt", default="The apple is", help="Prompt text")
    parser.add_argument("-n", "--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("-t", "--temp", type=float, default=0.0, help="Temperature (0 = greedy)")
    args = parser.parse_args()

    # Hyperparameters (must match what was used to train)
    vocab_size = 1024
    num_layers = 9
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 2
    tie_embeddings = True
    tied_embed_init_std = 0.005
    logit_softcap = 30.0
    rope_base = 10000.0
    qk_gain_init = 1.5
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"

    print(f"Loading model from {args.model}...")
    state_dict = torch.load(args.model, map_location="cpu", weights_only=False)
    
    print(f"Loaded {len(state_dict)} state dict keys")
    print(f"Sample keys: {list(state_dict.keys())[:5]}")

    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=tie_embeddings,
        tied_embed_init_std=tied_embed_init_std,
        logit_softcap=logit_softcap,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
    )

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)
    
    tokens = np.array(sp.encode(args.prompt), dtype=np.int64)
    x = torch.tensor(tokens.reshape(1, -1), dtype=torch.long)
    print(f"Input tokens: {tokens}")
    
    with torch.no_grad():
        for i in range(args.max_tokens):
            logits = model(x)
            next_logits = logits[-1]
            next_token = torch.argmax(next_logits).item()
            
            print(f"Step {i}: token={next_token}, piece='{sp.id_to_piece([next_token])}'")
            
            if next_token == sp.eos_id():
                break
            
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            
            if i >= 9:
                break
    
    output = sp.decode(x[0].tolist())
    print("-" * 40)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
