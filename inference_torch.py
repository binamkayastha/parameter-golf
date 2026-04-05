#!/usr/bin/env python3
"""Simple inference script for PyTorch GPT models (train_gpt.py)."""
import argparse
import io
import os
import sys
import zlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm
from pathlib import Path

def dequantize_state_dict_int8(obj: dict) -> dict:
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def load_model_torch(model_path: str, args):
    model_path = Path(model_path)
    
    if model_path.suffix == ".ptz":
        with open(model_path, "rb") as f:
            quant_blob = f.read()
        quant_raw = zlib.decompress(quant_blob)
        quant_obj = torch.load(io.BytesIO(quant_raw), map_location="cpu", weights_only=False)
        state_dict = dequantize_state_dict_int8(quant_obj)
    elif model_path.suffix == ".pt":
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")
    
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))


# Minimal replication of GPT architecture from train_gpt.py
class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim, dtype=torch.float32), torch.zeros(dim, dtype=torch.float32)]))
    
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale[None, None, :] * attn_out
        x = x + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * qk_gain_init)
        self.rope = RotaryEmbedding(self.head_dim, rope_base)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q)
        k = self.rope(k)
        q = q * self.q_gain[None, :, None, None]
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.proj(x * x)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat([-x2, x1], dim=-1) * sin + torch.cat([x1, x2], dim=-1) * cos


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
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
    
    def generate_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, None)


def sample_top_p(logits: torch.Tensor, p: float = 0.9) -> int:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= p
    keep_indices = sorted_indices[mask]
    if keep_indices.numel() == 0:
        keep_indices = sorted_indices[:1]
        sorted_probs = sorted_probs[:1]
    else:
        sorted_probs = sorted_probs[mask]
    sorted_probs = sorted_probs / sorted_probs.sum()
    chosen_idx = torch.multinomial(sorted_probs.float(), 1).item()
    return keep_indices[chosen_idx].item()


def generate(model, sp, prompt: str, max_tokens: int = 100, temp: float = 0.0, top_p: float = 0.0):
    tokens = np.array(sp.encode(prompt), dtype=np.int64)
    x = torch.tensor(tokens.reshape(1, -1), dtype=torch.long)
    
    for _ in range(max_tokens):
        logits = model.generate_logits(x)
        next_token_logits = logits[0, -1]
        
        if temp > 0:
            next_token_logits = next_token_logits / temp
        
        if top_p > 0:
            next_token = sample_top_p(next_token_logits, top_p)
        elif temp > 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs).item()
        else:
            next_token = torch.argmax(next_token_logits).item()
        
        if next_token == sp.eos_id():
            break
        
        x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
    
    return sp.decode(x[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on PyTorch GPT model")
    parser.add_argument("model", help="Path to model (.pt or .int8.ptz)")
    parser.add_argument("--prompt", "-p", default="The apple is", help="Prompt text")
    parser.add_argument("--max-tokens", "-n", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", "-t", type=float, default=0.0, help="Temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling threshold (0 = disabled)")
    args = parser.parse_args()
    
    tokenizer_path = Hyperparameters().tokenizer_path
    print(f"Loading model from {args.model}...")
    model = load_model_torch(args.model, Hyperparameters())
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    print(f"Prompt: {args.prompt}")
    print("-" * 40)
    output = generate(model, sp, args.prompt, args.max_tokens, args.temp, args.top_p)
    print(output)
    print("-" * 40)
