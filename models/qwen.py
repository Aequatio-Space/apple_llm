from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000
    rope_traditional: bool = False


class QwenAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, 1, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, 1, -1).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        attn_weights = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            attn_weights += mask
        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(attn_weights.dtype)

        out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class QwenBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = QwenAttention(args)
        self.mlp = SwiGLU(args.hidden_size, args.intermediate_size)
        self.norm1 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm2 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        a, cache = self.attn(self.norm1(x), mask, cache)
        x = x + a
        m = self.mlp(self.norm2(x))
        return x + m, cache


class QwenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.blocks = [QwenBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, attention_mask=None, cache=None):
        x = self.embed_tokens(x)

        if attention_mask is not None:
            attention_mask = 1 - attention_mask
            attention_mask *= -1e9
            attention_mask = attention_mask.astype(x.dtype)

        if x.shape[1] > 1 and attention_mask is None:
            attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            attention_mask = attention_mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        for i, blk in enumerate(self.blocks):
            x, cache[i] = blk(x, attention_mask, cache[i])

        return self.norm(x), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = QwenModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.value_head = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, input_ids, attention_mask=None, cache=None):
        hidden, cache = self.model(input_ids, attention_mask, cache)
        return self.lm_head(hidden), cache, self.value_head(hidden).squeeze(-1)

    def generate(self, x, temp=1.0):
        def sample(logits):
            return mx.argmax(logits, axis=-1) if temp == 0 else mx.random.categorical(logits / temp)

        cache = []
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.model.embed_tokens.weight.dtype)

        x = self.model.embed_tokens(x)
        for blk in self.model.blocks:
            x, c = blk(x, mask)
            cache.append(c)

        x = self.model.norm(x)
        y = sample(self.lm_head(x[:, -1]))
        yield y

        while True:
            x = y[:, None]
            x = self.model.embed_tokens(x)
            for i, blk in enumerate(self.model.blocks):
                x, cache[i] = blk(x, cache=cache[i])
            x = self.model.norm(x)
            y = sample(self.lm_head(x[:, -1]))
            yield y
