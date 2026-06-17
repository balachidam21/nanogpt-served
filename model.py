"""
A tiny GPT — char-level, single-head attention.

This is Project Zero's model. It is deliberately the smallest thing that learns:
single-head causal self-attention (the softmax(QKᵀ/√d)·V block you derived on
paper), a feed-forward MLP, residual connections, and pre-norm LayerNorms.

Multi-head attention and the KV cache come LATER, on-demand:
  - multi-head: a refactor of CausalSelfAttention once single-head trains.
  - KV cache:   Sub-project B — see the naive `generate()` at the bottom, which
                recomputes the whole context every step (the waste you spotted).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # number of distinct characters (set from the data)
    block_size: int = 128     # max context length — the T in (B, T, C)
    n_embd: int = 128         # embedding / channel dim — the C
    n_layer: int = 4          # number of transformer blocks stacked
    n_head: int = 1           # single-head for now; multi-head arrives on-demand
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention: the softmax(QKᵀ/√d)·V block, run in
    parallel across n_head heads each of width head_dim = n_embd // n_head."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must split evenly into n_head heads"
        self.n_head = config.n_head
        # One Linear of width n_embd produces ALL heads' projections at once;
        # we slice it into heads inside forward(). (Q, K, V each a projection of x.)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # output projection
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular causal mask (1 = allowed, 0 = blocked). Registered as a
        # buffer so it moves with .to(device) but isn't a learnable parameter.
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        # KV cache — None during training; filled during cached generation.
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape          # batch, time (tokens), channels (n_embd)
        H = self.n_head            # number of heads (4)
        d = C // H                 # head dim (32 when C=128, H=4)
        q = self.query(x).view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
        k = self.key(x).view(B, T, H, d).transpose(1, 2)     # (B, H, T, d)
        v = self.value(x).view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)

        # KV-cache append (implemented Jun 7) — the "append one row per step" from June 3.
        # When use_cache is True:
        #   1. if a cache already exists (self.cache_k is not None), CONCAT the past
        #      k, v with the new k, v along the TIME axis (dim=2).
        #        hint: k = torch.cat([self.cache_k, k], dim=2)   # (B, H, T_past+T, d)
        #      do the same for v.
        #   2. then STORE the (now-full) k, v back into self.cache_k / self.cache_v
        #      so the next decode step can extend them.
        #   q is NOT cached — one query per step, used once (the no-Q-cache asymmetry).
        #   When use_cache is False (training), leave k, v untouched.

        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim=2)
            v = torch.cat([self.cache_v, v], dim=2)

        if use_cache:
            self.cache_k, self.cache_v = k, v

        Tk = k.size(2)             # total keys/values now = T_past + T
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)    # (B, H, T, Tk)
        if T == Tk:                # prefill / training: square → needs the causal mask
            scores = scores.masked_fill(self.mask[:T, :Tk] == 0, float('-inf'))
        # else: decode step (T==1, Tk>1) — the single new query may attend to ALL
        # cached keys (they are all its past), so no mask. Masking was the LICENSE
        # for the cache: the past is frozen, so cached k, v never go stale.
        weights = F.softmax(scores, dim=-1)                  # (B, H, T, Tk)
        weights = self.attn_dropout(weights)
        y = weights @ v                                      # (B, H, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)     # merge heads → (B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back."""

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache=use_cache)   # residual around attention
        x = x + self.mlp(self.ln2(x))                         # residual around feed-forward
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, use_cache=False, pos_offset=0):
        B, T = idx.shape
        # absolute positions — with a KV cache we feed tokens starting at pos_offset,
        # NOT from 0, so the positional embedding must be indexed from the real offset.
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)   # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # (B, T, C); pos broadcasts
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Naive autoregressive generation — recomputes the FULL context every
        step (no KV cache). This is the O(n²)-per-step waste Sub-project B fixes."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]   # crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on last step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def reset_caches(self):
        """Clear every layer's KV cache (call once before cached generation)."""
        for block in self.blocks:
            block.attn.reset_cache()

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, temperature=1.0):
        """KV-cached generation. PREFILL the prompt once (filling every layer's
        cache), then DECODE one token at a time — each step feeds ONLY the new token;
        the cache supplies all the past. Compare wall-clock vs naive generate().
        NOTE: total length must stay <= block_size (this toy has no cache eviction;
        real servers use a rolling / paged cache)."""
        self.reset_caches()
        pos = idx.size(1)                                       # next absolute position
        logits, _ = self(idx, use_cache=True, pos_offset=0)     # PREFILL the whole prompt
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature             # last step's logits
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            logits, _ = self(next_id, use_cache=True, pos_offset=pos)  # DECODE: 1 token
            pos += 1
        return idx
