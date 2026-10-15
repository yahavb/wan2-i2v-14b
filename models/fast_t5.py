"""Fullgraph UMT5-XXL text encoder on Neuron, tensor-parallel over every rank.

Port of staging vllm-omni-neuron's umt5_encoder (TP over the world group, head-sharded
relative position bias, one compiled graph) onto the original Wan T5Encoder in
wan/modules/t5.py:

  * TP over the world: 64 heads / tp per rank. q/k/v and the FFN fc1/gate are
    column-parallel, o and fc2 row-parallel with in-graph all-reduces (funcol).
  * Relative position bias: text length is fixed (512), so each block's bias for its
    local heads is a constant [1, n_local, L, L] built once on CPU (no bucket math or
    embedding gather in the graph).
  * Each block is torch.compile(fullgraph=True, dynamic=False); all 24 share one NEFF.
    Both prompts (cond, null) run as batch 2.
  * The 256k-row token embedding (2 GB bf16) stays on CPU: it is a 512-row lookup.

Output matches T5EncoderModel + the caller's zero padding: positions past each prompt's
length are zeroed.
"""

import logging

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

logger = logging.getLogger(__name__)
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)


def _rms(x, w, eps):
    x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return w * x.type_as(w)


def _gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0))))


class _Block(nn.Module):
    def __init__(self, blk, rank, tp, pos_bias, device, group):
        super().__init__()
        a, f = blk.attn, blk.ffn
        n = a.num_heads // tp
        da, df = a.dim_attn // tp, f.dim_ffn // tp
        r0, f0 = rank * da, rank * df
        P = lambda t: nn.Parameter(t.detach().clone().contiguous().to(device), requires_grad=False)
        self.n, self.c, self.eps = n, a.head_dim, blk.norm1.eps
        self.group = group
        self.norm1, self.norm2 = P(blk.norm1.weight), P(blk.norm2.weight)
        self.q, self.k, self.v = (P(m.weight[r0:r0 + da]) for m in (a.q, a.k, a.v))
        self.o = P(a.o.weight[:, r0:r0 + da])
        self.fc1, self.gate = P(f.fc1.weight[f0:f0 + df]), P(f.gate[0].weight[f0:f0 + df])
        self.fc2 = P(f.fc2.weight[:, f0:f0 + df])
        self.register_buffer("pos_bias", pos_bias.to(device), persistent=False)

    def _ar(self, x):
        return funcol.all_reduce(x, "sum", self.group) if self.group is not None else x

    def forward(self, x, mask_bias):           # x [B, L, C] bf16, mask_bias [B, 1, 1, L] bf16
        B, L, _ = x.shape
        h = _rms(x, self.norm1, self.eps)
        q = (h @ self.q.t()).view(B, L, self.n, self.c)
        k = (h @ self.k.t()).view(B, L, self.n, self.c)
        v = (h @ self.v.t()).view(B, L, self.n, self.c)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + (self.pos_bias + mask_bias)  # T5: no scaling
        attn = torch.softmax(attn.float(), dim=-1).type_as(x)
        y = torch.einsum("bnij,bjnc->binc", attn, v).reshape(B, L, self.n * self.c)
        x = x + self._ar(y @ self.o.t())
        h = _rms(x, self.norm2, self.eps)
        y = (h @ self.fc1.t()) * _gelu(h @ self.gate.t())
        return x + self._ar(y @ self.fc2.t())


class _Final(nn.Module):
    def __init__(self, norm, device):
        super().__init__()
        self.w = nn.Parameter(norm.weight.detach().clone().to(device), requires_grad=False)
        self.eps = norm.eps

    def forward(self, x, keep):                 # keep [B, L, 1]: 1 for real tokens, 0 for padding
        return _rms(x, self.w, self.eps) * keep


class FastT5:
    """encode([prompt, n_prompt]) -> [2, text_len, 4096] bf16 on device, zero-padded."""

    def __init__(self, t5_model, device, dtype=torch.bfloat16):   # t5_model: T5EncoderModel on CPU
        self.tok, self.text_len, self.device, self.dtype = t5_model.tokenizer, t5_model.text_len, device, dtype
        enc = t5_model.model
        self.embedding = enc.token_embedding    # stays on CPU
        tp = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        group = dist.group.WORLD if tp > 1 else None
        L = self.text_len
        blocks = []
        for blk in enc.blocks:
            pe = blk.pos_embedding               # umt5: per-block (shared_pos=False)
            rel = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
            buckets = pe._relative_position_bucket(rel)
            bias = pe.embedding.weight.detach()[buckets]          # [L, L, 64]
            n = pe.num_heads // tp
            bias = bias[:, :, rank * n:(rank + 1) * n].permute(2, 0, 1).unsqueeze(0)  # [1, n, L, L]
            blocks.append(_Block(blk, rank, tp, bias.to(dtype).contiguous(), device, group))
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        self.blocks = [torch.compile(b, **COMPILE_KW) for b in blocks]
        self.final = torch.compile(_Final(enc.norm, device), **COMPILE_KW)
        logger.info(f"[fast_t5] TP={tp} heads/rank={enc.num_heads // tp} layers={len(blocks)} fullgraph=True")

    def encode(self, texts):
        ids, mask = self.tok(texts, return_mask=True, add_special_tokens=True)   # [B, L] CPU
        dt = self.dtype
        x = self.embedding(ids).to(dt).to(self.device)
        mask_bias = ((1 - mask.float()) * torch.finfo(dt).min).to(dt).view(len(texts), 1, 1, -1).to(self.device)
        keep = mask.to(dt).unsqueeze(-1).to(self.device)
        for b in self.blocks:
            x = b(x, mask_bias)
        return self.final(x, keep)
