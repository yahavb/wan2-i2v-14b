"""Fullgraph DiT for Wan2.2-I2V-A14B on Neuron (TP only, SP=1, both experts resident).

Port of ~/wan2-ti2v-5b/models/fast_dit.py, with the attention core from staging
vllm-omni-neuron (nkilib attention_cte for self AND cross attention, as staging runs
at CP=1). Every compiled unit uses torch.compile(fullgraph=True, dynamic=False):

  _FastTime   time_embedding + time_projection (once per step; I2V timestep is uniform
              over tokens, so e is [1, 1, 6, C] instead of the per-token [1, L, 6, C])
  _FastPatch  concat(latent, y) -> patch embedding -> [2, L, C] (batched CFG)
  _FastText   text embedding (once per prompt)
  _FastCrossKV  each block's cross-attn K (normed) / V (once per prompt per expert)
  _FastBlock  one WanAttentionBlock: norms, modulation, QKV, QK-RMS, RoPE, self-attn,
              O-proj, cross-attn, FFN, with the TP all-reduces in-graph (funcol)
  _FastHead   head + unpatchify + CFG combine (guide scale per expert)

FAST_ATTN_IMPL: cte (default, nkilib attention_cte) | sdpa (F.scaled_dot_product_attention).
FAST_KERNEL_WRAP: hop (default, torch_neuronx.nki_hop.wrap_nki) | op (torch_neuronx.nki_op).
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol

from models.tp_utils import get_tp_group, get_tp_world_size

logger = logging.getLogger(__name__)

FAST_ATTN_IMPL = os.environ.get("FAST_ATTN_IMPL", "cte")
FAST_KERNEL_WRAP = os.environ.get("FAST_KERNEL_WRAP", "hop")
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)


# ─── attention_cte kernel (staging cp_attention._wan_attention_kernel, tp_out=False) ───

_cte_call = None


def _get_cte():
    global _cte_call
    if _cte_call is None:
        import nki
        import nki.language as nl
        from nkilib.core.attention.attention_cte import attention_cte

        @nki.jit
        def _cte_kernel(q, k, v):
            """[BN, Sq, D] x [BN, Sk, D] -> [BN, Sq, D]; q pre-scaled by the caller."""
            return attention_cte(
                q=q, k=k, v=v, scale=1.0, causal_mask=False,
                tp_q=True, tp_k=True, tp_out=False, cache_softmax=False,
                softmax_dtype=nl.float32, mm_out_dtype=nl.float32)

        if FAST_KERNEL_WRAP == "op":
            from torch_neuronx import nki_op

            @nki_op("wan14b::attention_cte")
            def _op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return _cte_kernel[LNC](q, k, v)

            _cte_call = _op
        else:
            from torch_neuronx.nki_hop import wrap_nki
            w = wrap_nki(_cte_kernel)
            try:
                _cte_call = w[LNC]
            except Exception:  # noqa: BLE001 -- no grid launch on this wrap_nki
                _cte_call = w
        logger.info(f"[fast_dit] attention_cte via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _cte_call


def _attn(q, k, v):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> [B, Lq, n, d]."""
    B, Lq, n, d = q.shape
    Lk = k.shape[1]
    if FAST_ATTN_IMPL == "sdpa":
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    q3 = (q * (d ** -0.5)).permute(0, 2, 1, 3).reshape(B * n, Lq, d)
    k3 = k.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    v3 = v.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    out = _get_cte()(q3, k3, v3)                                  # [BN, Lq, d]
    return out.reshape(B, n, Lq, d).transpose(1, 2)


# ─── TP helpers (in-graph functional collectives) ──────────────────────────────────

def _all_reduce(x):
    if get_tp_world_size() <= 1:
        return x
    return funcol.all_reduce(x, "sum", get_tp_group())


def _row_linear(layer, x):
    """RowParallelLinear: local matmul, in-graph all-reduce, then the (full) bias."""
    out = _all_reduce(F.linear(x, layer.weight, None))
    return out + layer.bias if layer.bias is not None else out


def _tp_rms_qk(q, k, norm_q, norm_k):
    """TPRMSNorm of q and k with ONE all-reduce of both sums of squares."""
    qf, kf = q.float(), k.float()
    Lq = q.shape[1]
    ss = _all_reduce(torch.cat([qf.pow(2).sum(-1, keepdim=True), kf.pow(2).sum(-1, keepdim=True)], dim=1))
    q = (qf * torch.rsqrt(ss[:, :Lq] / norm_q.global_dim + norm_q.eps)).type_as(q) * norm_q.weight
    k = (kf * torch.rsqrt(ss[:, Lq:] / norm_k.global_dim + norm_k.eps)).type_as(k) * norm_k.weight
    return q, k


def _tp_rms(x, norm):
    xf = x.float()
    ss = _all_reduce(xf.pow(2).sum(-1, keepdim=True))
    return (xf * torch.rsqrt(ss / norm.global_dim + norm.eps)).type_as(x) * norm.weight


def _rope(x, cos, sin):
    """x [B, L, n, d]; cos/sin [1, L, 1, d/2] fp32. Interleaved-pair rotation."""
    B, L, n, d = x.shape
    xp = x.float().reshape(B, L, n, d // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1).reshape(B, L, n, d).to(x.dtype)


# ─── Compiled units ──────────────────────────────────────────────────────────────────

class _FastBlock(nn.Module):
    """One WanAttentionBlock (uniform timestep: e is [1, 1, 6, C])."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, e, cos, sin, ck, cv):
        blk = self.block
        sa, ca = blk.self_attn, blk.cross_attn
        B, L, _ = x.shape
        n, d = sa.num_heads, sa.head_dim
        e = [t.squeeze(2) for t in (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)]  # [1, 1, C]

        h = blk.norm1(x).to(torch.bfloat16) * (1 + e[1]) + e[0]
        q, k = _tp_rms_qk(F.linear(h, sa.q.weight, sa.q.bias),
                          F.linear(h, sa.k.weight, sa.k.bias), sa.norm_q, sa.norm_k)
        v = F.linear(h, sa.v.weight, sa.v.bias).view(B, L, n, d)
        y = _attn(_rope(q.view(B, L, n, d), cos, sin), _rope(k.view(B, L, n, d), cos, sin), v)
        x = x + _row_linear(sa.o, y.reshape(B, L, n * d)) * e[2]

        h = blk.norm3(x)
        q = _tp_rms(F.linear(h, ca.q.weight, ca.q.bias), ca.norm_q).view(B, L, n, d)
        x = x + _row_linear(ca.o, _attn(q, ck, cv).reshape(B, L, n * d))

        h = blk.norm2(x).to(torch.bfloat16) * (1 + e[4]) + e[3]
        h = blk.ffn[1](F.linear(h, blk.ffn[0].weight, blk.ffn[0].bias))
        return x + _row_linear(blk.ffn[2], h) * e[5]


class _FastCrossKV(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, context):
        ca = self.block.cross_attn
        B, Lc, _ = context.shape
        n, d = ca.num_heads, ca.head_dim
        k = _tp_rms(F.linear(context, ca.k.weight, ca.k.bias), ca.norm_k).view(B, Lc, n, d)
        v = F.linear(context, ca.v.weight, ca.v.bias).view(B, Lc, n, d)
        return k, v


class _FastTime(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.time_embedding = model.time_embedding
        self.time_projection = model.time_projection
        self.dim = model.dim

    def forward(self, s):                       # s [1, 1, freq_dim] bf16
        e = self.time_embedding(s)              # [1, 1, C]
        return e, self.time_projection(e).unflatten(2, (6, self.dim))  # [1, 1, 6, C]


class _FastPatch(nn.Module):
    def __init__(self, patch_embedding):
        super().__init__()
        self.patch_embedding = patch_embedding

    def forward(self, latent, y):               # [16, F, H, W], [20, F, H, W]
        x = self.patch_embedding(torch.cat([latent, y], dim=0).unsqueeze(0))
        x = x.flatten(2).transpose(1, 2)        # [1, L, C]
        return x.expand(2, -1, -1).contiguous()


class _FastText(nn.Module):
    def __init__(self, text_embedding):
        super().__init__()
        self.text_embedding = text_embedding

    def forward(self, ctx):
        return self.text_embedding(ctx)


class _FastHead(nn.Module):
    """Head + unpatchify + CFG combine -> [C_out, F, H, W]."""

    def __init__(self, head, grid, patch_size, out_dim, guide_scale):
        super().__init__()
        self.head, self.grid, self.patch_size = head, grid, patch_size
        self.out_dim, self.guide_scale = out_dim, guide_scale

    def forward(self, x, e):                    # e [1, 1, C]
        hd = self.head
        f, h, w = self.grid
        p, q, r = self.patch_size
        c = self.out_dim
        m = (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        y = hd.head(hd.norm(x) * (1 + m[1].squeeze(2)) + m[0].squeeze(2))
        B = y.shape[0]
        y = y.view(B, f, h, w, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        y = y.reshape(B, c, f * p, h * q, w * r).to(torch.bfloat16)
        cond, uncond = y[0], y[1]
        return uncond + self.guide_scale * (cond - uncond)


def _rope_tables(freqs, grid, head_dim, device):
    """Same angle grid as rope_apply_neuron, built once -> cos/sin [1, L, 1, d/2]."""
    f, h, w = grid
    c = head_dim // 2
    fr = freqs.cpu().float().split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    ang = torch.cat([
        fr[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(1, f * h * w, 1, c)
    return torch.cos(ang).to(device), torch.sin(ang).to(device)


class FastWanDiT:
    """Batched-CFG, fullgraph-compiled forward over one TP-sharded expert.

        fast = FastWanDiT(model, grid=(F, H/2, W/2), guide_scale=g, device=dev)
        fast.set_context(ctx_cond, ctx_null)     # [1, 512, 4096] each, once per prompt
        noise_pred = fast(latent, y, t)          # latent [16, F, H, W], y [20, F, H, W]
    """

    def __init__(self, model, grid, guide_scale, device):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        self.model = model
        self.grid = tuple(int(g) for g in grid)
        self.device = device
        self.cos, self.sin = _rope_tables(model.freqs, self.grid, model.dim // model.num_heads, device)
        if FAST_ATTN_IMPL == "cte":
            _get_cte()  # build the kernel wrapper BEFORE tracing (imports inside a trace break fullgraph)
        self.time = torch.compile(_FastTime(model), **COMPILE_KW)
        self.patch = torch.compile(_FastPatch(model.patch_embedding), **COMPILE_KW)
        self.text = torch.compile(_FastText(model.text_embedding), **COMPILE_KW)
        self.blocks = [torch.compile(_FastBlock(b), **COMPILE_KW) for b in model.blocks]
        self.cross_kv = [torch.compile(_FastCrossKV(b), **COMPILE_KW) for b in model.blocks]
        self.head = torch.compile(
            _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **COMPILE_KW)
        self.context = None
        logger.info(f"[fast_dit] grid={self.grid} L={math.prod(self.grid)} attn={FAST_ATTN_IMPL} "
                    f"fullgraph=True blocks={len(self.blocks)} cfg_batch=2 guide={guide_scale}")

    def set_context(self, ctx_cond, ctx_null):
        context = self.text(torch.cat([ctx_cond, ctx_null], dim=0).to(torch.bfloat16))
        self.context = [kv(context) for kv in self.cross_kv]

    def __call__(self, latent, y, t):
        from wan.modules.model import sinusoidal_embedding_1d
        s = sinusoidal_embedding_1d(self.model.freq_dim, torch.tensor([float(t)], dtype=torch.float64))
        e, e0 = self.time(s.to(torch.bfloat16).view(1, 1, -1).to(self.device))
        x = self.patch(latent.to(torch.bfloat16), y.to(torch.bfloat16))
        for blk, (ck, cv) in zip(self.blocks, self.context):
            x = blk(x, e0, self.cos, self.sin, ck, cv)
        return self.head(x, e)

    @staticmethod
    def log_graph_breaks():
        from torch._dynamo.utils import counters
        breaks = dict(counters.get("graph_break", {}))
        if breaks:
            logger.warning(f"[fast_dit] graph breaks ({sum(breaks.values())}): {breaks}")
        else:
            logger.info("[fast_dit] no graph breaks")
