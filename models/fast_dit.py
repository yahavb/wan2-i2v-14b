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

FAST_ATTN_IMPL: cte (default, nkilib attention_cte) | sdpa (F.scaled_dot_product_attention)
                | constmax (self-attn: staging const-max kernel, kernels/nkilib/experimental/
                  attention/ring_attention_const_max_fwd.py, num_workers=1; cross-attn: cte,
                  as staging does).
FAST_KERNEL_WRAP: hop (default, torch_neuronx.nki_hop.wrap_nki) | op (torch_neuronx.nki_op).
FAST_MLP=1   : FFN via nkilib mlp (staging feed_forward._wan_mlp_kernel: skip_gate_proj,
               GELU_Tanh_Approx), down bias pre-divided by TP, then the in-graph all-reduce.
FAST_ADALN=1 : staging fused adaln kernel (kernels/nkilib/experimental/norm/adaln_quant.py) for
               norm1+modulation, norm3 (cross-attn affine LN) with the self-attn gated residual
               add, and norm2+modulation with the cross-attn residual add (staging block order).
FAST_OPROJ=1 : self/cross O-proj via nkilib output_projection_cte (staging cp_attention
               _wan_o_proj_kernel) on d-major attention output (attention_cte tp_out=True).
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
FAST_MLP = os.environ.get("FAST_MLP", "0") == "1"
FAST_OPROJ = os.environ.get("FAST_OPROJ", "0") == "1"
FAST_ADALN = os.environ.get("FAST_ADALN", "0") == "1"
# Compile scope (torch.compile scope study, docs/torch_compile_scope.md):
#   block        one graph per transformer block, fullgraph=True (default)
#   block_breaks same region, torch._dynamo.graph_break() after self-attn and cross-attn,
#                fullgraph=False -> 3 graphs per block
#   sublayer     self-attn / cross-attn / FFN compiled as 3 separate fullgraph=True units
#   model        all 40 blocks in one fullgraph=True graph
FAST_SCOPE = os.environ.get("FAST_SCOPE", "block")
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)


# ─── attention_cte kernel (staging cp_attention._wan_attention_kernel, tp_out=False) ───

def _wrap(kernel, op_name, grid=True):
    """Launch an @nki.jit kernel from inside a fullgraph graph, on all LNC cores
    (grid=False: single-program launch, for kernels that do not shard across cores)."""
    if FAST_KERNEL_WRAP == "op":
        from torch_neuronx import nki_op

        @nki_op(op_name)
        def _op(*args: torch.Tensor) -> torch.Tensor:
            return kernel[LNC](*args) if grid else kernel(*args)

        return _op
    from torch_neuronx.nki_hop import wrap_nki
    w = wrap_nki(kernel)
    if not grid:
        return w
    try:
        return w[LNC]
    except Exception:  # noqa: BLE001 -- no grid launch on this wrap_nki
        return w


# ─── legacy NKI attention (kernels/self_attention.py, kernels/cross_attention.py) ───
# Same kernels the eager path ran (FAST_ATTN_IMPL=legacy), called in-graph: per CFG item,
# [n, d, L] layout, Q padded to 128, K/V padded to 2048 with a -inf mask, P transposed by
# an identity matmul. Single-program launch, as in the eager path.

_legacy = {}


def _init_legacy(L, Lc, device):
    import math as _m
    import nki
    from kernels.self_attention import wan_flash_self_attn
    from kernels.cross_attention import wan_cross_attn
    d = 128
    lk = -(-L // 2048) * 2048
    mask = torch.zeros(128, lk, dtype=torch.bfloat16)
    mask[:, L:] = float("-inf")
    scale, nsec = 1.0 / _m.sqrt(d), lk // 2048

    @nki.jit
    def _self(q, k, v, identity, mask):
        return wan_flash_self_attn(q, k, v, identity, mask, softmax_scale=scale, num_sections=nsec)

    @nki.jit
    def _cross(q, k, v, identity):
        return wan_cross_attn(q, k, v, identity, softmax_scale=scale)

    _legacy.update(self=_wrap(_self, "wan14b::legacy_self", grid=False),
                   cross=_wrap(_cross, "wan14b::legacy_cross", grid=False),
                   identity=torch.eye(128, dtype=torch.bfloat16).to(device),
                   mask=mask.to(device), L=L, lk=lk)
    logger.info(f"[fast_dit] legacy NKI attention: L={L} K padded to {lk}, single-program")


def _legacy_attn(q, k, v, cross):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> [B, Lq, n, d], one kernel call per CFG item."""
    B, Lq, n, d = q.shape
    lq = -(-Lq // 128) * 128
    outs = []
    for bi in range(B):
        qn = F.pad(q[bi].permute(1, 2, 0), (0, lq - Lq))                     # [n, d, Lq_pad]
        kn = k[bi].permute(1, 2, 0)                                           # [n, d, Lk]
        vn = v[bi].permute(1, 0, 2)                                           # [n, Lk, d]
        if cross:
            o = _legacy["cross"](qn.contiguous(), kn.contiguous(), vn.contiguous(), _legacy["identity"])
        else:
            pad = _legacy["lk"] - k.shape[1]
            kn, vn = F.pad(kn, (0, pad)), F.pad(vn, (0, 0, 0, pad))
            o = _legacy["self"](qn.contiguous(), kn.contiguous(), vn.contiguous(),
                                _legacy["identity"], _legacy["mask"])
        outs.append(o[:Lq].unsqueeze(0))                                     # [1, Lq, n, d]
    return torch.cat(outs, dim=0)


_cte_calls = {}


def _get_cte(tp_out=False):
    """attention_cte: [BN, Sq, D] x [BN, Sk, D] -> [BN, Sq, D] (tp_out=False) or
    [BN, D, Sq] (tp_out=True, the d-major layout output_projection_cte consumes)."""
    if tp_out not in _cte_calls:
        import nki
        import nki.language as nl
        from nkilib.core.attention.attention_cte import attention_cte

        @nki.jit
        def _cte_kernel(q, k, v):
            return attention_cte(
                q=q, k=k, v=v, scale=1.0, causal_mask=False,
                tp_q=True, tp_k=True, tp_out=tp_out, cache_softmax=False,
                softmax_dtype=nl.float32, mm_out_dtype=nl.float32)

        _cte_calls[tp_out] = _wrap(_cte_kernel, f"wan14b::attention_cte_tp{int(tp_out)}")
        logger.info(f"[fast_dit] attention_cte tp_out={tp_out} via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _cte_calls[tp_out]


_mlp_call = None


def _get_mlp():
    global _mlp_call
    if _mlp_call is None:
        import nki
        from nkilib.core.mlp.mlp import mlp as nkilib_mlp
        from nkilib.core.utils.common_types import ActFnType, DtypeMode, NormType, QuantizationType

        @nki.jit
        def _mlp_kernel(hidden, up_weight, down_weight, up_bias, down_bias):
            """GELU_tanh(hidden @ up + up_b) @ down + down_b; hidden [B, T, H], up [H, I],
            down [I, H], biases [1, *]. Same arguments as staging _wan_mlp_kernel."""
            return nkilib_mlp(
                hidden_tensor=hidden, gate_proj_weights_tensor=up_weight,
                up_proj_weights_tensor=up_weight, down_proj_weights_tensor=down_weight,
                normalization_weights_tensor=None, gate_proj_bias_tensor=up_bias,
                up_proj_bias_tensor=up_bias, down_proj_bias_tensor=down_bias,
                normalization_bias_tensor=None, fused_add_tensor=None, store_fused_add_result=False,
                activation_fn=ActFnType.GELU_Tanh_Approx, normalization_type=NormType.NO_NORM,
                quantization_type=QuantizationType.NONE, gate_w_scale=None, up_w_scale=None,
                down_w_scale=None, gate_up_in_scale=None, down_in_scale=None,
                quant_clipping_bound=0.0, output_dtype=None, store_output_in_sbuf=False, eps=1e-6,
                skip_gate_proj=True, use_tkg_gate_up_proj_column_tiling=True,
                use_tkg_down_proj_column_tiling=True, use_tkg_down_proj_optimized_layout=False,
                gate_clamp_upper_limit=None, gate_clamp_lower_limit=None,
                up_clamp_upper_limit=None, up_clamp_lower_limit=None,
                force_cte_mode=False, dtype_mode=DtypeMode.AUTO)

        _mlp_call = _wrap(_mlp_kernel, "wan14b::mlp")
        logger.info(f"[fast_dit] nkilib mlp via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _mlp_call


_adaln_calls = None


def _get_adaln():
    """Three tensor-only launchers (kargs are built inside the jit, never passed through the graph):
    plain (norm -> *mul + add), residual+gate, residual without gate. All LayerNorm, OCP dtype mode."""
    global _adaln_calls
    if _adaln_calls is None:
        import nki
        from kernels.nkilib.core.utils.common_types import DtypeMode, NormType, QuantizationType
        from kernels.nkilib.experimental.norm.adaln_quant import AdaLNQuantKernelArgs, adaln_quant_kernel

        def kargs(eps):
            return AdaLNQuantKernelArgs(norm_type=NormType.LAYER_NORM,
                                        quantization_type=QuantizationType.NONE, eps=eps)

        @nki.jit
        def _plain(hidden, mul, add):
            return adaln_quant_kernel(hidden, kargs(1e-6), mul=mul, add=add, dtype_mode=DtypeMode.OCP)

        @nki.jit
        def _res_gate(hidden, mul, add, residual, gate):
            return adaln_quant_kernel(hidden, kargs(1e-6), mul=mul, add=add, dtype_mode=DtypeMode.OCP,
                                      residual=residual, gate=gate)

        @nki.jit
        def _res(hidden, mul, add, residual):
            return adaln_quant_kernel(hidden, kargs(1e-6), mul=mul, add=add, dtype_mode=DtypeMode.OCP,
                                      residual=residual)

        _adaln_calls = (_wrap(_plain, "wan14b::adaln"), _wrap(_res_gate, "wan14b::adaln_res_gate"),
                        _wrap(_res, "wan14b::adaln_res"))
        logger.info(f"[fast_dit] adaln_quant via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _adaln_calls


def _row(v, H, dtype):
    return v.reshape(1, H).to(dtype)


_oproj_call = None


def _get_oproj():
    global _oproj_call
    if _oproj_call is None:
        import nki
        from nkilib.core.output_projection.output_projection_cte import output_projection_cte
        from nkilib.core.utils.common_types import QuantizationType

        @nki.jit
        def _oproj_kernel(active, weight, bias):
            """active [B, N, D, S], weight [N*D, H], bias [1, H] -> [B, S, H]."""
            return output_projection_cte(active, weight, bias, QuantizationType.NONE, None, None)

        _oproj_call = _wrap(_oproj_kernel, "wan14b::output_projection_cte")
        logger.info(f"[fast_dit] nkilib output_projection_cte via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _oproj_call


_constmax_call = None


def _get_constmax():
    global _constmax_call
    if _constmax_call is None:
        from kernels.nkilib.experimental.attention.ring_attention_const_max_fwd import (
            ring_attention_const_max_fwd)
        if FAST_KERNEL_WRAP == "op":
            from torch_neuronx import nki_op

            @nki_op("wan14b::attention_const_max")
            def _op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return ring_attention_const_max_fwd[LNC](
                    q, k, v, num_workers=1, softmax_scale=q.shape[-1] ** -0.5)

            _constmax_call = lambda q, k, v: _op(q, k, v)
        else:
            from torch_neuronx.nki_hop import wrap_nki
            w = wrap_nki(ring_attention_const_max_fwd)
            try:
                w = w[LNC]
            except Exception:  # noqa: BLE001
                pass
            _constmax_call = lambda q, k, v: w(q, k, v, num_workers=1, softmax_scale=q.shape[-1] ** -0.5)
        logger.info(f"[fast_dit] const-max self-attention via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _constmax_call


def _self_attn(q, k, v):
    """q/k/v [B, L, n, d] token-major -> [B, L, n, d]."""
    if FAST_ATTN_IMPL == "legacy":
        return _legacy_attn(q, k, v, cross=False)
    if FAST_ATTN_IMPL == "constmax":
        return _get_constmax()(q.contiguous(), k.contiguous(), v.contiguous()).transpose(1, 2)  # [B, n, L, d] -> [B, L, n, d]
    return _attn(q, k, v)


def _attn_dmajor(q, k, v):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> d-major [B, n, d, Lq] (attention_cte tp_out=True)."""
    B, Lq, n, d = q.shape
    Lk = k.shape[1]
    q3 = (q * (d ** -0.5)).permute(0, 2, 1, 3).reshape(B * n, Lq, d)
    k3 = k.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    v3 = v.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    return _get_cte(True)(q3, k3, v3).reshape(B, n, d, Lq)


def _o_proj(layer, y_tok, y_dmajor):
    """O projection + in-graph all-reduce. FAST_OPROJ uses the kernel on the d-major output."""
    if FAST_OPROJ:
        return _all_reduce(_get_oproj()(y_dmajor, layer.o_w_t, layer.o_b_scaled))
    return _row_linear(layer, y_tok)


def _attn(q, k, v):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> [B, Lq, n, d]."""
    B, Lq, n, d = q.shape
    Lk = k.shape[1]
    if FAST_ATTN_IMPL == "legacy":
        return _legacy_attn(q, k, v, cross=True)
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

    def _e(self, e):
        return [t.squeeze(2) for t in (self.block.modulation.unsqueeze(0) + e).chunk(6, dim=2)]  # [1, 1, C]

    # Sub-layers of the unfused (no adaln / o_proj / mlp kernel) block. The scope variants
    # put graph boundaries between them; the math is identical in every scope.
    def self_part(self, x, e0, cos, sin):
        blk, e = self.block, self._e(e0)
        sa = blk.self_attn
        B, L, _ = x.shape
        n, d = sa.num_heads, sa.head_dim
        h = blk.norm1(x).to(torch.bfloat16) * (1 + e[1]) + e[0]
        q, k = _tp_rms_qk(F.linear(h, sa.q.weight, sa.q.bias),
                          F.linear(h, sa.k.weight, sa.k.bias), sa.norm_q, sa.norm_k)
        v = F.linear(h, sa.v.weight, sa.v.bias).view(B, L, n, d)
        q, k = _rope(q.view(B, L, n, d), cos, sin), _rope(k.view(B, L, n, d), cos, sin)
        return x + _row_linear(sa.o, _self_attn(q, k, v).reshape(B, L, n * d)) * e[2]

    def cross_part(self, x, ck, cv):
        blk = self.block
        ca = blk.cross_attn
        B, L, _ = x.shape
        n, d = ca.num_heads, ca.head_dim
        q = _tp_rms(F.linear(blk.norm3(x), ca.q.weight, ca.q.bias), ca.norm_q).view(B, L, n, d)
        return x + _row_linear(ca.o, _attn(q, ck, cv).reshape(B, L, n * d))

    def ffn_part(self, x, e0):
        blk, e = self.block, self._e(e0)
        h = blk.norm2(x).to(torch.bfloat16) * (1 + e[4]) + e[3]
        return x + _row_linear(blk.ffn[2], blk.ffn[1](F.linear(h, blk.ffn[0].weight, blk.ffn[0].bias))) * e[5]

    def forward(self, x, e, cos, sin, ck, cv):
        if not (FAST_ADALN or FAST_OPROJ or FAST_MLP):
            x = self.self_part(x, e, cos, sin)
            if FAST_SCOPE == "block_breaks":
                torch._dynamo.graph_break()          # graph 1 ends: self-attention
            x = self.cross_part(x, ck, cv)
            if FAST_SCOPE == "block_breaks":
                torch._dynamo.graph_break()          # graph 2 ends: cross-attention
            return self.ffn_part(x, e)                # graph 3: FFN
        blk = self.block
        sa, ca = blk.self_attn, blk.cross_attn
        B, L, _ = x.shape
        n, d = sa.num_heads, sa.head_dim
        e = self._e(e)

        C = x.shape[-1]
        if FAST_ADALN:
            plain, res_gate, res = _get_adaln()
            h = plain(x.reshape(B * L, C), _row(1 + e[1], C, x.dtype), _row(e[0], C, x.dtype)).view(B, L, C)
        else:
            h = blk.norm1(x).to(torch.bfloat16) * (1 + e[1]) + e[0]
        q, k = _tp_rms_qk(F.linear(h, sa.q.weight, sa.q.bias),
                          F.linear(h, sa.k.weight, sa.k.bias), sa.norm_q, sa.norm_k)
        v = F.linear(h, sa.v.weight, sa.v.bias).view(B, L, n, d)
        q, k = _rope(q.view(B, L, n, d), cos, sin), _rope(k.view(B, L, n, d), cos, sin)
        if FAST_OPROJ and FAST_ATTN_IMPL == "cte":
            y = _o_proj(sa, None, _attn_dmajor(q, k, v))
        else:
            y = _row_linear(sa.o, _self_attn(q, k, v).reshape(B, L, n * d))
        if FAST_ADALN:   # x = x + gate*y ; h = norm3(x) (affine LN), one kernel
            n3 = blk.norm3
            x, h = res_gate(y.reshape(B * L, C), _row(n3.weight, C, x.dtype), _row(n3.bias, C, x.dtype),
                            x.reshape(B * L, C), _row(e[2], C, torch.float32))
            x, h = x.view(B, L, C), h.view(B, L, C)
        else:
            x = x + y * e[2]
            h = blk.norm3(x)
        q = _tp_rms(F.linear(h, ca.q.weight, ca.q.bias), ca.norm_q).view(B, L, n, d)
        if FAST_OPROJ:
            y = _o_proj(ca, None, _attn_dmajor(q, ck, cv))
        else:
            y = _row_linear(ca.o, _attn(q, ck, cv).reshape(B, L, n * d))

        if FAST_ADALN:   # x = x + y ; h = norm2(x) * (1 + e4) + e3, one kernel
            x, h = res(y.reshape(B * L, C), _row(1 + e[4], C, x.dtype), _row(e[3], C, x.dtype),
                       x.reshape(B * L, C))
            x, h = x.view(B, L, C), h.view(B, L, C)
        else:
            x = x + y
            h = blk.norm2(x).to(torch.bfloat16) * (1 + e[4]) + e[3]
        if FAST_MLP:
            f = blk.ffn
            y = _get_mlp()(h.reshape(1, B * L, -1), f.up_w_t, f.down_w_t, f.up_b2, f.down_b_scaled)
            y = _all_reduce(y).reshape(B, L, -1)
        else:
            y = _row_linear(blk.ffn[2], blk.ffn[1](F.linear(h, blk.ffn[0].weight, blk.ffn[0].bias)))
        return x + y * e[5]


class _Part(nn.Module):
    """One sub-layer of a block as its own compiled unit (FAST_SCOPE=sublayer)."""

    def __init__(self, fast_block, which):
        super().__init__()
        self.fb, self.which = fast_block, which

    def forward(self, *args):
        return getattr(self.fb, self.which)(*args)


class _FastModel(nn.Module):
    """All blocks in one graph (FAST_SCOPE=model); kv = (ck0, cv0, ck1, cv1, ...)."""

    def __init__(self, fast_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(fast_blocks)

    def forward(self, x, e, cos, sin, *kv):
        for i, b in enumerate(self.blocks):
            x = b(x, e, cos, sin, kv[2 * i], kv[2 * i + 1])
        return x


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


def _prepare_kernel_weights(model):
    """Kernel weight layouts, made once (both kernels take [in, out]); originals freed.
    Row-parallel biases are pre-divided by TP so the all-reduce sum adds them once."""
    tp = get_tp_world_size()

    def buf(mod, name, t):
        mod.register_buffer(name, t.contiguous(), persistent=False)

    for blk in model.blocks:
        if FAST_MLP:
            f = blk.ffn
            buf(f, "up_w_t", f[0].weight.t())                    # [H, I_local]
            buf(f, "up_b2", f[0].bias.view(1, -1))
            buf(f, "down_w_t", f[2].weight.t())                  # [I_local, H]
            buf(f, "down_b_scaled", (f[2].bias / tp).view(1, -1))
            for lin in (f[0], f[2]):
                lin.weight.data = torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device)
        if FAST_OPROJ:
            for att in (blk.self_attn, blk.cross_attn):
                buf(att, "o_w_t", att.o.weight.t())              # [N*D local, H]
                buf(att, "o_b_scaled", (att.o.bias / tp).view(1, -1))
                att.o.weight.data = torch.empty(0, dtype=att.o.weight.dtype, device=att.o.weight.device)


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
        # Build kernel wrappers BEFORE tracing (imports inside a trace break fullgraph).
        if FAST_ATTN_IMPL in ("cte", "constmax"):
            _get_cte()          # cross-attention (and self-attention for cte)
        if FAST_ATTN_IMPL == "constmax":
            _get_constmax()
        if FAST_OPROJ:
            _get_cte(True)
            _get_oproj()
        if FAST_MLP:
            _get_mlp()
        if FAST_ADALN:
            _get_adaln()
        _prepare_kernel_weights(model)
        self.time = torch.compile(_FastTime(model), **COMPILE_KW)
        self.patch = torch.compile(_FastPatch(model.patch_embedding), **COMPILE_KW)
        self.text = torch.compile(_FastText(model.text_embedding), **COMPILE_KW)
        assert FAST_SCOPE in ("block", "block_breaks", "sublayer", "model"), FAST_SCOPE
        if FAST_SCOPE in ("block_breaks", "sublayer"):
            assert not (FAST_ADALN or FAST_OPROJ or FAST_MLP), "split scopes run the unfused block"
        if FAST_ATTN_IMPL == "legacy":
            _init_legacy(math.prod(self.grid), model.text_len, device)
        fbs = [_FastBlock(b) for b in model.blocks]
        self.blocks = self.parts = self.whole = None
        if FAST_SCOPE == "block":
            self.blocks = [torch.compile(fb, **COMPILE_KW) for fb in fbs]
        elif FAST_SCOPE == "block_breaks":
            self.blocks = [torch.compile(fb, **{**COMPILE_KW, "fullgraph": False}) for fb in fbs]
        elif FAST_SCOPE == "sublayer":
            self.parts = [tuple(torch.compile(_Part(fb, w), **COMPILE_KW)
                                for w in ("self_part", "cross_part", "ffn_part")) for fb in fbs]
        else:
            self.whole = torch.compile(_FastModel(fbs), **COMPILE_KW)
        self.n_blocks = len(fbs)
        self.cross_kv = [torch.compile(_FastCrossKV(b), **COMPILE_KW) for b in model.blocks]
        self.head = torch.compile(
            _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **COMPILE_KW)
        self.context = None
        logger.info(f"[fast_dit] grid={self.grid} L={math.prod(self.grid)} attn={FAST_ATTN_IMPL} "
                    f"scope={FAST_SCOPE} blocks={self.n_blocks} cfg_batch=2 guide={guide_scale}")

    def set_context(self, ctx_cond, ctx_null):
        context = self.text(torch.cat([ctx_cond, ctx_null], dim=0).to(torch.bfloat16))
        self.context = [kv(context) for kv in self.cross_kv]

    def __call__(self, latent, y, t):
        from wan.modules.model import sinusoidal_embedding_1d
        s = sinusoidal_embedding_1d(self.model.freq_dim, torch.tensor([float(t)], dtype=torch.float64))
        e, e0 = self.time(s.to(torch.bfloat16).view(1, 1, -1).to(self.device))
        x = self.patch(latent.to(torch.bfloat16), y.to(torch.bfloat16))
        if self.whole is not None:
            x = self.whole(x, e0, self.cos, self.sin, *[t for kv in self.context for t in kv])
        elif self.parts is not None:
            for (sp, cp, fp), (ck, cv) in zip(self.parts, self.context):
                x = fp(cp(sp(x, e0, self.cos, self.sin), ck, cv), e0)
        else:
            for blk, (ck, cv) in zip(self.blocks, self.context):
                x = blk(x, e0, self.cos, self.sin, ck, cv)
        return self.head(x, e)

    @staticmethod
    def compile_stats():
        """Dynamo's own counters (cumulative per process): graphs compiled, graph breaks."""
        from torch._dynamo.utils import counters
        breaks = dict(counters.get("graph_break", {}))
        return (f"scope={FAST_SCOPE} attn={FAST_ATTN_IMPL} "
                f"unique_graphs={counters['stats'].get('unique_graphs', 0)} "
                f"graph_breaks={sum(breaks.values())} reasons={list(breaks)[:4]}")

    @staticmethod
    def log_graph_breaks():
        from torch._dynamo.utils import counters
        breaks = dict(counters.get("graph_break", {}))
        if breaks:
            logger.warning(f"[fast_dit] graph breaks ({sum(breaks.values())}): {breaks}")
        else:
            logger.info("[fast_dit] no graph breaks")
