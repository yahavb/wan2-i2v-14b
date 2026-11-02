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

from models.tp_utils import get_tp_group, get_tp_rank, get_tp_world_size

logger = logging.getLogger(__name__)

FAST_ATTN_IMPL = os.environ.get("FAST_ATTN_IMPL", "cte")
FAST_KERNEL_WRAP = os.environ.get("FAST_KERNEL_WRAP", "hop")
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
FAST_MLP = os.environ.get("FAST_MLP", "0") == "1"
FAST_OPROJ = os.environ.get("FAST_OPROJ", "0") == "1"
FAST_ADALN = os.environ.get("FAST_ADALN", "0") == "1"
# FAST_QKV=1: self-attn q|k|v as ONE matmul on concatenated weights (weights load once, one
# larger matmul); outputs are the same dot products, split after. FAST_QKV_CHUNKS=N (with
# FAST_ADALN): run norm1+modulation and the QKV matmul over N row chunks so the vector engine
# can normalize chunk i+1 while the tensor engine multiplies chunk i (per-row math unchanged).
FAST_QKV = os.environ.get("FAST_QKV", "0") == "1"
FAST_QKV_CHUNKS = int(os.environ.get("FAST_QKV_CHUNKS", "1"))
# FAST_QKV_NKI=1: norm1 + modulation + q|k|v projection + bias in ONE nkilib qkv (CTE) kernel.
# AdaLN with a uniform timestep is LayerNorm with gamma = 1 + scale and beta = shift, which the
# kernel fuses with the matmul and pipelines tile by tile (vector-engine norm overlapping the
# tensor-engine matmul). Replaces the adaln kernel + QKV matmuls of the self-attention input.
FAST_QKV_NKI = os.environ.get("FAST_QKV_NKI", "0") == "1"
# FAST_CROSSQ_NKI=1: cross-attention q = norm3(x) @ Wq + b in the same nkilib qkv kernel
# (affine LayerNorm with norm3's weight/bias), replacing the adaln res_gate kernel + q matmul.
# The gated residual x + gate * y becomes a plain torch add.
FAST_CROSSQ_NKI = os.environ.get("FAST_CROSSQ_NKI", "0") == "1"
# FAST_FFNUP_NKI=1: FFN up-projection = (norm2(x) * (1 + e4) + e3) @ W_up + b in the nkilib qkv
# kernel (LayerNorm gamma = 1 + e4, beta = e3), replacing the adaln res kernel + up matmul.
# The residual x + y becomes a plain torch add.
FAST_FFNUP_NKI = os.environ.get("FAST_FFNUP_NKI", "0") == "1"
# FAST_QKV_SPLIT=1 (with FAST_QKV_NKI): q, k and v as three calls of the same kernel on [H, D]
# weights (13 MB each at TP 4) instead of one on the fused [H, 3D] (39 MB, more than one core's
# 24 MB SBUF); the fused LayerNorm runs three times.
FAST_QKV_SPLIT = os.environ.get("FAST_QKV_SPLIT", "0") == "1"
# FAST_XRES_KERNEL=1 (with FAST_CROSSQ_NKI + FAST_ADALN): the gated residual x + gate * y before
# the cross-attention runs in the adaln res_gate kernel (vector engine) instead of a torch add,
# which the compiler lowers with tensor-engine transposes; its normalized output is unused.
FAST_XRES_KERNEL = os.environ.get("FAST_XRES_KERNEL", "0") == "1"
COMPILE_KW = dict(backend="neuron", dynamic=False, fullgraph=True)


# ─── attention_cte kernel (staging cp_attention._wan_attention_kernel, tp_out=False) ───

def _wrap(kernel, op_name):
    """Launch an @nki.jit kernel from inside a fullgraph graph, on all LNC cores."""
    if FAST_KERNEL_WRAP == "op":
        from torch_neuronx import nki_op

        @nki_op(op_name)
        def _op(*args: torch.Tensor) -> torch.Tensor:
            return kernel[LNC](*args)

        return _op
    from torch_neuronx.nki_hop import wrap_nki
    w = wrap_nki(kernel)
    try:
        return w[LNC]
    except Exception:  # noqa: BLE001 -- no grid launch on this wrap_nki
        return w


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


_qkv_call = None


def _get_qkv():
    """nkilib qkv: input [B, S, H], weight [H, I], bias [1, I], gamma [1, H], beta [1, H]
    -> LayerNorm(input) * gamma + beta, @ weight + bias  ->  [B, S, I]."""
    global _qkv_call
    if _qkv_call is None:
        import nki
        from nkilib.core.qkv.qkv import qkv
        from nkilib.core.utils.common_types import NormType, QKVOutputLayout

        @nki.jit
        def _qkv_kernel(hidden, weight, bias, gamma, beta):
            return qkv(hidden, weight, QKVOutputLayout.BSD, bias=bias,
                       fused_norm_type=NormType.LAYER_NORM, gamma_norm_weights=gamma,
                       layer_norm_bias=beta, norm_eps=1e-6)

        _qkv_call = _wrap(_qkv_kernel, "wan14b::qkv_ln")
        logger.info(f"[fast_dit] nkilib qkv (LayerNorm fused) via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _qkv_call


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

        C = x.shape[-1]
        if FAST_QKV_NKI and FAST_QKV_SPLIT:  # q, k, v as three kernel calls (each weight fits SBUF)
            g1, b1 = _row(1 + e[1], C, x.dtype), _row(e[0], C, x.dtype)
            q, k, v = (_get_qkv()(x, w, bb, g1, b1).view(B, L, -1)
                       for w, bb in ((sa.q_wt, sa.q_b2), (sa.k_wt, sa.k_b2), (sa.v_wt, sa.v_b2)))
            if FAST_ADALN:
                plain, res_gate, res = _get_adaln()
        elif FAST_QKV_NKI:
            qkv = _get_qkv()(x, sa.qkv_wt, sa.qkv_b2, _row(1 + e[1], C, x.dtype), _row(e[0], C, x.dtype))
            q, k, v = qkv.view(B, L, -1).split(sa.qkv_split, dim=-1)
            if FAST_ADALN:
                plain, res_gate, res = _get_adaln()
        elif FAST_QKV and FAST_ADALN and FAST_QKV_CHUNKS > 1:
            plain, res_gate, res = _get_adaln()
            mul, add = _row(1 + e[1], C, x.dtype), _row(e[0], C, x.dtype)
            qkv = torch.cat([F.linear(plain(xc, mul, add), sa.qkv_w, sa.qkv_b)
                             for xc in torch.tensor_split(x.reshape(B * L, C), FAST_QKV_CHUNKS)], 0)
            q, k, v = qkv.view(B, L, -1).split(sa.qkv_split, dim=-1)
        else:
            if FAST_ADALN:
                plain, res_gate, res = _get_adaln()
                h = plain(x.reshape(B * L, C), _row(1 + e[1], C, x.dtype), _row(e[0], C, x.dtype)).view(B, L, C)
            else:
                h = blk.norm1(x).to(torch.bfloat16) * (1 + e[1]) + e[0]
            if FAST_QKV:
                q, k, v = F.linear(h, sa.qkv_w, sa.qkv_b).split(sa.qkv_split, dim=-1)
            else:
                q, k, v = (F.linear(h, sa.q.weight, sa.q.bias), F.linear(h, sa.k.weight, sa.k.bias),
                           F.linear(h, sa.v.weight, sa.v.bias))
        q, k = _tp_rms_qk(q, k, sa.norm_q, sa.norm_k)
        v = v.reshape(B, L, n, d)
        q, k = _rope(q.view(B, L, n, d), cos, sin), _rope(k.view(B, L, n, d), cos, sin)
        if FAST_OPROJ and FAST_ATTN_IMPL == "cte":
            y = _o_proj(sa, None, _attn_dmajor(q, k, v))
        else:
            y = _row_linear(sa.o, _self_attn(q, k, v).reshape(B, L, n * d))
        if FAST_CROSSQ_NKI:  # x = x + gate*y (torch) ; q = norm3(x) @ Wq + b (one kernel)
            n3 = blk.norm3
            if FAST_XRES_KERNEL and FAST_ADALN:
                x, _ = res_gate(y.reshape(B * L, C), _row(n3.weight, C, x.dtype), _row(n3.bias, C, x.dtype),
                                x.reshape(B * L, C), _row(e[2], C, torch.float32))
                x = x.view(B, L, C)
            else:
                x = x + (y.reshape(B, L, C) * e[2]).to(x.dtype)
            q = _get_qkv()(x, ca.q_wt, ca.q_b2, n3.w_row.to(x.dtype), n3.b_row.to(x.dtype))
            q = _tp_rms(q.view(B, L, -1), ca.norm_q).view(B, L, n, d)
        else:
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

        if FAST_FFNUP_NKI:  # x = x + y (torch) ; u = (norm2(x)*(1+e4)+e3) @ W_up + b (one kernel)
            x = x + y.reshape(B, L, C)
            f0 = blk.ffn[0]
            u = _get_qkv()(x, f0.w_t, f0.b2, _row(1 + e[4], C, x.dtype), _row(e[3], C, x.dtype))
            y = _row_linear(blk.ffn[2], blk.ffn[1](u.view(B, L, -1)))
            return x + y * e[5]
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


def prepare_kernel_weights(model):
    """Kernel weight layouts, made once (both kernels take [in, out]); originals freed.
    Row-parallel biases are pre-divided by TP so the all-reduce sum adds them once.
    Call it on the host model before .to(device) (inference_neuron_i2v.py does): the layouts
    are then copied to the device once, with no device-side originals or transposition
    temporaries left in the allocator cache. FastWanDiT calls it again; the second call is a
    no-op."""
    if getattr(model, "_kernel_weights_ready", False):
        return
    model._kernel_weights_ready = True
    tp = get_tp_world_size()

    def buf(mod, name, t):
        mod.register_buffer(name, t.contiguous(), persistent=False)

    for blk in model.blocks:
        if FAST_FFNUP_NKI:
            f0 = blk.ffn[0]
            buf(f0, "w_t", f0.weight.t())                                     # [H, I local]
            buf(f0, "b2", f0.bias.view(1, -1))
            f0.weight.data = torch.empty(0, dtype=f0.weight.dtype, device=f0.weight.device)
        if FAST_CROSSQ_NKI:
            ca = blk.cross_attn
            buf(ca, "q_wt", ca.q.weight.t())                                  # [H, N*D local]
            buf(ca, "q_b2", ca.q.bias.view(1, -1))
            n3 = blk.norm3
            buf(n3, "w_row", n3.weight.view(1, -1))
            buf(n3, "b_row", n3.bias.view(1, -1))
            ca.q.weight.data = torch.empty(0, dtype=ca.q.weight.dtype, device=ca.q.weight.device)
        if FAST_QKV_NKI:
            sa = blk.self_attn
            if FAST_QKV_SPLIT:   # one [H, D] weight per projection
                for nm, lin in (("q", sa.q), ("k", sa.k), ("v", sa.v)):
                    buf(sa, f"{nm}_wt", lin.weight.t())
                    buf(sa, f"{nm}_b2", lin.bias.view(1, -1))
            else:
                w = torch.cat([sa.q.weight, sa.k.weight, sa.v.weight], 0)        # [I, H]
                buf(sa, "qkv_wt", w.t())                                          # [H, I]
                buf(sa, "qkv_b2", torch.cat([sa.q.bias, sa.k.bias, sa.v.bias], 0).view(1, -1))
            sa.qkv_split = (sa.q.weight.shape[0], sa.k.weight.shape[0], sa.v.weight.shape[0])
            for lin in (sa.q, sa.k, sa.v):
                lin.weight.data = torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device)
        elif FAST_QKV:
            sa = blk.self_attn
            w = torch.cat([sa.q.weight, sa.k.weight, sa.v.weight], 0).contiguous()
            b = torch.cat([sa.q.bias, sa.k.bias, sa.v.bias], 0).contiguous()
            buf(sa, "qkv_w", w)
            buf(sa, "qkv_b", b)
            sa.qkv_split = (sa.q.weight.shape[0], sa.k.weight.shape[0], sa.v.weight.shape[0])
            for lin in (sa.q, sa.k, sa.v):                       # free the originals
                lin.weight.data = torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device)
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


# FAST_EMBED_HOST=1: the text embedding and the time embedding / projection (232M replicated
# params per expert, 0.43 GiB per rank in bf16) live on the host and visit the device only to run:
# set_context() (once per prompt) and cache_timesteps() (once per schedule: the time modulation
# depends on t alone, so each step's e / e0 is computed once by the same graph and reused).
# Same bits as computing them every step; frees 0.86 GiB per rank for both experts, which a
# single device at LNC1 (12 GiB per core) needs.
FAST_EMBED_HOST = os.environ.get("FAST_EMBED_HOST", "0") == "1"
# DUMP_BLOCK_IN=<dir>: rank 0 saves real block inputs for scripts/validate/block_bench.py on each
# expert's first call (its text context, e / e0, and x before the blocks in DUMP_BLOCK_LAYERS).
DUMP_BLOCK_IN = os.environ.get("DUMP_BLOCK_IN", "")
DUMP_BLOCK_LAYERS = [int(v) for v in os.environ.get("DUMP_BLOCK_LAYERS", "0").split(",") if v]
# NAN_CHECK_DIR=<dir> (debug, syncs every block): log the first block whose input is not finite and
# save the previous block's input (rank 0) there, to replay the failing block in the block bench.
NAN_CHECK_DIR = os.environ.get("NAN_CHECK_DIR", "")
# FAST_SP=1 (TP > 1, with FAST_QKV_NKI + FAST_QKV_SPLIT + FAST_OPROJ + FAST_CROSSQ_NKI + FAST_ADALN +
# FAST_MLP): Megatron sequence parallelism inside the TP group. Between blocks each rank keeps only
# its B*L/tp rows of the residual stream: the blocks all-gather before the q / k / v, cross-q and
# FFN kernels and reduce-scatter after the two O-projections and the FFN (the same bytes as the
# three all-reduces), so the residual adds, gates and the norm2 kernel run on a quarter of the
# rows. _FastPatchSP keeps the rank's rows; _FastHeadSP runs the head on them and all-gathers.
FAST_SP = os.environ.get("FAST_SP", "0") == "1"
# FAST_SP_LN=1 (with FAST_SP): the two LayerNorms before the kernels that need every row also run
# on this rank's rows, in the adaln kernel, and are all-gathered instead of x: norm1 + modulation
# (then three qkv kernel calls without their fused LayerNorm, which each call would recompute on
# all rows) and the gated residual + norm3 (res_gate; then the cross-q kernel without a norm).
FAST_SP_LN = os.environ.get("FAST_SP_LN", "0") == "1"


def build_kernels():
    """Build the kernel wrappers the FAST_* flags use BEFORE tracing (an import or an nki.jit
    closure created inside a trace breaks fullgraph)."""
    if FAST_QKV_NKI or FAST_CROSSQ_NKI or FAST_FFNUP_NKI:
        _get_qkv()
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


class FastWanDiT:
    """Batched-CFG, fullgraph-compiled forward over one TP-sharded expert.

        fast = FastWanDiT(model, grid=(F, H/2, W/2), guide_scale=g, device=dev)
        fast.set_context(ctx_cond, ctx_null)     # [1, 512, 4096] each, once per prompt
        fast.cache_timesteps(ts)                 # optional: e / e0 for these t, computed once
        noise_pred = fast(latent, y, t)          # latent [16, F, H, W], y [20, F, H, W]
    """

    def __init__(self, model, grid, guide_scale, device, name=""):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        self.model = model
        self.name = name
        self._dump_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        self._dump = bool(DUMP_BLOCK_IN) and self._dump_rank0
        self._nan_seen, self._prev_x = False, None
        self.grid = tuple(int(g) for g in grid)
        self.device = device
        self.cos, self.sin = _rope_tables(model.freqs, self.grid, model.dim // model.num_heads, device)
        build_kernels()
        prepare_kernel_weights(model)
        self.time = torch.compile(_FastTime(model), **COMPILE_KW)
        self.text = torch.compile(_FastText(model.text_embedding), **COMPILE_KW)
        self.cross_kv = [torch.compile(_FastCrossKV(b), **COMPILE_KW) for b in model.blocks]
        if FAST_SP:          # residual stream sharded by token between blocks
            assert not (DUMP_BLOCK_IN or NAN_CHECK_DIR), "DUMP_BLOCK_IN / NAN_CHECK_DIR need FAST_SP=0"
            shape = (2, math.prod(self.grid))                      # (CFG batch, L)
            self.patch = torch.compile(_FastPatchSP(model.patch_embedding), **COMPILE_KW)
            self.blocks = [torch.compile(_FastBlockSP(b, shape), **COMPILE_KW) for b in model.blocks]
            self.head = torch.compile(
                _FastHeadSP(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **COMPILE_KW)
        else:
            self.patch = torch.compile(_FastPatch(model.patch_embedding), **COMPILE_KW)
            self.blocks = [torch.compile(_FastBlock(b), **COMPILE_KW) for b in model.blocks]
            self.head = torch.compile(
                _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **COMPILE_KW)
        self.context = None
        self._time_cache = {}
        self._embed_on_host = False
        if FAST_EMBED_HOST:
            self._embed_to(torch.device("cpu"))
        logger.info(f"[fast_dit] grid={self.grid} L={math.prod(self.grid)} attn={FAST_ATTN_IMPL} "
                    f"fullgraph=True blocks={len(self.blocks)} cfg_batch=2 guide={guide_scale} "
                    f"embed_host={FAST_EMBED_HOST} sp={FAST_SP}")

    def _embed_to(self, device):
        """Move the text / time embedding modules between the host and the device."""
        m = self.model
        for mod in (m.text_embedding, m.time_embedding, m.time_projection):
            mod.to(device)
        self._embed_on_host = device.type == "cpu"
        if self._embed_on_host:
            import gc
            gc.collect()
            torch.neuron.empty_cache()

    def _on_device(self, fn):
        """Run fn with the embedding modules on the device (FAST_EMBED_HOST moves them back after)."""
        if not self._embed_on_host:
            return fn()
        self._embed_to(self.device)
        try:
            return fn()
        finally:
            self._embed_to(torch.device("cpu"))

    def set_context(self, ctx_cond, ctx_null):
        def run():
            context = self.text(torch.cat([ctx_cond, ctx_null], dim=0).to(torch.bfloat16))
            if self._dump:
                os.makedirs(DUMP_BLOCK_IN, exist_ok=True)
                torch.save(context.cpu(), os.path.join(DUMP_BLOCK_IN, f"{self.name}_ctx.pt"))
            return [kv(context) for kv in self.cross_kv]
        self.context = self._on_device(run)

    def _time_embed(self, t):
        from wan.modules.model import sinusoidal_embedding_1d
        s = sinusoidal_embedding_1d(self.model.freq_dim, torch.tensor([float(t)], dtype=torch.float64))
        return self.time(s.to(torch.bfloat16).view(1, 1, -1).to(self.device))

    def cache_timesteps(self, ts):
        """Compute and keep e / e0 for every t in ts (one device visit for the embedding weights)."""
        todo = [float(t) for t in ts if float(t) not in self._time_cache]
        if todo:
            self._time_cache.update(self._on_device(lambda: {t: self._time_embed(t) for t in todo}))

    def __call__(self, latent, y, t):
        if float(t) in self._time_cache:
            e, e0 = self._time_cache[float(t)]
        elif self._embed_on_host:
            self.cache_timesteps([t])
            e, e0 = self._time_cache[float(t)]
        else:
            e, e0 = self._time_embed(t)
        x = self.patch(latent.to(torch.bfloat16), y.to(torch.bfloat16))
        dump, self._dump = self._dump, False
        if dump:
            torch.save({"e": e.cpu(), "e0": e0.cpu(), "t": float(t)}, os.path.join(DUMP_BLOCK_IN, f"{self.name}_e.pt"))
        for i, (blk, (ck, cv)) in enumerate(zip(self.blocks, self.context)):
            if dump and i in DUMP_BLOCK_LAYERS:
                torch.save(x.cpu(), os.path.join(DUMP_BLOCK_IN, f"{self.name}_L{i}_x.pt"))
            if NAN_CHECK_DIR and not self._nan_seen and not bool(torch.isfinite(x).all()):
                self._nan_seen = True       # NAN_CHECK_DIR (debug): first non-finite block input
                logger.warning(f"[fast_dit] {self.name} t={float(t):.1f}: non-finite input to block {i}")
                if self._dump_rank0:
                    os.makedirs(NAN_CHECK_DIR, exist_ok=True)
                    torch.save({"x": self._prev_x, "e0": e0.cpu(), "layer": i - 1, "t": float(t)},
                               os.path.join(NAN_CHECK_DIR, f"{self.name}_bad.pt"))
            if NAN_CHECK_DIR and not self._nan_seen:
                self._prev_x = x.cpu()
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


# ─── FAST_SP: sequence-parallel residual stream (defined last: the units above keep their lines) ──

def _sp_check():
    assert get_tp_world_size() > 1, "FAST_SP needs TP > 1"
    assert FAST_QKV_NKI and FAST_QKV_SPLIT and FAST_OPROJ and FAST_CROSSQ_NKI and FAST_ADALN and FAST_MLP, \
        "FAST_SP supports FAST_QKV_NKI + FAST_QKV_SPLIT + FAST_OPROJ + FAST_CROSSQ_NKI + FAST_ADALN + FAST_MLP"
    assert not (FAST_FFNUP_NKI or FAST_XRES_KERNEL) and FAST_ATTN_IMPL == "cte", \
        "FAST_SP: FAST_FFNUP_NKI, FAST_XRES_KERNEL off and FAST_ATTN_IMPL=cte"


_qkv_plain_call = None


def _get_qkv_plain():
    """nkilib qkv without a fused norm: input [B, S, H] @ weight [H, I] + bias [1, I] -> [B, S, I]."""
    global _qkv_plain_call
    if _qkv_plain_call is None:
        import nki
        from nkilib.core.qkv.qkv import qkv
        from nkilib.core.utils.common_types import QKVOutputLayout

        @nki.jit
        def _qkv_plain_kernel(hidden, weight, bias):
            return qkv(hidden, weight, QKVOutputLayout.BSD, bias=bias)

        _qkv_plain_call = _wrap(_qkv_plain_kernel, "wan14b::qkv_plain")
        logger.info(f"[fast_dit] nkilib qkv (no norm) via {FAST_KERNEL_WRAP} lnc={LNC}")
    return _qkv_plain_call


def _sp_rows(BL):
    """This rank's rows of the flattened [B*L, C] residual stream (rank r: rows r*S .. r*S + S)."""
    tp = get_tp_world_size()
    assert BL % tp == 0, f"FAST_SP: B*L = {BL} not divisible by TP {tp}"
    S = BL // tp
    return get_tp_rank() * S, S


class _FastPatchSP(nn.Module):
    """_FastPatch, then this rank's rows [B*L/tp, C] of the [2, L, C] CFG batch."""

    def __init__(self, patch_embedding):
        super().__init__()
        self.patch_embedding = patch_embedding

    def forward(self, latent, y):
        x = self.patch_embedding(torch.cat([latent, y], dim=0).unsqueeze(0)).flatten(2).transpose(1, 2)
        L, C = x.shape[1], x.shape[2]
        r0, S = _sp_rows(2 * L)
        return x.expand(2, -1, -1).reshape(2 * L, C)[r0:r0 + S].contiguous()


class _FastBlockSP(nn.Module):
    """_FastBlock's math with the residual stream sharded by token: xs [B*L/tp, C] in and out.
    The q / k / v, cross-q and FFN kernels take the all-gathered rows; the O-projections and the FFN
    are reduce-scattered back to this rank's rows (rank order = row order)."""

    def __init__(self, block, shape):
        super().__init__()
        _sp_check()
        if FAST_SP_LN:
            _get_qkv_plain()     # build before tracing
        self.block = block
        self.B, self.L = shape

    def forward(self, xs, e, cos, sin, ck, cv):
        blk, B, L = self.block, self.B, self.L
        sa, ca, n3, f = blk.self_attn, blk.cross_attn, blk.norm3, blk.ffn
        n, d = sa.num_heads, sa.head_dim
        g = get_tp_group()
        C = xs.shape[-1]
        dt = xs.dtype
        e = [t.reshape(1, C) for t in (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)]
        g1, b1 = _row(1 + e[1], C, dt), _row(e[0], C, dt)
        wb = ((sa.q_wt, sa.q_b2), (sa.k_wt, sa.k_b2), (sa.v_wt, sa.v_b2))
        if FAST_SP_LN:      # norm1 + modulation on this rank's rows, then plain projections
            plain, res_gate = _get_adaln()[:2]
            h = funcol.all_gather_tensor(plain(xs, g1, b1), 0, g).view(B, L, C)
            q, k, v = (_get_qkv_plain()(h, w, bb).view(B, L, -1) for w, bb in wb)
        else:
            x = funcol.all_gather_tensor(xs, 0, g).view(B, L, C)
            q, k, v = (_get_qkv()(x, w, bb, g1, b1).view(B, L, -1) for w, bb in wb)
        q, k = _tp_rms_qk(q, k, sa.norm_q, sa.norm_k)
        v = v.reshape(B, L, n, d)
        q, k = _rope(q.view(B, L, n, d), cos, sin), _rope(k.view(B, L, n, d), cos, sin)
        y = _get_oproj()(_attn_dmajor(q, k, v), sa.o_w_t, sa.o_b_scaled)
        ys = funcol.reduce_scatter_tensor(y.reshape(B * L, C), "sum", 0, g)
        if FAST_SP_LN:      # xs + gate * ys and norm3 (affine) of it on this rank's rows, one kernel
            xs, hs = res_gate(ys, _row(n3.weight, C, dt), _row(n3.bias, C, dt), xs, _row(e[2], C, torch.float32))
            h = funcol.all_gather_tensor(hs, 0, g).view(B, L, C)
            q = _get_qkv_plain()(h, ca.q_wt, ca.q_b2)
        else:
            xs = xs + (ys * e[2]).to(dt)
            x = funcol.all_gather_tensor(xs, 0, g).view(B, L, C)
            q = _get_qkv()(x, ca.q_wt, ca.q_b2, n3.w_row.to(dt), n3.b_row.to(dt))
        q = _tp_rms(q.view(B, L, -1), ca.norm_q).view(B, L, n, d)
        y = _get_oproj()(_attn_dmajor(q, ck, cv), ca.o_w_t, ca.o_b_scaled)
        ys = funcol.reduce_scatter_tensor(y.reshape(B * L, C), "sum", 0, g)
        res = _get_adaln()[2]
        xs, hs = res(ys, _row(1 + e[4], C, dt), _row(e[3], C, dt), xs)
        h = funcol.all_gather_tensor(hs, 0, g)
        y = _get_mlp()(h.reshape(1, B * L, C), f.up_w_t, f.down_w_t, f.up_b2, f.down_b_scaled)
        ys = funcol.reduce_scatter_tensor(y.reshape(B * L, C), "sum", 0, g)
        return xs + ys * e[5]


class _FastHeadSP(_FastHead):
    """_FastHead on this rank's rows, then the all-gather of the (64-wide) head output."""

    def forward(self, xs, e):                   # xs [B*L/tp, C], e [1, 1, C]
        hd = self.head
        f, h, w = self.grid
        p, q, r = self.patch_size
        c = self.out_dim
        C = xs.shape[-1]
        m = [t.reshape(1, C) for t in (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)]
        ys = hd.head(hd.norm(xs) * (1 + m[1]) + m[0])                       # [S, p*q*r*c]
        y = funcol.all_gather_tensor(ys.contiguous(), 0, get_tp_group())     # [B*L, p*q*r*c]
        y = y.view(-1, f, h, w, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        y = y.reshape(y.shape[0], c, f * p, h * q, w * r).to(torch.bfloat16)
        cond, uncond = y[0], y[1]
        return uncond + self.guide_scale * (cond - uncond)