"""Sequence Parallel self-attention for WanModel.

With TP=4 SP=2 (8 ranks total), the NKI self-attention kernel supports
seq_q != seq_k natively (q padded to 128, k padded to SECTION (2048) multiples).

SP attention flow:
  1. All-gather x across ALL ranks (world) → full [B, seq_len, dim]
  2. Compute Q, K, V with TP-sharded projections (full sequence)
  3. Apply RoPE to full Q, K
  4. Slice Q by SP rank: Q_local = Q[:, sp_start:sp_end]
  5. NKI self-attention: Q_local (seq_len//sp) @ K_full, V_full
  6. O projection (RowParallel — all-reduce across TP group of 4)
  7. Slice output by tp_rank within SP shard → [B, seq_len//world, dim]
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from models.parallel_state import get_sp_group, get_sp_rank, get_sp_degree
from models.tp_utils import get_tp_rank, get_tp_world_size


# MUST equal SECTION in kernels/self_attention.py and SELF_ATTN_SEQLEN_MULTIPLE
# in wan/modules/attention.py. 2048 cuts the SP K-pad waste (45%->12%).
SELF_ATTN_SEQLEN_MULTIPLE = 2048

# Cached attention mask. Shape/contents are identical across every self-attn
# call (l_q, l_k constant), so build once instead of ~800x per generation.
_mask_cache = {}


def _get_sp_mask(P, seqlen_k_padded, l_k, pad_k, dtype, device):
    key = (P, seqlen_k_padded, l_k, dtype, device)
    m = _mask_cache.get(key)
    if m is None:
        m = torch.zeros(P, seqlen_k_padded, dtype=dtype, device=device)
        if pad_k > 0:
            m[:, l_k:] = float('-inf')
        _mask_cache[key] = m
    return m


def _sp_nki_self_attention(q_local, k_full, v_full, dtype=torch.bfloat16):
    """NKI self-attention with Q shorter than K (for SP).

    q_local: [1, L_q, n, d] — SP shard of queries
    k_full:  [1, L_k, n, d] — full keys
    v_full:  [1, L_k, n, d] — full values

    Uses the self-attention kernel which supports seq_q != seq_k.
    """
    from wan.modules.attention import _nki_self_attn, _get_identity

    b, l_q, n, d = q_local.shape
    l_k = k_full.shape[1]

    # Batched CFG: kernel takes [n,d,L] (no sample-batch). Loop over B (cond+uncond)
    # and stack. (Was assert b==1; SP path now also hit at B=2 under batched CFG.)
    P = 128
    pad_q = (P - l_q % P) % P
    pad_k = (SELF_ATTN_SEQLEN_MULTIPLE - l_k % SELF_ATTN_SEQLEN_MULTIPLE) % SELF_ATTN_SEQLEN_MULTIPLE
    seqlen_k_padded = l_k + pad_k
    num_sections = seqlen_k_padded // SELF_ATTN_SEQLEN_MULTIPLE
    mask = _get_sp_mask(P, seqlen_k_padded, l_k, pad_k, dtype, q_local.device)
    identity = _get_identity(q_local.device, dtype)
    softmax_scale = 1.0 / math.sqrt(d)

    outs = []
    for bi in range(b):
        q_nki = q_local[bi].permute(1, 2, 0).contiguous()  # [n, d, L_q]
        k_nki = k_full[bi].permute(1, 2, 0).contiguous()   # [n, d, L_k]
        v_nki = v_full[bi].permute(1, 0, 2).contiguous()   # [n, L_k, d]
        if pad_q > 0:
            q_nki = F.pad(q_nki, (0, pad_q))
        if pad_k > 0:
            k_nki = F.pad(k_nki, (0, pad_k))
            v_nki = F.pad(v_nki, (0, 0, 0, pad_k))
        out_nki = _nki_self_attn(
            q_nki, k_nki, v_nki, identity, mask,
            softmax_scale=softmax_scale, num_sections=num_sections)
        outs.append(out_nki[:l_q].unsqueeze(0))  # [1, L_q, n, d]

    return torch.cat(outs, dim=0)                 # [B, L_q, n, d]


def make_sp_self_attn_forward(self_attn):
    """Create SP-aware self-attention forward."""
    from wan.modules.attention import _NKI_SELF_AVAILABLE, attention as flash_attention
    from wan.modules.rope_neuron import rope_apply_neuron, rope_apply_neuron_offset

    sp_degree = get_sp_degree()
    tp_degree = get_tp_world_size()
    world_size = sp_degree * tp_degree
    sp_rank = get_sp_rank()
    tp_rank = get_tp_rank()
    use_nki = _NKI_SELF_AVAILABLE and os.environ.get("USE_NKI_KERNELS", "1") == "1"
    # LOCAL_Q=1: project Q only on this rank's L/sp slice (half the Q-proj matmul)
    # instead of full Q then discard. K/V stay full. Offset RoPE keeps positions
    # correct. Gated so it A/Bs cleanly and reverts instantly if video regresses.
    local_q = os.environ.get("LOCAL_Q", "0") == "1"

    def forward(x, seq_lens, grid_sizes, freqs):
        b = x.shape[0]
        L_local = x.shape[1]
        dim = self_attn.dim
        n = self_attn.num_heads  # local heads after TP sharding
        d = self_attn.head_dim

        # x is the SP shard (L/sp), REPLICATED across the TP group. Gather over the
        # SP GROUP (sp_degree shards -> full L). Gather with SP-shard as the OUTER
        # dim so BATCH and sequence-shard don't interleave at B>1 (the same bug
        # fixed in sp_model.py): [sp, b, L_local, dim] -> permute -> [b, L_full, dim].
        L_full = L_local * sp_degree
        # all_gather output must be input with dim0 * sp_degree: [sp*b, L_local, dim].
        gathered = torch.empty(sp_degree * b, L_local, dim, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=get_sp_group())
        # gathered = [sp0_b0, sp0_b1, sp1_b0, sp1_b1] -> [sp, b, L_local, dim]
        # -> [b, sp, L_local, dim] -> [b, L_full, dim] (each item's shards concatenated)
        x_full = gathered.reshape(sp_degree, b, L_local, dim).permute(1, 0, 2, 3).reshape(b, L_full, dim)
        from models.sp_model import _padlog
        _padlog("sp_attn:x_full_gather", x_full)

        # K, V always full (attention needs the full sequence).
        k = self_attn.norm_k(self_attn.k(x_full)).view(b, L_full, n, d)
        v = self_attn.v(x_full).view(b, L_full, n, d)
        k = rope_apply_neuron(k, grid_sizes, freqs)

        sp_start = sp_rank * L_local
        if local_q:
            # Project Q ONLY on this rank's slice — half the Q-proj matmul. Apply
            # RoPE at the matching position offset so rotations are identical to
            # the full-Q path. x_full[:, sp_start:...] is this rank's tokens.
            x_q = x_full[:, sp_start:sp_start + L_local]
            q_local = self_attn.norm_q(self_attn.q(x_q)).view(b, L_local, n, d)
            q_local = rope_apply_neuron_offset(q_local, grid_sizes, freqs, sp_start)
        else:
            # Full Q then slice (original path).
            q = self_attn.norm_q(self_attn.q(x_full)).view(b, L_full, n, d)
            q = rope_apply_neuron(q, grid_sizes, freqs)
            q_local = q[:, sp_start:sp_start + L_local]

        # Attention: Q shard against full K, V
        if use_nki and q_local.device.type == "neuron":
            out = _sp_nki_self_attention(q_local, k, v)
        else:
            # Fallback: SDPA (works with different Q/K lengths)
            q_t = q_local.transpose(1, 2).to(torch.bfloat16)
            k_t = k.transpose(1, 2).to(torch.bfloat16)
            v_t = v.transpose(1, 2).to(torch.bfloat16)
            out = F.scaled_dot_product_attention(q_t, k_t, v_t)
            out = out.transpose(1, 2).contiguous()

        # O projection (RowParallel: matmul + all-reduce across TP group). Valid
        # because all TP-group ranks hold the SAME L/sp tokens here.
        out = out.flatten(2)            # [B, L_local, n*d]
        out = self_attn.o(out)          # [B, L_local, dim]

        # Return this rank's L/sp shard unchanged — TP ranks share it, SP ranks
        # hold complementary halves (gathered back in sp_model head step).
        return out

    return forward


def patch_model_for_sp(model):
    """Patch all self-attention blocks to use SP."""
    sp_degree = get_sp_degree()
    if sp_degree <= 1:
        return model

    for block in model.blocks:
        block.self_attn.forward = make_sp_self_attn_forward(block.self_attn)

    print(f"[SP] Patched {len(model.blocks)} blocks for SP={sp_degree}")
    return model
