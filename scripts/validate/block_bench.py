"""One Wan2.2-I2V-A14B DiT block at the pipeline's shape, TP over all ranks, with timing ablations.

  torchrun --nproc_per_node=<tp> scripts/validate/block_bench.py [variant ...]

  ref      models.fast_dit._FastBlock exactly as the pipeline runs it (current FAST_* env)
  no_ar    every all-reduce removed (identity)              -> collective cost
  no_sa    self-attention returns its v                     -> self-attention kernel cost
  no_xa    cross-attention returns its q                    -> cross-attention kernel cost
  ar_only  the 3 all-reduces of a [B, L, C] bf16 tensor alone
  rsag_only  the same as 3 reduce-scatters + all-gathers (the Megatron sequence-parallel split)
  cfg2     the two CFG halves as independent chains, stage-interleaved (all-reduce / compute overlap)
  cfg2_seq the two CFG halves one after the other (the cost of halving the kernels' rows)
  sp       models.fast_dit._FastBlockSP (FAST_SP): this rank's B*L/tp rows in and out (compares rows of rank 0)
  no_qkn   QK RMSNorm skipped (self q/k and cross q; their sum-of-squares all-reduce too)
  no_rope  RoPE skipped
  no_prep  the self-attention prologue skipped (QK-norm, RoPE, softmax scale, q / k / v permutes)
  no_gelu  the FFN's GELU skipped
The ablations change the math: timing probes only. ref prints rel_rms vs an fp32 CPU reference
(REF32=1, cached); BB_SAVE=<file> saves ref's output, BB_CMP=<file> compares it with a saved one
(bit-equality and rel_rms: A/B of a kernel change across two processes).

Inputs: REAL=<dir> written by the pipeline with DUMP_BLOCK_IN=<dir> (the block's input x at step 0,
e0, the text context of EXPERT); without it random data (the PE power limit depends on the data,
so time with REAL). Env: LAYER (0), EXPERT (high|low), REPS (20), OUT_H/OUT_W/FRAME_NUM
(368/480/21), MODEL_PATH.
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-I2V-A14B")
LAYER = int(os.environ.get("LAYER", "0"))
EXPERT = os.environ.get("EXPERT", "high")
REPS = int(os.environ.get("REPS", "20"))
REAL = os.environ.get("REAL", "")
DIM, FFN, HEADS, D, LC = 5120, 13824, 40, 128, 512
H, W, F_ = int(os.environ.get("OUT_H", "368")), int(os.environ.get("OUT_W", "480")), int(os.environ.get("FRAME_NUM", "21"))
GRID = ((F_ - 1) // 4 + 1, H // 16, W // 16)
L = GRID[0] * GRID[1] * GRID[2]
B = 2


def load_block(layer, expert):
    """WanAttentionBlock `layer` of the expert, fp32 on the CPU, from the safetensors shards."""
    from safetensors import safe_open
    from wan.modules.model import WanAttentionBlock
    sub = os.path.join(MODEL_PATH, f"{expert}_noise_model")
    wmap = json.load(open(os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    pre = f"blocks.{layer}."
    sd = {}
    for f in sorted({v for k, v in wmap.items() if k.startswith(pre)}):
        with safe_open(os.path.join(sub, f), framework="pt") as fh:
            for k in fh.keys():
                if k.startswith(pre):
                    sd[k[len(pre):]] = fh.get_tensor(k)
    blk = WanAttentionBlock(DIM, FFN, HEADS, (-1, -1), True, True, 1e-6)
    missing, unexpected = blk.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return blk.eval().requires_grad_(False), list(missing)


def ref32_out(blk32, x, e0, ctx, rank):
    """The block in fp32 on the CPU (scripts/validate/ref_fp32.py's math), cached per input set."""
    tag = os.path.basename(REAL.rstrip("/")) if REAL else "rand"
    path = f"/tmp/blk14_ref32_{EXPERT}_L{LAYER}_{tag}_{L}.pt"
    if rank == 0 and not os.path.exists(path):
        from models.fast_dit import _rope_tables
        from ref_fp32 import block_fwd, lin, rms
        from wan.modules.rope_neuron import rope_params_neuron
        torch.set_num_threads(int(os.environ.get("REF32_THREADS", "48")))
        ca = blk32.cross_attn
        c32 = ctx.float().cpu()
        ck = rms(lin(c32, ca.k), ca.norm_k).view(B, LC, HEADS, D)
        cv = lin(c32, ca.v).view(B, LC, HEADS, D)
        freqs = torch.cat([rope_params_neuron(1024, D - 4 * (D // 6)), rope_params_neuron(1024, 2 * (D // 6)),
                           rope_params_neuron(1024, 2 * (D // 6))], dim=1)
        cos, sin = _rope_tables(freqs, GRID, D, torch.device("cpu"))
        t0 = time.perf_counter()
        torch.save(block_fwd(blk32, x.float().cpu(), e0.float().cpu(), cos, sin, ck, cv), path)
        print(f"[block_bench] fp32 CPU reference {path} in {time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()
    return torch.load(path)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def cfg_halves_unit(fd, blk, interleave):
    """The block with the two CFG halves as independent chains in one graph (cfg2 / cfg2_seq).
    Same math as _FastBlock for FAST_QKV_NKI + FAST_QKV_SPLIT + FAST_OPROJ + FAST_CROSSQ_NKI +
    FAST_ADALN + FAST_MLP, cut at the three [B, L, C] all-reduces. interleave=True issues stage s
    of both halves before stage s + 1, so one half's all-reduce can overlap the other's compute."""
    assert fd.FAST_QKV_NKI and fd.FAST_QKV_SPLIT and fd.FAST_OPROJ and fd.FAST_CROSSQ_NKI
    assert fd.FAST_ADALN and fd.FAST_MLP and not fd.FAST_FFNUP_NKI and not fd.FAST_XRES_KERNEL
    sa, ca, n3, f = blk.self_attn, blk.cross_attn, blk.norm3, blk.ffn
    n, d = sa.num_heads, sa.head_dim

    def s1(x, e, cos, sin, ck, cv):
        B, L, C = x.shape
        g1, b1 = fd._row(1 + e[1], C, x.dtype), fd._row(e[0], C, x.dtype)
        q, k, v = (fd._get_qkv()(x, w, bb, g1, b1).view(B, L, -1)
                   for w, bb in ((sa.q_wt, sa.q_b2), (sa.k_wt, sa.k_b2), (sa.v_wt, sa.v_b2)))
        q, k = fd._tp_rms_qk(q, k, sa.norm_q, sa.norm_k)
        v = v.reshape(B, L, n, d)
        q, k = fd._rope(q.view(B, L, n, d), cos, sin), fd._rope(k.view(B, L, n, d), cos, sin)
        return x, fd._get_oproj()(fd._attn_dmajor(q, k, v), sa.o_w_t, sa.o_b_scaled)

    def s2(x, y, e, cos, sin, ck, cv):
        B, L, C = x.shape
        x = x + (y.reshape(B, L, C) * e[2]).to(x.dtype)
        q = fd._get_qkv()(x, ca.q_wt, ca.q_b2, n3.w_row.to(x.dtype), n3.b_row.to(x.dtype))
        q = fd._tp_rms(q.view(B, L, -1), ca.norm_q).view(B, L, n, d)
        return x, fd._get_oproj()(fd._attn_dmajor(q, ck, cv), ca.o_w_t, ca.o_b_scaled)

    def s3(x, y, e, cos, sin, ck, cv):
        B, L, C = x.shape
        res = fd._get_adaln()[2]
        x, h = res(y.reshape(B * L, C), fd._row(1 + e[4], C, x.dtype), fd._row(e[3], C, x.dtype),
                   x.reshape(B * L, C))
        y = fd._get_mlp()(h.reshape(1, B * L, -1), f.up_w_t, f.down_w_t, f.up_b2, f.down_b_scaled)
        return x.view(B, L, C), y

    def unit(x_, e_, c_, s_, k_, v_):
        e = [t.squeeze(2) for t in (blk.modulation.unsqueeze(0) + e_).chunk(6, dim=2)]
        halves = [[x_[i:i + 1], None, k_[i:i + 1], v_[i:i + 1]] for i in range(2)]

        def step(h, s):
            x, y = s(h[0], *([] if s is s1 else [h[1]]), e, c_, s_, h[2], h[3])
            h[0], h[1] = x, fd._all_reduce(y)
        if interleave:
            for s in (s1, s2, s3):
                for h in halves:
                    step(h, s)
        else:
            for h in halves:
                for s in (s1, s2, s3):
                    step(h, s)
        return torch.cat([h[0] + h[1].reshape(h[0].shape) * e[5] for h in halves], 0)
    return unit


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import models.fast_dit as fd
    from models.tp_utils import init_tp_group, shard_model_tp
    from wan.modules.rope_neuron import rope_params_neuron
    torch._dynamo.config.cache_size_limit = 64
    variants = sys.argv[1:] or ["ref", "no_ar", "no_sa", "no_xa", "ar_only"]

    blk, missing = load_block(LAYER, EXPERT)
    blk32 = None
    if os.environ.get("REF32", "0") == "1":
        import copy
        blk32 = copy.deepcopy(blk)
    m = nn.Module()
    m.blocks, m.num_heads = nn.ModuleList([blk]), HEADS
    init_tp_group(tp_degree=world)
    shard_model_tp(m, rank, world)
    m = m.to(torch.bfloat16)
    fd.build_kernels()
    fd.prepare_kernel_weights(m)
    m = m.to(dev)
    blk = m.blocks[0]

    g = torch.Generator().manual_seed(0)
    if REAL:
        x = torch.load(os.path.join(REAL, f"{EXPERT}_L{LAYER}_x.pt")).to(torch.bfloat16)
        e0 = torch.load(os.path.join(REAL, f"{EXPERT}_e.pt"))["e0"].to(torch.bfloat16)
        ctx = torch.load(os.path.join(REAL, f"{EXPERT}_ctx.pt")).to(torch.bfloat16)
    else:
        x = (torch.randn(B, L, DIM, generator=g) * 0.5).to(torch.bfloat16)
        e0 = (torch.randn(1, 1, 6, DIM, generator=g) * 0.1).to(torch.bfloat16)
        ctx = (torch.randn(B, LC, DIM, generator=g) * 0.5).to(torch.bfloat16)
    assert tuple(x.shape) == (B, L, DIM), (tuple(x.shape), (B, L, DIM))
    y32 = ref32_out(blk32, x, e0, ctx, rank) if blk32 is not None else None
    x, e0, ctx = x.to(dev), e0.to(dev), ctx.to(dev)
    freqs = torch.cat([rope_params_neuron(1024, D - 4 * (D // 6)), rope_params_neuron(1024, 2 * (D // 6)),
                       rope_params_neuron(1024, 2 * (D // 6))], dim=1)
    cos, sin = fd._rope_tables(freqs, GRID, D, dev)
    ck, cv = torch.compile(fd._FastCrossKV(blk), **fd.COMPILE_KW)(ctx)
    if rank == 0:
        print(f"[block_bench] {EXPERT} layer {LAYER} (missing {missing}) tp={world} L={L} grid={GRID} "
              f"real={bool(REAL)} attn={fd.FAST_ATTN_IMPL} lnc={fd.LNC}", flush=True)

    first = None
    for var in variants:
        saved = {k: getattr(fd, k) for k in ("_all_reduce", "_attn_dmajor", "_self_attn", "_attn",
                                             "_tp_rms_qk", "_tp_rms", "_rope")}
        gelu = blk.ffn[1]
        if var == "no_ar":
            fd._all_reduce = lambda t: t
        elif var == "no_qkn":
            fd._tp_rms_qk = lambda q, k, nq, nk: (q, k)
            fd._tp_rms = lambda t, nrm: t
        elif var == "no_rope":
            fd._rope = lambda t, c_, s_: t
        elif var == "no_prep":   # self-attention prologue: QK-norm, RoPE, scale, layout copies
            fd._tp_rms_qk = lambda q, k, nq, nk: (q, k)
            fd._rope = lambda t, c_, s_: t

            def flat(q, k, v, _o=saved["_attn_dmajor"]):
                B_, Lq_, n_, d_ = q.shape
                if k.shape[1] != Lq_:
                    return _o(q, k, v)
                q3, k3, v3 = (t.reshape(B_ * n_, Lq_, d_) for t in (q, k, v))   # no permutes
                return fd._get_cte(True)(q3, k3, v3).reshape(B_, n_, d_, Lq_)
            fd._attn_dmajor = flat
        elif var == "no_gelu":
            blk.ffn[1] = nn.Identity()
        elif var in ("no_sa", "no_xa"):
            self_side = var == "no_sa"

            def probe(q, k, v, _o=saved["_attn_dmajor"], _s=self_side):
                if (k.shape[1] == q.shape[1]) == _s:      # the ablated side: skip the kernel
                    out = (v if _s else q) * 1.0
                    return out.permute(0, 2, 3, 1).contiguous()   # d-major [B, n, d, Lq]
                return _o(q, k, v)
            fd._attn_dmajor = probe
        if var == "ar_only":
            def unit(x_, e_, c_, s_, k_, v_):
                for _ in range(3):
                    x_ = fd._all_reduce(x_) * 0.25
                return x_
        elif var == "rsag_only":   # the same 3 reductions as reduce-scatter + all-gather (Megatron SP)
            import torch.distributed._functional_collectives as funcol
            from models.tp_utils import get_tp_group

            def unit(x_, e_, c_, s_, k_, v_):
                g = get_tp_group()
                for _ in range(3):
                    sh = funcol.reduce_scatter_tensor(x_.reshape(-1, x_.shape[-1]), "sum", 0, g) * 0.25
                    x_ = funcol.all_gather_tensor(sh, 0, g).view_as(x_)
                return x_
        elif var in ("cfg2", "cfg2_seq"):
            unit = cfg_halves_unit(fd, blk, interleave=var == "cfg2")
        elif var == "sp":      # FAST_SP's block: this rank's B*L/tp rows in and out
            unit = fd._FastBlockSP(blk, (B, L))
        else:
            unit = fd._FastBlock(blk)
        f = torch.compile(unit, **fd.COMPILE_KW)
        S = B * L // world
        x_in = x.reshape(B * L, DIM)[rank * S:(rank + 1) * S].contiguous() if var == "sp" else x
        t0 = time.perf_counter()
        try:
            y = f(x_in, e0, cos, sin, ck, cv)
            torch.neuron.synchronize()
        except Exception as ex:  # noqa: BLE001
            if rank == 0:
                print(f"{var:>8}: FAILED {type(ex).__name__}: {str(ex)[:600]}", flush=True)
            for k, v in saved.items():
                setattr(fd, k, v)
            blk.ffn[1] = gelu
            continue
        t_first = time.perf_counter() - t0
        for _ in range(3):
            y = f(x_in, e0, cos, sin, ck, cv)
        torch.neuron.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            y = f(x_in, e0, cos, sin, ck, cv)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        if rank == 0:
            yc = y.float().cpu()
            acc = ""
            if var == "sp":     # rank 0 holds rows 0..S-1: compare them, as [B, L, C]-shaped refs
                if y32 is not None:
                    acc += f" rel_rms_vs_fp32(rows 0..{S - 1})={rel(yc, y32.reshape(B * L, DIM)[:S]):.5f}"
                if first is not None and first[0] == "ref":
                    acc += f" rel_rms_vs_ref(rows 0..{S - 1})={rel(yc, first[1].reshape(B * L, DIM)[:S]):.5f}"
            else:
                if var == "ref" and y32 is not None:
                    acc += f" rel_rms_vs_fp32={rel(yc, y32):.5f}"
                if var == "ref" and os.environ.get("BB_CMP"):      # another process's saved ref output
                    yo = torch.load(os.environ["BB_CMP"])
                    acc += f" vs_BB_CMP: equal={torch.equal(yc, yo)} rel_rms={rel(yc, yo):.5f}"
                if var == "ref" and os.environ.get("BB_SAVE"):
                    torch.save(yc, os.environ["BB_SAVE"])
                if first is None:
                    first = (var, yc)
                elif var == "ref" or first[0] == "ref":
                    acc += f" rel_rms_vs_{first[0]}={rel(yc, first[1]):.5f}"
            print(f"{var:>8}: {ms:8.3f} ms/block (first call {t_first:.1f}s){acc}", flush=True)
        for k, v in saved.items():
            setattr(fd, k, v)
        blk.ffn[1] = gelu
        torch._dynamo.reset()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
