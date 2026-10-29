"""fp32 CPU reference of the step-0 CFG noise prediction for Wan2.2-I2V-A14B (no TP, no bf16).

Reads <dump>/step0.pt (latent_in, y, t, guide, expert, ctx, ctx_null; written by
inference_neuron_i2v.py with WAN_DUMP_DIR and FAST_DIT=1), runs the unsharded expert that step 0
used in float32 on CPU with exact SDPA, and caches <out>/ref_fp32.pt. Then reports every given
dump's step-0 noise_pred error against it: the score of each Neuron configuration against the
true math. The math is written out here because the repo's WanModel modules cast to bf16
inside (norms, time embedding) for Neuron.
Usage: python3 ref_fp32.py <dump_with_step0> <out_dir> [<dump> ...]
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-I2V-A14B")
SUBFOLDER = {"high": "high_noise_model", "low": "low_noise_model"}


def rms(x, norm):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight


def lin(x, m):
    return F.linear(x, m.weight, m.bias)


def attn(q, k, v):  # [B, L, n, d]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def rope(x, cos, sin):  # same interleaved-pair rotation as models/fast_dit.py, fp32
    B, L, n, d = x.shape
    xp = x.reshape(B, L, n, d // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1).reshape(B, L, n, d)


def block_fwd(blk, x, e0, cos, sin, ck, cv):
    """One WanAttentionBlock, uniform timestep (e0 [1, 1, 6, C]), fp32 throughout."""
    sa, ca = blk.self_attn, blk.cross_attn
    B, L, C = x.shape
    n, d = sa.num_heads, sa.head_dim
    e = [t.squeeze(2) for t in (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)]  # [1, 1, C]
    h = F.layer_norm(x, (C,), eps=blk.norm1.eps) * (1 + e[1]) + e[0]
    q = rope(rms(lin(h, sa.q), sa.norm_q).view(B, L, n, d), cos, sin)
    k = rope(rms(lin(h, sa.k), sa.norm_k).view(B, L, n, d), cos, sin)
    v = lin(h, sa.v).view(B, L, n, d)
    x = x + lin(attn(q, k, v).reshape(B, L, C), sa.o) * e[2]
    h = F.layer_norm(x, (C,), blk.norm3.weight, blk.norm3.bias, eps=blk.norm3.eps)
    q = rms(lin(h, ca.q), ca.norm_q).view(B, L, n, d)
    x = x + lin(attn(q, ck, cv).reshape(B, L, C), ca.o)
    h = F.layer_norm(x, (C,), eps=blk.norm2.eps) * (1 + e[4]) + e[3]
    return x + blk.ffn(h) * e[5]


def main():
    src, out_dir, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    ref_path = os.path.join(out_dir, "ref_fp32.pt")
    if not os.path.exists(ref_path):
        torch.set_num_threads(int(os.environ.get("THREADS", "22")))
        torch.set_grad_enabled(False)
        from models.fast_dit import _rope_tables
        from wan.modules.model import WanModel, sinusoidal_embedding_1d
        s0 = torch.load(os.path.join(src, "step0.pt"))
        model = WanModel.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER[s0["expert"]]).eval().float()
        lat, y = s0["latent_in"].float(), s0["y"].float()
        _, nf, H, W = lat.shape
        grid = (nf, H // 2, W // 2)
        cos, sin = _rope_tables(model.freqs, grid, model.dim // model.num_heads, torch.device("cpu"))
        s = sinusoidal_embedding_1d(model.freq_dim, torch.tensor([float(s0["t"])], dtype=torch.float64))
        e = model.time_embedding(s.float().view(1, 1, -1))                 # [1, 1, C]
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))          # [1, 1, 6, C]
        context = model.text_embedding(torch.stack([s0["ctx"], s0["ctx_null"]], 0).float())
        x = model.patch_embedding(torch.cat([lat, y], 0).unsqueeze(0)).flatten(2).transpose(1, 2)
        x = x.expand(2, -1, -1).contiguous()
        for i, blk in enumerate(model.blocks):
            ca = blk.cross_attn
            ck = rms(lin(context, ca.k), ca.norm_k).view(2, -1, ca.num_heads, ca.head_dim)
            cv = lin(context, ca.v).view(2, -1, ca.num_heads, ca.head_dim)
            x = block_fwd(blk, x, e0, cos, sin, ck, cv)
            if i % 10 == 9:
                print(f"block {i} done", flush=True)
        hd = model.head
        m = [t.squeeze(2) for t in (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)]
        B, L, Cd = x.shape
        out = hd.head(F.layer_norm(x, (Cd,), eps=hd.norm.eps) * (1 + m[1]) + m[0])
        f, h, w = grid
        p, q, r = model.patch_size
        c = model.out_dim
        out = out.view(B, f, h, w, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, c, f * p, h * q, w * r)
        pred = out[1] + s0["guide"] * (out[0] - out[1])
        torch.save({"noise_pred": pred, "t": s0["t"]}, ref_path)
        print(f"wrote {ref_path} {tuple(pred.shape)}", flush=True)
    ref = torch.load(ref_path)["noise_pred"]
    for d in [src] + others:
        p = os.path.join(d, "step0.pt")
        if not os.path.exists(p):
            continue
        a = torch.load(p)["noise_pred"].float()
        e = ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"{os.path.basename(d.rstrip('/')):>24}: step0 noise_pred rel_rms vs fp32 CPU = {e:.5f}", flush=True)


if __name__ == "__main__":
    main()
