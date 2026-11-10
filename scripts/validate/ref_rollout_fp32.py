"""fp32 CPU reference R of the WHOLE rollout (10 steps, both experts, CFG) + an fp32 CPU VAE decode.

python3 ref_rollout_fp32.py <dump_with_step0> <out_dir>

Reads <dump>/step0.pt (latent_in = the initial noise, y, ctx, ctx_null; written by
inference_neuron_i2v.py with WAN_DUMP_DIR and FAST_DIT=1) and runs the same schedule as the
pipeline (FlowUniPCMultistepScheduler, shift = config.sample_shift, expert switch at
config.boundary, guide per expert) with the unsharded experts in float32 on CPU and exact SDPA
(the block math of ref_fp32.py). The conditioning y is the dump's (the Neuron VAE encode); every
other number is fp32. Writes the gate's R directory (scripts/validate_gate.py --mode pixels):
  frames.npy          uint8 [F, H, W, 3], fp32 CPU decode of R's final latent
  forward_step0.pt    step-0 CFG noise prediction [16, F, H, W]
  latent.pt           R's final latent
Resumable: each step's latent is saved under <out_dir>/steps/.
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_fp32 import SUBFOLDER, MODEL_PATH, block_fwd, lin, rms  # noqa: E402

NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))


def forward(model, lat, y, t, guide, ctx2, cos, sin, grid):
    from wan.modules.model import sinusoidal_embedding_1d
    s = sinusoidal_embedding_1d(model.freq_dim, torch.tensor([float(t)], dtype=torch.float64))
    e = model.time_embedding(s.float().view(1, 1, -1))
    e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    context = model.text_embedding(ctx2)
    x = model.patch_embedding(torch.cat([lat, y], 0).unsqueeze(0)).flatten(2).transpose(1, 2)
    x = x.expand(2, -1, -1).contiguous()
    for blk in model.blocks:
        ca = blk.cross_attn
        ck = rms(lin(context, ca.k), ca.norm_k).view(2, -1, ca.num_heads, ca.head_dim)
        cv = lin(context, ca.v).view(2, -1, ca.num_heads, ca.head_dim)
        x = block_fwd(blk, x, e0, cos, sin, ck, cv)
    hd = model.head
    m = [u.squeeze(2) for u in (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)]
    B, L, Cd = x.shape
    out = hd.head(F.layer_norm(x, (Cd,), eps=hd.norm.eps) * (1 + m[1]) + m[0])
    f, h, w = grid
    p, q, r = model.patch_size
    c = model.out_dim
    out = out.view(B, f, h, w, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, c, f * p, h * q, w * r)
    return out[1] + guide * (out[0] - out[1])


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(out_dir, "steps"), exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", str(os.cpu_count()))))
    torch.set_grad_enabled(False)
    from models.fast_dit import _rope_tables
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    cfg = WAN_CONFIGS["i2v-A14B"]
    s0 = torch.load(os.path.join(src, "step0.pt"))
    lat, y = s0["latent_in"].float(), s0["y"].float()
    ctx2 = torch.stack([s0["ctx"], s0["ctx_null"]], 0).float()
    _, nf, H, W = lat.shape
    grid = (nf, H // 2, W // 2)
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1,
                                      use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS, device=torch.device("cpu"), shift=cfg.sample_shift)
    boundary = cfg.boundary * cfg.num_train_timesteps
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    models, cos = {}, None
    for i, t in enumerate(sch.timesteps):
        name = "high" if t.item() >= boundary else "low"
        guide = cfg.sample_guide_scale[1] if name == "high" else cfg.sample_guide_scale[0]
        sp = os.path.join(out_dir, "steps", f"{i:02d}.pt")
        if os.path.exists(sp):   # resume: the scheduler is multistep, so replay its state
            ck = torch.load(sp)
            noise_pred = ck["noise_pred"]
        else:
            if name not in models:
                models.clear()
                models[name] = WanModel.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER[name]).eval().float()
            m = models[name]
            if cos is None:
                cos, sin = _rope_tables(m.freqs, grid, m.dim // m.num_heads, torch.device("cpu"))
            t0 = time.time()
            noise_pred = forward(m, lat, y, t.item(), guide, ctx2, cos, sin, grid)
            torch.save({"noise_pred": noise_pred}, sp)
            print(f"step {i} t={t.item():.1f} {name} guide={guide} {time.time() - t0:.0f}s", flush=True)
        if i == 0:
            if s0["expert"] != name or abs(float(s0["t"]) - t.item()) > 1e-3:
                raise SystemExit(f"schedule mismatch: dump step 0 {s0['expert']} t={s0['t']}, R {name} t={t.item()}")
            torch.save(noise_pred, os.path.join(out_dir, "forward_step0.pt"))
        lat = sch.step(noise_pred.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)
    models.clear()
    torch.save(lat, os.path.join(out_dir, "latent.pt"))
    from wan.modules.vae2_1 import Wan2_1_VAE
    vae = Wan2_1_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.1_VAE.pth"), device=torch.device("cpu"))
    out = vae.decode([lat])[0]
    fr = ((out * 0.5 + 0.5) * 255).byte().permute(1, 2, 3, 0).contiguous().numpy()
    np.save(os.path.join(out_dir, "frames.npy"), fr)
    print(f"wrote {out_dir}: frames {fr.shape}", flush=True)


if __name__ == "__main__":
    main()
