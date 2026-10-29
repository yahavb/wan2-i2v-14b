"""Exact (untiled, fp32, CPU) Wan2.1 VAE decode of a dumped final latent: the VAE reference.

python3 vae_cpu_ref.py <dump_dir_with_output.pt> <out_dir> [<dump_dir to score> ...]
Writes <out_dir>/frames.npy (uint8 [F, H, W, 3], the pipeline's conversion: clamp, *0.5+0.5,
*255, truncate) and prints the PSNR of every given dump's frames against it. The pipeline
decodes the bf16 latent on Neuron (tiled); this decodes the same latent in fp32 on CPU, so the
difference is the VAE's own error.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-I2V-A14B")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    src, out_dir, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    out_path = os.path.join(out_dir, "frames.npy")
    if not os.path.exists(out_path):
        torch.set_num_threads(int(os.environ.get("THREADS", "22")))
        torch.set_grad_enabled(False)
        from wan.modules.vae2_1 import Wan2_1_VAE
        vae = Wan2_1_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.1_VAE.pth"), device=torch.device("cpu"))
        z = torch.load(os.path.join(src, "output.pt"))["latent"].float()
        t0 = time.time()
        out = vae.decode([z])[0]  # float32 [3, F, H, W], clamped to [-1, 1]
        print(f"[vae_cpu_ref] fp32 CPU decode {time.time() - t0:.0f}s {tuple(out.shape)}", flush=True)
        fr = ((out * 0.5 + 0.5) * 255).byte().permute(1, 2, 3, 0).contiguous().numpy()
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, fr)
    ref = np.load(out_path)
    for d in others:
        fr = torch.load(os.path.join(d, "output.pt"))["frames"].numpy()
        per = [psnr(fr[i], ref[i]) for i in range(ref.shape[0])]
        print(f"[vae_cpu_ref] {os.path.basename(d.rstrip('/'))}: PSNR {psnr(fr, ref):.2f} dB, worst frame "
              f"{min(per):.2f} dB (#{int(np.argmin(per))}), frame {len(per) // 2} {per[len(per) // 2]:.2f} dB",
              flush=True)


if __name__ == "__main__":
    main()
