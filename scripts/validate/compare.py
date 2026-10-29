"""Compare dumped outputs (WAN_DUMP_DIR of inference_neuron_i2v.py) against a reference dump.

Usage: python3 compare.py <ref_dir> <dir> [<dir> ...]
Prints, per dir: latent rel-RMS vs ref, video PSNR (all frames), worst-frame PSNR, frame 0 (the
conditioning image frame) and last frame, max |diff| in LSB, and the step-0 noise_pred rel-RMS
(with whether both runs had the same step-0 input).
"""
import math
import os
import sys

import numpy as np
import torch


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def load(d):
    o = torch.load(os.path.join(d, "output.pt"))
    s0 = os.path.join(d, "step0.pt")
    return o["latent"].float(), o["frames"].numpy(), (torch.load(s0) if os.path.exists(s0) else None)


def main():
    ref_dir, dirs = sys.argv[1], sys.argv[2:]
    ref_lat, ref_fr, ref_s0 = load(ref_dir)
    print(f"ref {ref_dir}: latent {tuple(ref_lat.shape)} frames {ref_fr.shape}")
    for d in dirs:
        lat, fr, s0 = load(d)
        rel = ((lat - ref_lat).pow(2).mean().sqrt() / ref_lat.pow(2).mean().sqrt()).item()
        per = [psnr(fr[i], ref_fr[i]) for i in range(fr.shape[0])]
        lsb = int(np.abs(fr.astype(np.int16) - ref_fr.astype(np.int16)).max())
        s = ""
        if ref_s0 is not None and s0 is not None:
            same_in = torch.equal(s0["latent_in"], ref_s0["latent_in"])
            r = ref_s0["noise_pred"]
            e = ((s0["noise_pred"] - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
            s = f" | step0 noise_pred rel_rms {e:.5f} (same input: {same_in})"
        print(f"{os.path.basename(d.rstrip('/')):>24}: latent rel_rms {rel:.4f} | video PSNR "
              f"{psnr(fr, ref_fr):6.2f} dB, worst frame {min(per):6.2f} dB (#{int(np.argmin(per))}), "
              f"frame0 {per[0]:6.2f} dB, last {per[-1]:6.2f} dB, max |diff| {lsb} LSB{s}")


if __name__ == "__main__":
    main()
