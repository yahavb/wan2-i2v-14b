"""WAN_DUMP_DIR dump -> the gate's directory format (scripts/validate_gate.py --mode pixels).

python3 dump_to_gate.py <dump_dir> <out_dir>
Writes frames.npy (uint8 [F, H, W, 3]), forward_step0.pt (step-0 CFG noise prediction) and latent.pt.
"""
import os
import sys

import numpy as np
import torch

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
o = torch.load(os.path.join(src, "output.pt"))
np.save(os.path.join(out, "frames.npy"), o["frames"].numpy())
torch.save(o["latent"], os.path.join(out, "latent.pt"))
p = os.path.join(src, "step0.pt")
if os.path.exists(p):
    torch.save(torch.load(p)["noise_pred"].float(), os.path.join(out, "forward_step0.pt"))
print(f"{src} -> {out}: frames {tuple(o['frames'].shape)}")
