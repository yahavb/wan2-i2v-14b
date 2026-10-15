#!/usr/bin/env python3
"""Identify the DiT NEFFs in the compile cache by WHAT IS IN THEM, not file size.

Adapted from odyssey-trainium/bringup/scripts/neuron/pick_dit_neff.py. Unlike
odyssey, the Wan2.2 DiT is NOT one fullgraph NEFF: only block.ffn,
patch_embedding, text_embedding and head are torch.compile'd; self/cross
attention run eagerly through the NKI kernels. So the DiT is a SET of NEFFs,
and we identify each one by the tensor shapes in its IO signature
(NeffInfo metadata -- never `strings | grep`, NEFFs are binary):

  ffn        : a dim == ffn_dim/TP            (13824/4 = 3456)
  attn_kernel: a dim == num_heads/TP AND a dim == head_dim  ([n, d, L] NKI layout)
  dit_other  : a dim == model dim             (5120: qkv/o proj, norms, embed, head)

T5 (dim 4096, head_dim 64) and the VAE (channels <= 384, 5-D convs) match none.

Output order: ffn, attn_kernel, dit_other; inside each role largest first.
The ffn / attention NEFFs run 40 blocks x NUM_STEPS per generation
(see neff-time-accounting), so they are the ones that matter.

Usage:
    python3 pick_dit_neffs.py <neff_dir> [--tp 4]            # report
    python3 pick_dit_neffs.py <neff_dir> --print-paths       # role<TAB>ws<TAB>path
Exit 1 if no DiT NEFF found.
"""
import argparse
import glob
import hashlib
import os
import sys

from neuronxcc.kra.NeffInfo import NeffInfo

DIM, FFN_DIM, NUM_HEADS = 5120, 13824, 40   # wan/configs/wan_i2v_A14B.py
HEAD_DIM = DIM // NUM_HEADS                  # 128
ROLE_ORDER = {"ffn": 0, "attn_kernel": 1, "dit_other": 2}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path, tp):
    try:
        ni = NeffInfo(path)
        sig = ni.getNeffJson()["metadata"]["signatures"]["default"]
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": f"{type(e).__name__}: {e}"}
    tensors = list(sig.get("inputs", {}).values()) + list(sig.get("outputs", {}).values())
    shapes = [tuple(t.get("shape") or ()) for t in tensors]
    dims = {d for s in shapes for d in s}

    if FFN_DIM // tp in dims:
        role = "ffn"
    elif any((NUM_HEADS // tp) in s and HEAD_DIM in s for s in shapes):
        role = "attn_kernel"
    elif DIM in dims:
        role = "dit_other"
    else:
        return None

    try:
        ws = ni.getCCRankWorldSize() if ni.hasCC() else 1
    except Exception:
        ws = 0  # unknown -> caller assumes collectives (safe)
    return {
        "path": path, "role": role, "ws": ws,
        "size": os.path.getsize(path),
        "n_in": len(sig.get("inputs", {})), "n_out": len(sig.get("outputs", {})),
        "shapes": shapes[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neff_dir", nargs="+")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP_DEGREE", "4")))
    ap.add_argument("--print-paths", action="store_true")
    args = ap.parse_args()

    neffs = sorted({p for d in args.neff_dir
                    for p in glob.glob(os.path.join(d, "**/*.neff"), recursive=True)})
    # The cache holds the same graph once per rank; probe each content once.
    uniq = {}
    for p in neffs:
        uniq.setdefault(md5(p), p)

    cands, errors = [], []
    for p in uniq.values():
        r = probe(p, args.tp)
        if r is None:
            continue
        (errors if "error" in r else cands).append(r)
    cands.sort(key=lambda c: (ROLE_ORDER[c["role"]], -c["size"]))

    if args.print_paths:
        for c in cands:
            print(f"{c['role']}\t{c['ws']}\t{c['path']}")
        return 0 if cands else 1

    print(f"scanned {len(neffs)} .neff ({len(uniq)} unique by content), TP={args.tp}")
    for e in errors:
        print(f"  unreadable: {os.path.basename(e['path'])} -- {e['error']}")
    if not cands:
        print("NO DiT NEFF FOUND (no NEFF has a dim of "
              f"{FFN_DIM // args.tp}, {DIM}, or [{NUM_HEADS // args.tp}, {HEAD_DIM}, L])")
        return 1
    print(f"DiT NEFFs: {len(cands)}")
    for c in cands:
        print(f"  {c['role']:<11} ws={c['ws']:<2} {c['size'] / 1e6:7.1f} MB  "
              f"io={c['n_in']}/{c['n_out']}  {os.path.basename(c['path'])}  {c['shapes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
