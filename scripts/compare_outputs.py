#!/usr/bin/env python3
"""Accuracy-gate metrics between a candidate output and a reference.

Usage: compare_outputs.py CANDIDATE REFERENCE [--exact] [--psnr-peak P] [--slice-axis A]
                          [--min-psnr DB] [--min-slice-psnr DB] [--max-rel-rms X]
Files: .npy (numpy), .pt/.pth (a torch tensor, or a dict of tensors: every shared tensor key is
compared). Any other file (an mp4, a raw dump) is byte-compared only. --exact means identical
values and dtype for arrays, identical bytes for other files.
Prints byte identity, shape and dtype, max abs error, rel-RMS (||c - r|| / ||r||), and PSNR over
the whole array and its worst slice along --slice-axis (for example the frame axis of a video).
PSNR needs a peak: 255 for uint8 data by default, otherwise pass --psnr-peak (2 for [-1, 1]).
Exit 0 = pass, 1 = a threshold failed or --exact and the files differ, 2 = not comparable.
Do not gate on cosine similarity: it is dominated by the bulk of the output and hides
localized errors, so this script does not report it.
"""
import argparse
import hashlib
import math
import os
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        import numpy as np
        return {"": np.load(path)}
    if ext in (".pt", ".pth"):
        import torch
        def np_of(t):  # numpy has no bfloat16: widen it (exact) before converting
            t = t.detach().cpu()
            return (t.float() if t.dtype == torch.bfloat16 else t).numpy()
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            return {k: np_of(v) if hasattr(v, "detach") else v for k, v in obj.items()}
        return {"": np_of(obj)}
    return None


def to64(x):
    import numpy as np
    return np.asarray(x, dtype=np.float64)


def psnr(mse, peak):
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def compare(c, r, a, key):
    import numpy as np
    label = key or "array"
    if not hasattr(c, "shape") or not hasattr(r, "shape"):
        same = c == r
        print(f"{label}: non-array values {'equal' if same else 'DIFFER'} ({c!r} vs {r!r})")
        return same or not a.exact, True
    if c.shape != r.shape:
        print(f"{label}: shape {tuple(c.shape)} vs {tuple(r.shape)}: not comparable")
        return False, False
    equal = c.dtype == r.dtype and bool(np.array_equal(c, r))
    c64, r64 = to64(c), to64(r)
    d = c64 - r64
    rms_r = math.sqrt(float((r64 ** 2).mean())) or 1e-30
    rel = math.sqrt(float((d ** 2).mean())) / rms_r
    dt = f"{c.dtype}" if c.dtype == r.dtype else f"{c.dtype} vs {r.dtype}"
    line = (f"{label}: shape {tuple(c.shape)} {dt}; identical values {equal}; max abs {float(np.abs(d).max()):.6g}; "
            f"rel-RMS {rel:.6g}")
    ok = True
    peak = a.psnr_peak if a.psnr_peak is not None else (255.0 if c.dtype == np.uint8 else None)
    if peak is not None:
        p = psnr(float((d ** 2).mean()), peak)
        line += f"; PSNR {p:.2f} dB"
        if a.min_psnr is not None and p < a.min_psnr:
            ok = False
        if a.slice_axis is not None and c.ndim > 1:
            ax = a.slice_axis % c.ndim
            axes = tuple(i for i in range(c.ndim) if i != ax)
            per = [psnr(float(m), peak) for m in (d ** 2).mean(axis=axes)]
            worst = min(range(len(per)), key=lambda i: per[i])
            line += f"; worst slice {per[worst]:.2f} dB (#{worst} on axis {ax})"
            if a.min_slice_psnr is not None and per[worst] < a.min_slice_psnr:
                ok = False
    if a.max_rel_rms is not None and rel > a.max_rel_rms:
        ok = False
    if a.exact and not equal:
        ok = False
    print(line)
    return ok, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--exact", action="store_true",
                    help="fail unless identical: same values and dtype for arrays, same bytes otherwise")
    ap.add_argument("--psnr-peak", type=float, default=None)
    ap.add_argument("--slice-axis", type=int, default=None)
    ap.add_argument("--min-psnr", type=float, default=None)
    ap.add_argument("--min-slice-psnr", type=float, default=None)
    ap.add_argument("--max-rel-rms", type=float, default=None)
    a = ap.parse_args()
    same_bytes = sha(a.candidate) == sha(a.reference)
    print(f"byte-identical files: {same_bytes}")
    c, r = load(a.candidate), load(a.reference)
    if c is None or r is None:  # not arrays: bytes are all there is to compare
        ok = same_bytes or not a.exact
        print("GATE: PASS" if ok else "GATE: FAIL")
        return 0 if ok else 1
    keys = [k for k in c if k in r]
    if not keys:
        print("no shared keys to compare")
        return 2
    ok_all, comparable = True, True
    for k in keys:
        ok, cmp_ok = compare(c[k], r[k], a, k)
        ok_all, comparable = ok_all and ok, comparable and cmp_ok
    if not comparable:
        return 2
    print("GATE: PASS" if ok_all else "GATE: FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
