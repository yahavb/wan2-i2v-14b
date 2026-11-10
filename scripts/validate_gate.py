#!/usr/bin/env python3
"""The CPU-referenced accuracy gate (references/validation-suite.md). Exit 0 = PASS, 1 = FAIL.

Every optimization candidate C is judged against a CPU fp32 reference R, relative to how far the
baseline B is from R. Works on dumps, so it is model-agnostic.

  validate_gate.py --mode pixels --ref R/ --base B/ --cand C/ [--cand-rep C2/] [--off OFF/ --main MAIN/]
                   --floor-early-db X | --max-early-rel Y | --min-sharpness S [--early-frames 9] [--early-slice 0:3]
                   [--ref-dec-base RB/ --ref-dec-cand RC/] [--exact] [--floor-db X]
                   [--delta-db 0.5] [--delta-worst-db 1.0] [--slice 12] [--json-out verdict.json]
  validate_gate.py --mode text   --ref R/ --base B/ --cand C/ [... same D-checks ...] [--eps-top1 0.2]

G0 (pixels, required): an absolute floor on B AND C, set from user-labelled outputs against the same R
        (floor between the worst approved and the best rejected one); without it the relative rows pass
        a candidate that matches a bad baseline. --min-sharpness catches blur directly.
pixels: each dir has frames.pt|frames.npy (uint8 [N, H, W, C]) and optionally forward_<tag>.pt.
        --ref-dec-base / --ref-dec-cand: fp32 CPU decodes of B's and C's own latents (G4).
        If the full R trajectory was not run, --ref may hold only forward_<tag>.pt (G3) and the
        G1/G2 rows report N/A; say so in the result.
text:   each dir has tokens.json ({prompt: [ids]}) and optionally teacher_forced.pt
        ({prompt: {"logprobs": [L, k], "indices": [L, k], "ref_logprob": [L]}}) scored on R's tokens.
"""
import argparse
import json
import math
import os
import sys


def load_any(path):
    if path.endswith(".npy"):
        import numpy as np
        import torch
        return torch.from_numpy(np.load(path))
    import torch
    return torch.load(path, map_location="cpu")


def frames(d):
    if not d:
        return None
    for f in ("frames.pt", "frames.npy"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            t = load_any(p)
            return t["frames"] if isinstance(t, dict) else t
    return None


def sharpness(fr):
    """Mean per-frame variance of the 4-neighbour Laplacian of the gray image (detail energy)."""
    import torch
    g = fr.float().mean(-1)
    lap = (-4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1] + g[:, 1:-1, :-2] + g[:, 1:-1, 2:])
    return float(lap.flatten(1).var(dim=1).mean())


def psnr(a, b):
    mse = (a.double() - b.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def per_frame(a, b):
    return [psnr(a[i], b[i]) for i in range(a.shape[0])]


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def same_dir(a, b, files):
    import torch
    for f in files:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if os.path.exists(pa) != os.path.exists(pb):
            return False
        if os.path.exists(pa):
            if f.endswith(".json"):
                if json.load(open(pa)) != json.load(open(pb)):
                    return False
            else:
                x, y = load_any(pa), load_any(pb)
                if isinstance(x, dict):
                    if x.keys() != y.keys() or not all(torch.equal(x[k], y[k]) for k in x if hasattr(x[k], "shape")):
                        return False
                elif not (x.shape == y.shape and torch.equal(x, y)):
                    return False
    return True


def dump_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith((".pt", ".npy", ".json"))) if d and os.path.isdir(d) else []


rows = []


def check(cid, ok, detail):
    st = "N/A " if ok is None else ("PASS" if ok else "FAIL")
    rows.append({"check": cid, "status": st.strip(), "detail": detail})
    print(f"  {st}  {cid:4} {detail}")


def gate_pixels(a):
    R, B, C = frames(a.ref), frames(a.base), frames(a.cand)
    # G0: an ABSOLUTE floor on B and C. G1-G4 are relative ("C no worse than B"), so on
    # their own they pass a candidate that matches a baseline which is itself far from R
    # (measured: const-max attention on every layer, 14.6 dB early-frame PSNR vs R, passed
    # G1-G4 and was rejected by eye). The floor comes from an accurate config (exact
    # attention) measured against the same R, on the EARLY output, before an autoregressive
    # rollout amplifies small differences.
    n = a.early_frames
    if a.floor_early_db is None and a.max_early_rel is None and a.min_sharpness is None:
        check("G0", False, "no absolute floor: pass --floor-early-db and/or --max-early-rel, set from an "
              "accurate (exact-attention / fp32-like) config against this R; a relative gate alone passes "
              "a bad baseline")
    if a.floor_early_db is not None:
        if R is not None and B is not None and C is not None and min(R.shape[0], B.shape[0], C.shape[0]) >= n:
            eb, ec = psnr(B[:n], R[:n]), psnr(C[:n], R[:n])
            check("G0", eb >= a.floor_early_db and ec >= a.floor_early_db,
                  f"first {n} frames PSNR vs R: B {eb:.2f} dB, C {ec:.2f} dB (floor {a.floor_early_db})")
        else:
            check("G0", False, f"--floor-early-db needs R, B and C frames with >= {n} frames")
    if a.max_early_rel is not None:
        tag0 = sorted(f for f in dump_files(a.ref) if f.startswith("forward_") and f.endswith(".pt")
                      and not f.endswith("_inputs.pt"))
        if tag0 and all(os.path.exists(os.path.join(d, tag0[0])) for d in (a.base, a.cand)):
            r0 = load_any(os.path.join(a.ref, tag0[0]))
            if a.early_slice:
                lo, hi = (int(x) for x in a.early_slice.split(":"))
                cut = lambda t: t[:, :, lo:hi] if t.dim() == 5 else t
            else:
                cut = lambda t: t
            e0b = rel(cut(load_any(os.path.join(a.base, tag0[0]))), cut(r0))
            e0c = rel(cut(load_any(os.path.join(a.cand, tag0[0]))), cut(r0))
            check("G0", e0b <= a.max_early_rel and e0c <= a.max_early_rel,
                  f"{tag0[0]}{'[' + a.early_slice + ']' if a.early_slice else ''} rel-RMS vs R: "
                  f"B {e0b:.4f}, C {e0c:.4f} (max {a.max_early_rel})")
        else:
            check("G0", False, "--max-early-rel needs forward_<tag>.pt in R, B and C")
    if a.min_sharpness is not None:
        # G0s: detail energy relative to R, for B AND C. A blurred output loses it; a generated
        # video that drifts in pixels but stays sharp keeps it, which PSNR cannot tell apart.
        # Calibrated on user-labelled o2-14b videos: rejected (blurred) 0.17, approved 0.97-0.99.
        if R is not None and B is not None and C is not None:
            sr = sharpness(R)
            sb, sc = sharpness(B) / sr, sharpness(C) / sr
            check("G0", sb >= a.min_sharpness and sc >= a.min_sharpness,
                  f"sharpness vs R: B {sb:.3f}, C {sc:.3f} (min {a.min_sharpness})")
        else:
            check("G0", False, "--min-sharpness needs R, B and C frames")
    if R is not None and B is not None and C is not None and R.shape == B.shape == C.shape:
        pb, pc = psnr(B, R), psnr(C, R)
        fb, fc = per_frame(B, R), per_frame(C, R)
        check("G1", pc >= pb - a.delta_db, f"PSNR vs R: C {pc:.2f} dB, B {pb:.2f} dB (allowed -{a.delta_db})")
        check("G2", min(fc) >= min(fb) - a.delta_worst_db,
              f"worst frame vs R: C {min(fc):.2f} dB (#{fc.index(min(fc))}), B {min(fb):.2f} dB")
        lsb = int((C.short() - R.short()).abs().max())
        sl = " ".join(f"{psnr(C[i:i + a.slice], R[i:i + a.slice]):.1f}" for i in range(0, C.shape[0], a.slice))
        print(f"        C vs R: max |diff| {lsb} LSB; PSNR per {a.slice} frames: {sl}")
        if a.floor_db is not None:
            check("G5", pc >= a.floor_db, f"PSNR(C, R) {pc:.2f} dB vs floor {a.floor_db}")
    else:
        check("G1", None, "no full-trajectory R frames (state which parts R covers)")
        check("G2", None, "no full-trajectory R frames")
    tags = sorted({f[len("forward_"):-3] for f in dump_files(a.ref) if f.startswith("forward_") and f.endswith(".pt")
                   and not f.endswith("_inputs.pt")})
    for t in tags:
        r = load_any(os.path.join(a.ref, f"forward_{t}.pt"))
        pb_, pc_ = (os.path.join(d, f"forward_{t}.pt") for d in (a.base, a.cand))
        if os.path.exists(pb_) and os.path.exists(pc_):
            eb, ec = rel(load_any(pb_), r), rel(load_any(pc_), r)
            check("G3", ec <= a.rel_ratio * eb,
                  f"forward {t} rel-RMS vs R: C {ec:.5f}, B {eb:.5f} (allowed x{a.rel_ratio})")
    if not tags:
        check("G3", None, "no forward_<tag>.pt in R")
    rb, rc = frames(a.ref_dec_base), frames(a.ref_dec_cand)
    if rb is not None and rc is not None and B is not None and C is not None:
        vb, vc = psnr(B, rb), psnr(C, rc)
        check("G4", vc >= vb - a.delta_db, f"decoder vs fp32 CPU decode of own latents: C {vc:.2f} dB, B {vb:.2f} dB")
    else:
        check("G4", None, "no fp32 CPU decodes given")
    if a.exact:
        check("G6", C is not None and B is not None and C.shape == B.shape and bool((C == B).all()),
              "declared exact: C == B bit for bit")


def gate_text(a):
    def toks(d):
        p = os.path.join(d, "tokens.json")
        return json.load(open(p)) if os.path.exists(p) else None
    R, B, C = toks(a.ref), toks(a.base), toks(a.cand)
    if None in (R, B, C):
        check("T1", None, "tokens.json missing")
        return
    keys = sorted(R)

    def agree(X):
        em = sum(X.get(k) == R[k] for k in keys)
        div = []
        for k in keys:
            x, r = X.get(k, []), R[k]
            n = next((i for i, (u, v) in enumerate(zip(x, r)) if u != v), min(len(x), len(r)))
            div.append(n)
        return em, sum(div) / len(div)
    emb, dvb = agree(B)
    emc, dvc = agree(C)
    check("T1", emc >= emb - a.eps_prompts and dvc >= 0.9 * dvb,
          f"exact match vs R: C {emc}/{len(keys)}, B {emb}/{len(keys)}; mean first divergence C {dvc:.1f}, B {dvb:.1f} tokens")
    tf = {}
    for name, d in (("B", a.base), ("C", a.cand)):
        p = os.path.join(d, "teacher_forced.pt")
        tf[name] = load_any(p) if os.path.exists(p) else None
    if tf["B"] is None or tf["C"] is None:
        check("T2", None, "teacher_forced.pt missing")
        check("T3", None, "teacher_forced.pt missing")
    else:
        import torch

        def stats(T):
            top1 = n = 0
            kl = 0.0
            for k in keys:
                if k not in T:
                    continue
                lp, idx = T[k]["logprobs"].float(), T[k]["indices"]
                ref_ids = torch.tensor(R[k][1:len(lp) + 1]) if len(R[k]) > len(lp) else torch.tensor(R[k][:len(lp)])
                L = min(len(ref_ids), lp.shape[0])
                top1 += (idx[:L, 0] == ref_ids[:L]).sum().item()
                n += L
                kl += (-T[k]["ref_logprob"][:L].float()).sum().item()   # cross-entropy of R's token under X
            return 100 * top1 / max(n, 1), kl / max(n, 1)
        t1b, ceb = stats(tf["B"])
        t1c, cec = stats(tf["C"])
        check("T2", t1c >= t1b - a.eps_top1, f"teacher-forced top-1 on R tokens: C {t1c:.2f}%, B {t1b:.2f}%")
        check("T3", cec <= 1.1 * ceb, f"teacher-forced cross-entropy of R tokens: C {cec:.4f}, B {ceb:.4f} nats (allowed x1.1)")
    if a.exact:
        check("T5", C == B, "declared exact: tokens(C) == tokens(B)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pixels", "text"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--cand-rep")
    ap.add_argument("--off")
    ap.add_argument("--main")
    ap.add_argument("--ref-dec-base")
    ap.add_argument("--ref-dec-cand")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--floor-db", type=float)
    ap.add_argument("--floor-early-db", type=float,
                    help="G0: min PSNR vs R of the first --early-frames frames, for B AND C")
    ap.add_argument("--max-early-rel", type=float,
                    help="G0: max rel-RMS vs R of the first forward_<tag> (optionally --early-slice), B AND C")
    ap.add_argument("--early-frames", type=int, default=9)
    ap.add_argument("--rel-ratio", type=float, default=1.1,
                    help="G3: allowed C/B ratio of forward relRMS vs R; calibrate on labelled outputs")
    ap.add_argument("--min-sharpness", type=float,
                    help="G0: min Laplacian-variance ratio vs R over all frames, B AND C (blur check)")
    ap.add_argument("--early-slice", default="",
                    help="lo:hi slice of dim 2 (time) of the first forward tensor, e.g. 0:3 = first chunk")
    ap.add_argument("--delta-db", type=float, default=0.5)
    ap.add_argument("--delta-worst-db", type=float, default=1.0)
    ap.add_argument("--slice", type=int, default=12)
    ap.add_argument("--eps-prompts", type=int, default=0)
    ap.add_argument("--eps-top1", type=float, default=0.2)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    print(f"════════ ACCURACY GATE ({a.mode}, vs CPU fp32 reference {a.ref}) ════════")
    (gate_pixels if a.mode == "pixels" else gate_text)(a)
    if a.cand_rep:
        check("D1", same_dir(a.cand, a.cand_rep, dump_files(a.cand)), "C run twice: identical dumps")
    else:
        check("D1", None, "no repeat run")
    if a.off and a.main:
        files = [f for f in dump_files(a.main) if f in dump_files(a.off)]
        check("D2", bool(files) and same_dir(a.off, a.main, files), "flags off == main: identical dumps")
    else:
        check("D2", None, "no flags-off / main pair")
    fails = [r for r in rows if r["status"] == "FAIL"]
    na = [r["check"] for r in rows if r["status"] == "N/A"]
    verdict = "FAIL" if fails else "PASS"
    print(f"  VERDICT: {verdict}" + (f"  (failed: {', '.join(r['check'] for r in fails)})" if fails else "")
          + (f"  (not covered: {', '.join(na)})" if na else ""))
    if a.json_out:
        json.dump({"verdict": verdict, "mode": a.mode, "checks": rows}, open(a.json_out, "w"), indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
