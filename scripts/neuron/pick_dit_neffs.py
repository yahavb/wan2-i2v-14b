#!/usr/bin/env python3
"""Identify the DiT NEFFs in the compile cache by WHAT IS IN THEM, not file size.

Adapted from odyssey-trainium/bringup/scripts/neuron/pick_dit_neff.py. Unlike
odyssey, the Wan2.2 DiT is NOT one fullgraph NEFF: only block.ffn,
patch_embedding, text_embedding and head are torch.compile'd; self/cross
attention run eagerly through the NKI kernels. So the DiT is a SET of NEFFs,
and we identify each one by the tensor shapes in its IO signature
(NeffInfo metadata -- never `strings | grep`, NEFFs are binary):

  attn_kernel: the embedded HLO debug text names an NKI attention kernel
               (wan_flash_self_attn / wan_cross_attn). Shapes can't find these:
               activations are flattened to 2-D (B, numel) in the signature.
  attn_path  : the ops AROUND the kernel in self/cross attention (q/k/v/o proj,
               qk-norm, RoPE, permute/contiguous, Q/K/V pad, output slice/cat).
               Matched on the attention layout, with H = heads/TP * head_dim:
                 - a [heads/TP, head_dim * t] tensor (kernel-side [n, d, L] layout)
                 - a row of exactly t * H elements (q/k/v/attn-out activations)
                 - an [H, DIM] or [DIM, H] weight (q/k/v/o projections)
               for t in the attention token counts (full / SP-local / padded /
               text context), derived from --seq/--sp/--ctx.
  collective : has collectives AND a row of t * DIM or t * H elements (SP gathers,
               TP all-reduce). VAE collectives match no DiT token count.
  nki_kernel : an unnamed custom-call WITH matmul work (HloMacCount > 0). The
               zero-MAC custom-calls not on the attention path are skipped.
  ffn        : a dim == ffn_dim/TP            (13824/4 = 3456)
  dit_other  : a dim == model dim             (5120: qkv/o proj, norms, embed, head)

T5 (dim 4096, head_dim 64) and the VAE (channels <= 384, 5-D convs) match none.

Output order: attn_kernel, attn_path, collective, nki_kernel, ffn, dit_other;
inside each role most MACs first
(HloMacCount from the NEFF's hlo_stats.json), then largest file.
The ffn / attention NEFFs run 40 blocks x NUM_STEPS per generation
(see neff-time-accounting), so they are the ones that matter.

Usage:
    python3 pick_dit_neffs.py <neff_dir> [--tp 4 --sp 2 --seq 4140 --ctx 512]   # report
    python3 pick_dit_neffs.py <neff_dir> --print-paths       # role<TAB>ws<TAB>path
Exit 1 if no DiT NEFF found.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import struct
import sys
import tarfile

from neuronxcc.kra.NeffInfo import NeffInfo

DIM, FFN_DIM, NUM_HEADS = 5120, 13824, 40   # wan/configs/wan_i2v_A14B.py
HEAD_DIM = DIM // NUM_HEADS                  # 128
Q_TILE, K_SECTION = 128, 2048                # models/sp_attention.py pad multiples
ROLE_ORDER = {"attn_kernel": 0, "attn_path": 1, "collective": 2,
              "nki_kernel": 3, "ffn": 4, "dit_other": 5}
ATTN_ROLES = ("attn_kernel", "attn_path", "collective")


def ceil_to(x, m):
    return (x + m - 1) // m * m


def attn_tokens(seq, sp, tp, ctx):
    """Every sequence length an attention tensor can have (see sp_attention.py)."""
    full = ceil_to(seq, sp * tp)           # sp_model pads to a multiple of world size
    local = full // sp
    return {seq, full, local, ceil_to(local, Q_TILE), ceil_to(full, K_SECTION),
            ctx, ceil_to(ctx, Q_TILE)}
KERNEL_NAMES = ("wan_flash_self_attn", "wan_cross_attn")


def neff_contents(path):
    """(HLO debug text, MAC count) from the tar.gz embedded after the NEFF header.
    Header size is the u64 at offset 8 (1024 on this SDK)."""
    text, macs = "", -1
    try:
        with open(path, "rb") as f:
            raw = f.read()
        hdr = struct.unpack_from("<Q", raw, 8)[0] or 1024
        with tarfile.open(fileobj=io.BytesIO(raw[hdr:]), mode="r:gz") as tf:
            for m in tf.getmembers():
                name = os.path.basename(m.name)
                if name == "hlo_stats.json":
                    macs = json.load(tf.extractfile(m)).get("HloMacCount", -1)
                elif name.startswith("debug_info_hlo"):
                    text += tf.extractfile(m).read().decode("latin-1")
    except Exception:  # noqa: BLE001
        pass
    return text, macs


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path, tp, tokens):
    try:
        ni = NeffInfo(path)
        sig = ni.getNeffJson()["metadata"]["signatures"]["default"]
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": f"{type(e).__name__}: {e}"}
    tensors = list(sig.get("inputs", {}).values()) + list(sig.get("outputs", {}).values())
    shapes = [tuple(t.get("shape") or ()) for t in tensors]
    dims = {d for s in shapes for d in s}

    n_local = NUM_HEADS // tp
    h_local = n_local * HEAD_DIM
    numels = {s[-1] for s in shapes if s}              # flattened row sizes
    head_layout = any(len(s) == 2 and s[0] == n_local and s[1] in {HEAD_DIM * t for t in tokens}
                      for s in shapes)
    attn_rows = bool(numels & {t * h_local for t in tokens})
    attn_weight = any(s in ((h_local, DIM), (DIM, h_local)) for s in shapes)
    dit_rows = bool(numels & ({t * DIM for t in tokens} | {t * h_local for t in tokens}))

    try:
        has_cc = bool(ni.hasCC())
        ws = ni.getCCRankWorldSize() if has_cc else 1
    except Exception:
        has_cc, ws = False, 0  # unknown -> caller assumes collectives (safe)

    text, macs = neff_contents(path)
    kernel = next((k for k in KERNEL_NAMES if k in text), None)
    if kernel:
        role = "attn_kernel"
    elif head_layout or attn_rows or attn_weight:
        role = "attn_path"
    elif has_cc and dit_rows:
        role = "collective"
    elif "custom-call(" in text:
        if macs <= 0:
            return None  # zero-MAC custom-call off the attention path: copy/fill
        role = "nki_kernel"
    elif FFN_DIM // tp in dims:
        role = "ffn"
    elif DIM in dims:
        role = "dit_other"
    else:
        return None

    return {
        "path": path, "role": role, "ws": ws, "macs": macs, "kernel": kernel,
        "size": os.path.getsize(path),
        "n_in": len(sig.get("inputs", {})), "n_out": len(sig.get("outputs", {})),
        "shapes": shapes[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neff_dir", nargs="+")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP_DEGREE", "4")))
    ap.add_argument("--sp", type=int, default=int(os.environ.get("SP_DEGREE", "2")))
    ap.add_argument("--seq", type=int, required=True,
                    help="DiT token count before padding (T_lat * H/16 * W/16)")
    ap.add_argument("--ctx", type=int, default=512, help="text context length (T5 text_len)")
    ap.add_argument("--print-paths", action="store_true")
    args = ap.parse_args()

    neffs = sorted({p for d in args.neff_dir
                    for p in glob.glob(os.path.join(d, "**/*.neff"), recursive=True)})
    # The cache holds the same graph once per rank; probe each content once.
    uniq = {}
    for p in neffs:
        uniq.setdefault(md5(p), p)

    tokens = attn_tokens(args.seq, args.sp, args.tp, args.ctx)
    cands, errors = [], []
    for p in uniq.values():
        r = probe(p, args.tp, tokens)
        if r is None:
            continue
        (errors if "error" in r else cands).append(r)
    cands.sort(key=lambda c: (ROLE_ORDER[c["role"]], -c["macs"], -c["size"]))
    # The same op is often compiled twice with its inputs in a different order
    # (e.g. fc1 as [x, W, b] and [W, b, x]). Keep one per (role, MACs, shapes).
    seen, kept = set(), []
    for c in cands:
        key = (c["role"], c["macs"], tuple(sorted(c["shapes"])))
        if key not in seen:
            seen.add(key)
            kept.append(c)
    cands = kept

    if args.print_paths:
        for c in cands:
            print(f"{c['role']}\t{c['ws']}\t{c['path']}")
        return 0 if cands else 1

    print(f"scanned {len(neffs)} .neff ({len(uniq)} unique by content), "
          f"TP={args.tp} SP={args.sp} seq={args.seq} ctx={args.ctx} "
          f"attention token counts={sorted(tokens)}")
    for e in errors:
        print(f"  unreadable: {os.path.basename(e['path'])} -- {e['error']}")
    if not cands:
        print(f"NO DiT NEFF FOUND (no NKI attention kernel, no dim of {FFN_DIM // args.tp} or {DIM})")
        return 1
    n_attn = sum(c["role"] in ATTN_ROLES for c in cands)
    print(f"DiT NEFFs: {len(cands)}  (attention chain: {n_attn})")
    for c in cands:
        print(f"  {c['role']:<11} ws={c['ws']:<2} {c['size'] / 1e6:7.1f} MB  "
              f"GMAC={c['macs'] / 1e9:8.1f}  io={c['n_in']}/{c['n_out']}  "
              f"{os.path.basename(c['path'])}  {c['kernel'] or c['shapes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
