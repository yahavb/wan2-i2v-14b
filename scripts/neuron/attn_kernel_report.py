#!/usr/bin/env python3
"""One line per captured attention-kernel NEFF: the numbers to compare across ATTN_IMPL.

Reads <dit_pairs>/*attn_kernel*/summary.json (written by the job's capture_dit_pairs).
The *_percent counters in summary.json are fractions (0.095 == 9.5%).

Usage: attn_kernel_report.py <dit_pairs_dir> <ATTN_IMPL>
"""
import glob
import json
import os
import sys


def main():
    root, impl = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "?")
    dirs = sorted(glob.glob(os.path.join(root, "*_attn_kernel_*")) +
                  glob.glob(os.path.join(root, "*_nki_kernel_*")))
    if not dirs:
        print(f"ATTN_KERNEL [{impl}] none captured under {root}")
        return 0
    for d in dirs:
        try:
            m = list(json.load(open(os.path.join(d, "summary.json"))).values())[0]
        except Exception as e:  # noqa: BLE001
            print(f"ATTN_KERNEL [{impl}] {os.path.basename(d)}: no summary ({type(e).__name__})")
            continue

        def pct(key):
            v = m.get(key)
            return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"

        t = m.get("total_time") or 0
        print(f"ATTN_KERNEL [{impl}] {os.path.basename(d)[-32:-24]}  time={t * 1e3:.2f}ms  "
              f"MFU={pct('mfu_estimated_percent')}  tensor={pct('tensor_engine_active_time_percent')}  "
              f"vector={pct('vector_engine_active_time_percent')}  "
              f"sw_dma_pkts={pct('software_dynamic_dma_packet_percent')}  "
              f"psum_wr={(m.get('psum_write_bytes') or 0) / 1e9:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
