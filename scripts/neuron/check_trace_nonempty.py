#!/usr/bin/env python3
"""Assert a captured NTFF actually recorded an execution.

WHY: a structurally valid NTFF can be ENTIRELY EMPTY. Capturing the DiT fullgraph NEFF
with `--profile-nth-exec=5` produced a 566 MB .ntff whose every counter was zero --
total_time=0, cc_op_count=0, dma_active_time=0, hbm_write_bytes=4128, and no
matmul_instruction_count key at all. The replay never reached execution 5, but the file
existed, so the capture loop counted it as OK and the pair was archived and uploaded.

Exit code is always 0 (never fail the run over a diagnostic); the verdict is the printed
line. Usage: check_trace_nonempty.py <summary.json> <hash>
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("    TRACE CHECK: no summary path given")
        return 0
    path = sys.argv[1]
    tag = (sys.argv[2][:8] if len(sys.argv) > 2 else "?")
    try:
        nodes = json.load(open(path))
        m = list(nodes.values())[0]
    except Exception as e:  # noqa: BLE001
        print(f"    TRACE CHECK {tag}: unreadable summary ({type(e).__name__}: {e})")
        return 0

    tt = m.get("total_time") or 0
    mm = m.get("matmul_instruction_count") or 0
    cc = m.get("cc_op_count") or 0
    dma = m.get("dma_active_time") or 0

    if tt == 0 or (mm == 0 and dma == 0):
        print(f"    *** TRACE EMPTY {tag}: total_time={tt} matmul={mm} cc_ops={cc} "
              f"dma_active={dma} -- the ntff EXISTS but nothing executed. "
              f"Do NOT analyze or upload it.")
    else:
        print(f"    trace OK {tag}: total_time={tt * 1e3:.1f}ms matmul={mm:,} "
              f"cc_ops={cc} dma_active={dma * 1e3:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
