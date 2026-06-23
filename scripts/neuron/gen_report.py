"""Generate the canonical Neuron 3-run benchmark report from a run's kubectl log.

The report has a FIXED two-layer structure that mirrors the 3-run flow:
  - high-level layer  = step [4] summary-json aggregation (engine shares, latency)
  - per-NEFF layer    = step [5] parquet+duckdb (each heavy NEFF's named HLO op)

Both layers MUST be present; if step [5] is missing the report says so rather
than inventing op names. Every number is pulled from the log — nothing invented.

Usage: python3 gen_report.py <kubectl-log-file>
"""
import sys
import re

log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
L = log.splitlines()


def grep(pat):
    rx = re.compile(pat)
    return [ln for ln in L if rx.search(ln)]


def first(pat, group=1, default="?"):
    rx = re.compile(pat)
    for ln in L:
        m = rx.search(ln)
        if m:
            return m.group(group)
    return default


print("# Neuron 3-Run Benchmark Report\n")

# ── provenance (proves which code ran) ──
print("## 1. Provenance (what code actually ran)")
for key in ("branch :", "commit :", "subject:"):
    ln = next((x for x in L if key in x), None)
    if ln:
        print("  " + ln.split("│")[-1].strip() if "│" in ln else "  " + ln.strip())
nki = first(r"PROVENANCE NKI:.*(self=\S+ cross=\S+ SEQLEN_MULT=\S+)", 1, "(not found)")
print("  NKI:", nki)

# ── latency (step 2 clean = the latency; step 3 profiled = overhead) ──
print("\n## 2. Latency")
clean = first(r"clean .*REAL.*?:\s*([0-9.]+ s/run)")
prof = first(r"profiled .*profiler on.*?:\s*([0-9.]+ s/run)")
print("  clean median (THE latency):", clean)
print("  profiled (overhead):", prof)
for ln in grep(r"Run [0-9]/3 DONE"):
    print("   ", ln.split("]:")[-1].strip())

# ── engine shares (step 4, summary-json layer) ──
print("\n## 3. Engine shares  [step 4: summary-json — general metrics]")
in_blk = False
for ln in L:
    if "MODEL-LEVEL engine time" in ln:
        in_blk = True
    elif "weighted total_time" in ln:
        print("  " + ln.strip())
        break
    elif in_blk and "=" in ln:
        print("  " + ln.strip())

# ── per-NEFF named ops (step 5, parquet+duckdb layer) ──
print("\n## 4. Per-NEFF operation breakdown  [step 5: parquet+duckdb — names the op]")
if "INSTRUCTION-LEVEL DMA ANALYSIS" not in log:
    print("  !! step [5] MISSING from log — cannot name ops. Re-run with step 5.")
else:
    cur = None
    for ln in L:
        m = re.search(r"^NEFF:\s*(\S+)", ln)
        if m:
            cur = m.group(1)[:12]
        mt = re.search(r"total_time=\s*([0-9.]+ms).*?matmul_count=(\d+)", ln)
        if mt and cur:
            print(f"\n  {cur}  {mt.group(1)}  matmul={mt.group(2)}")
        md = re.search(r"dma    :.*", ln)
        if md:
            print("    " + ln.strip())
        mh = re.search(r"hlo=(%\S+ = \w+)", ln)
        if mh:
            print("    op:", mh.group(1))

print("\n## 5. Verdict")
print("  - Latency is the clean median above; compare engine SHARES, not absolute us.")
print("  - If DMA-family dominates and heavy NEFFs are pad/concatenate with matmul=0,")
print("    the model is DMA-bound on layout ops (not compute). Fix in the KERNEL,")
print("    not Python — the compiler re-emits the same HLO regardless of torch-level code.")
