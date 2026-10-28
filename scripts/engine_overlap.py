#!/usr/bin/env python3
"""Engine utilization and overlap of one compiled graph, from its instruction table, with the
source lines on the serial path.

A NeuronCore has several engines (tensor, vector, scalar/activation, gpsimd) plus DMA queues
that run concurrently. A graph is fast when the engines are busy AT THE SAME TIME: the time
only one engine is busy is the serial path, and it is the target for re-partitioning (tile the
work so the tensor engine runs the next tile while the vector/scalar engines finish this one,
move elementwise work to an idle engine, double-buffer the DMA).

Usage: engine_overlap.py <parquet-dir | Instruction.parquet> [--top 12] [--json-out FILE]
Input: the Instruction table written by `neuron-explorer view --output-format parquet`
(needs duckdb). Start/duration/engine/source columns are detected by name; the script prints
the columns it used and the table's columns when it cannot find one.
Prints for the window from the first instruction start to the last end:
  - per engine: busy (union of its instructions), share of the window, time it is the ONLY
    busy engine (serial), and instruction time summed
  - all-engine idle, time with exactly 1 / 2 / 3+ engines busy
  - pairwise overlap (time both are busy) for the busiest engines
  - top source lines per engine by instruction time, and top source lines by SERIAL time
    (instruction time during which no other engine was busy)
Instruction time summed can exceed busy time when an engine's instructions overlap in the
trace (queued or pipelined); busy is the union.
"""
import argparse
import collections
import json
import os
import sys

START = ("timestamp", "start_ts", "start_time", "start_ns", "timestamp_ns", "ts", "start")
DUR = ("duration_ns", "duration", "dur_ns", "dur")
END = ("end_ts", "end_time", "end_ns", "end")
ENGINE = ("engine", "engine_name", "queue", "engine_type")
SRC = ("nki_source_location", "bir_debug_info_source_location", "source_location")
OPC = ("opcode", "op_name", "instruction")
HLO = ("hlo_name",)


def pick(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def fmt(ns):
    return f"{ns / 1e6:8.3f} ms"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    path = os.path.join(a.path, "Instruction.parquet") if os.path.isdir(a.path) else a.path
    if not os.path.exists(path):
        print(f"no Instruction.parquet at {path}")
        return 2
    import duckdb
    con = duckdb.connect()
    cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    c_start, c_dur, c_end = pick(cols, START), pick(cols, DUR), pick(cols, END)
    c_eng = pick(cols, ENGINE)
    srcs = [c for c in (pick(cols, (s,)) for s in SRC) if c]
    c_op, c_hlo = pick(cols, OPC), pick(cols, HLO)
    if not c_start or not c_eng or not (c_dur or c_end):
        print(f"missing a start, engine or duration/end column. columns: {cols}")
        return 2
    end_expr = f"{c_start} + {c_dur}" if c_dur else c_end
    src_expr = ("COALESCE(" + ", ".join(f"NULLIF(CAST({c} AS VARCHAR), '')" for c in srcs) + ", '')") if srcs else "''"
    label = " | ".join(f"CAST({c} AS VARCHAR)" for c in (c_op, c_hlo) if c) or "''"
    print(f"columns used: start={c_start} end={'start+' + c_dur if c_dur else c_end} engine={c_eng} "
          f"source={srcs or 'none'} label={c_op}/{c_hlo}")
    rows = con.execute(
        f"SELECT CAST({c_eng} AS VARCHAR), CAST({c_start} AS BIGINT), CAST({end_expr} AS BIGINT), "
        f"{src_expr}, {label} FROM read_parquet('{path}') WHERE {c_start} IS NOT NULL ORDER BY 2").fetchall()
    rows = [r for r in rows if r[2] is not None and r[2] > r[1]]
    if not rows:
        print("no instructions with a positive duration")
        return 2
    t0, t1 = min(r[1] for r in rows), max(r[2] for r in rows)
    window = t1 - t0
    engines = sorted({r[0] for r in rows})
    idx = {e: i for i, e in enumerate(engines)}

    # sweep line over instruction starts/ends: per-engine active counts
    ev = []
    for e, s, t, _, _ in rows:
        ev.append((s, 1, idx[e]))
        ev.append((t, -1, idx[e]))
    ev.sort(key=lambda x: (x[0], x[1]))
    cnt = [0] * len(engines)
    busy = [0] * len(engines)
    alone = [0] * len(engines)
    pair = collections.Counter()
    nbusy = collections.Counter()
    segs = []  # (start, end, active engine set) for serial attribution
    prev = t0
    for t, d, i in ev:
        if t > prev:
            act = [k for k in range(len(engines)) if cnt[k] > 0]
            dt = t - prev
            nbusy[min(len(act), 3)] += dt
            for k in act:
                busy[k] += dt
            if len(act) == 1:
                alone[act[0]] += dt
                segs.append((prev, t, act[0]))
            for x in range(len(act)):
                for y in range(x + 1, len(act)):
                    pair[(act[x], act[y])] += dt
            prev = t
        cnt[i] += d

    summed = collections.Counter()
    by_src = collections.defaultdict(collections.Counter)
    for e, s, t, src, _ in rows:
        summed[e] += t - s
        by_src[e][src or "(no source)"] += t - s

    # serial time per (engine, source): overlap of each instruction with the single-engine segments
    serial_src = collections.Counter()
    seg_by_eng = collections.defaultdict(list)
    for s, t, k in segs:
        seg_by_eng[engines[k]].append((s, t))
    import bisect
    for e, lst in seg_by_eng.items():
        starts = [s for s, _ in lst]
        for ee, s, t, src, _ in rows:
            if ee != e:
                continue
            j = max(bisect.bisect_right(starts, s) - 1, 0)
            while j < len(lst) and lst[j][0] < t:
                lo, hi = max(s, lst[j][0]), min(t, lst[j][1])
                if hi > lo:
                    serial_src[(e, src or "(no source)")] += hi - lo
                j += 1

    order = sorted(range(len(engines)), key=lambda k: -busy[k])
    print(f"\nwindow {fmt(window)}  ({len(rows)} instructions, {len(engines)} engines/queues)")
    print(f"\n{'engine':24} {'busy':>11} {'share':>6} {'serial (alone)':>15} {'instr time sum':>15}")
    for k in order:
        e = engines[k]
        print(f"{e[:24]:24} {fmt(busy[k])} {100 * busy[k] / window:5.1f}% {fmt(alone[k]):>15} {fmt(summed[e]):>15}")
    idle = nbusy[0]
    print(f"\nall idle      {fmt(idle)}  {100 * idle / window:5.1f}%")
    for n, name in ((1, "1 engine "), (2, "2 engines"), (3, "3+ engines")):
        print(f"{name}     {fmt(nbusy[n])}  {100 * nbusy[n] / window:5.1f}%")
    print("\npairwise overlap (time both busy), busiest engines:")
    top_e = order[:6]
    for x in range(len(top_e)):
        for y in range(x + 1, len(top_e)):
            p, q = sorted((top_e[x], top_e[y]))
            v = pair[(p, q)]
            if v:
                print(f"  {engines[p][:20]:20} + {engines[q][:20]:20} {fmt(v)}  "
                      f"({100 * v / max(min(busy[p], busy[q]), 1):4.0f}% of the smaller)")
    print(f"\ntop source lines by SERIAL time (no other engine busy) -- re-partition / overlap targets:")
    for (e, src), v in serial_src.most_common(a.top):
        print(f"  {fmt(v)}  {e[:14]:14} {src[-90:]}")
    for k in order[:4]:
        e = engines[k]
        tot = summed[e] or 1
        nosrc = by_src[e].get("(no source)", 0)
        print(f"\n{e}: top source lines by instruction time ({100 * nosrc / tot:.0f}% has no source location)")
        for src, v in by_src[e].most_common(a.top):
            print(f"  {fmt(v)}  {100 * v / tot:5.1f}%  {src[-90:]}")
    if a.json_out:
        json.dump({"window_ns": window, "idle_ns": idle,
                   "engines": {engines[k]: {"busy_ns": busy[k], "alone_ns": alone[k], "sum_ns": summed[engines[k]]}
                               for k in order},
                   "n_busy_ns": {str(n): nbusy[n] for n in (1, 2, 3)},
                   "serial_src": [[e, s, v] for (e, s), v in serial_src.most_common(50)]},
                  open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
