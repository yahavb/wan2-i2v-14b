#!/usr/bin/env python3
"""Where the device time of the LAST run goes, from a full-run trace: busy, idle, idle gaps and
the top kernels/graphs, per phase.

Input (one of):
  --neuron-parquet FILE  SystemProfileEvents.parquet exported from an AWS Neuron runtime system
                         trace (neuron_trace_to_parquet.sh; needs duckdb). Uses nc_exec_running;
                         --nc N keeps one NeuronCore when the trace holds several.
  --csv FILE             any trace as CSV with the header start_ns,end_ns,name (one device).
Window, the last run:
  default                everything after the last idle gap longer than --run-gap-s (0.5).
  --anchor NAME --per-run K
                         the run whose K executions of NAME are the last K in the trace. It
                         starts at the largest idle gap within --lookback-s (2) before the first
                         of them and ends with the trace. NAME may be "most-frequent" (the main
                         loop's body, e.g. a transformer block).
Phases (optional):
  --split-after NAME     phase 1 runs to the last execution of NAME in the window; phase 2 starts
                         at the first later execution whose name phase 1 never ran ("most-frequent"
                         allowed). --phase-names a,b labels them.
Prints for each phase: wall, busy (sum of executions), idle, gap percentiles, every idle gap over
--gap-ms (1.0) with its offset and neighbours, and the --top (15; 0 = all) names by device time.
--json-out FILE writes the same numbers for other scripts.
"""
import argparse
import collections
import csv
import json
import sys


def load_neuron(path, nc):
    import duckdb
    where = "name = 'nc_exec_running'" + (f" AND nc_idx = {int(nc)}" if nc is not None else "")
    rows = duckdb.connect().execute(
        f"SELECT start_ts, end_ts, extra_attributes_json FROM read_parquet('{path}') "
        f"WHERE {where} ORDER BY start_ts").fetchall()
    return [(int(a), int(b), json.loads(j).get("model_name", "?") if j else "?") for a, b, j in rows]


def load_csv(path):
    with open(path, newline="") as fh:
        rows = [(int(float(r["start_ns"])), int(float(r["end_ns"])), r["name"]) for r in csv.DictReader(fh)]
    return sorted(rows)


def resolve(name, ex):
    return collections.Counter(x[2] for x in ex).most_common(1)[0][0] if name == "most-frequent" else name


def last_run(ex, a):
    if a.anchor:
        anchor = resolve(a.anchor, ex)
        idx = [i for i, x in enumerate(ex) if x[2] == anchor]
        if len(idx) < a.per_run:
            sys.exit(f"anchor {anchor} ran {len(idx)} times, fewer than --per-run {a.per_run}")
        first = idx[-a.per_run]
        lo = first
        while lo > 0 and ex[first][0] - ex[lo - 1][1] < a.lookback_s * 1e9:
            lo -= 1
        gaps = [(ex[i][0] - ex[i - 1][1], i) for i in range(lo + 1, first + 1)]
        start = max(gaps)[1] if gaps else first
        print(f"anchor {anchor}: {len(idx)} executions in the trace, the last {a.per_run} define the run")
        return start
    start = 0
    for i in range(1, len(ex)):
        if ex[i][0] - ex[i - 1][1] > a.run_gap_s * 1e9:
            start = i
    return start


def split(win, name):
    target = resolve(name, win)
    last = max((i for i, x in enumerate(win) if x[2] == target), default=None)
    if last is None:
        sys.exit(f"--split-after {target}: not in the window")
    seen = {x[2] for x in win[:last + 1]}
    b = next((i for i in range(last + 1, len(win)) if win[i][2] not in seen), len(win))
    return [win[:b], win[b:]] if b < len(win) else [win]


def phase_stats(seg, gap_ms, top):
    t0 = seg[0][0]
    wall = (max(x[1] for x in seg) - t0) / 1e9
    busy = sum(x[1] - x[0] for x in seg) / 1e9
    gaps = [max(0, y[0] - x[1]) for x, y in zip(seg, seg[1:])]
    srt = sorted(gaps)
    q = lambda f: srt[int(f * (len(srt) - 1))] / 1e3 if srt else 0.0
    big = [i for i, g in enumerate(gaps) if g > gap_ms * 1e6]
    by = collections.defaultdict(lambda: [0, 0])
    for x in seg:
        by[x[2]][0] += x[1] - x[0]
        by[x[2]][1] += 1
    order = sorted(by.items(), key=lambda kv: -kv[1][0])
    return {
        "wall_s": wall, "busy_s": busy, "idle_s": wall - busy, "executions": len(seg),
        "distinct_names": len(by),
        "gap_us": {"p50": q(.5), "p90": q(.9), "p99": q(.99)},
        "big_gaps": [{"offset_ms": (seg[i][1] - t0) / 1e6, "gap_ms": gaps[i] / 1e6, "after_exec": i,
                      "after": seg[i][2], "before": seg[i + 1][2]} for i in big],
        "small_gaps_s": (sum(gaps) - sum(gaps[i] for i in big)) / 1e9,
        "top": [{"name": nm, "total_ms": t / 1e6, "n": c, "avg_us": t / c / 1e3}
                for nm, (t, c) in (order if top == 0 else order[:top])],
        "rest_ms": sum(v[0] for _, v in order[top:]) / 1e6 if top else 0.0,
        "rest_names": max(0, len(order) - top) if top else 0,
    }


def show(name, s, gap_ms):
    print(f"\n== {name}: wall {s['wall_s']:.3f} s, device busy {s['busy_s']:.3f} s "
          f"({100 * s['busy_s'] / s['wall_s']:.1f}%), idle {s['idle_s']:.3f} s, {s['executions']} executions, "
          f"{s['distinct_names']} distinct")
    g = s["gap_us"]
    big_sum = sum(x["gap_ms"] for x in s["big_gaps"]) / 1e3
    print(f"   gaps us p50 {g['p50']:.1f} p90 {g['p90']:.1f} p99 {g['p99']:.1f}; {len(s['big_gaps'])} gaps "
          f"> {gap_ms:g} ms sum {big_sum:.3f} s; smaller gaps sum {s['small_gaps_s']:.3f} s")
    for t in s["top"]:
        print(f"  {t['total_ms']:9.1f} ms {100 * t['total_ms'] / 1e3 / s['wall_s']:5.1f}% of wall  "
              f"n={t['n']:5d}  avg {t['avg_us']:8.1f} us  {t['name']}")
    if s["rest_names"]:
        print(f"  {s['rest_ms']:9.1f} ms in the other {s['rest_names']} names")
    if s["big_gaps"]:
        print(f"  idle gaps > {gap_ms:g} ms (offset from the phase start):")
    for x in s["big_gaps"]:
        print(f"    +{x['offset_ms']:9.1f} ms: {x['gap_ms']:7.1f} ms after #{x['after_exec']} ...{x['after'][-16:]} "
              f"before ...{x['before'][-16:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--neuron-parquet")
    src.add_argument("--csv")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--run-gap-s", type=float, default=0.5)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--per-run", type=int, default=None)
    ap.add_argument("--lookback-s", type=float, default=2.0)
    ap.add_argument("--split-after", default=None)
    ap.add_argument("--phase-names", default=None)
    ap.add_argument("--gap-ms", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if a.anchor and not a.per_run:
        ap.error("--anchor needs --per-run")
    ex = load_neuron(a.neuron_parquet, a.nc) if a.neuron_parquet else load_csv(a.csv)
    if not ex:
        sys.exit("no executions in the trace")
    start = last_run(ex, a)
    win = ex[start:]
    print(f"{len(ex)} executions in the trace; the last run has {len(win)}, "
          f"starting {(win[0][0] - ex[0][0]) / 1e9:.3f} s into the trace")
    phases = split(win, a.split_after) if a.split_after else [win]
    names = a.phase_names.split(",") if a.phase_names else [f"phase {i + 1}" for i in range(len(phases))]
    if len(phases) > 1:
        print(f"phase handoff gap {(phases[1][0][0] - phases[0][-1][1]) / 1e6:.1f} ms")
    out = []
    for nm, seg in zip(names, phases):
        s = phase_stats(seg, a.gap_ms, a.top)
        show(nm, s, a.gap_ms)
        out.append(dict(name=nm, **s))
    if a.json_out:
        json.dump({"phases": out}, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
