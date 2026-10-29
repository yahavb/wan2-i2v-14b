#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimization memory for accelerator perf work: a JSONL ledger of measured configs.

The point is memoization, not analysis. When each experiment costs a 10-40 minute compile,
the expensive mistake is re-running something already known -- especially a known negative.

    add       record a measurement (or a refuted hypothesis)
    known     has this config been measured? exits 0 if yes, 1 if no  -> use in scripts
    refuted   list refuted hypotheses + the measurement that killed each
    best      best config that PASSES the gate
    report    full table, ordered by the primary metric
    axes      which knobs have actually been varied, and over what values

Config identity is the canonical JSON of the config dict, so key order and whitespace
cannot create duplicate rows for the same experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Optional

SCHEMA_VERSION = 1


def _canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def config_id(cfg: dict) -> str:
    return hashlib.sha256(_canon(cfg).encode()).hexdigest()[:16]


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[ledger] skipping malformed line {i}: {exc}", file=sys.stderr)
    return rows


def append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
def cmd_add(a) -> int:
    cfg = json.loads(a.config) if a.config else {}
    row: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "kind": a.kind,
        "label": a.label,
        "config": cfg,
        "config_id": config_id(cfg) if cfg else None,
        "verdict": a.verdict,
    }
    for k, v in (("latency_ms", a.latency_ms), ("psnr_db", a.psnr_db),
                 ("cos", a.cos), ("latency_1core_ms", a.latency_1core_ms)):
        if v is not None:
            row[k] = v
    if a.hypothesis:
        row["hypothesis"] = a.hypothesis
    if a.killed_by:
        row["killed_by"] = a.killed_by
    if a.note:
        row["note"] = a.note
    if a.profile:
        row["profile"] = json.loads(a.profile)
    if a.env:
        row["env"] = a.env

    if cfg and not a.allow_dup:
        for r in load(a.ledger):
            if r.get("config_id") == row["config_id"]:
                print(f"[ledger] REFUSING duplicate: {row['config_id']} already measured "
                      f"as {r.get('label')!r} "
                      f"(latency {r.get('latency_ms')}, psnr {r.get('psnr_db')}). "
                      f"Pass --allow-dup to override.")
                return 1
    append(a.ledger, row)
    print(f"[ledger] recorded {a.kind}/{a.verdict} {a.label!r}"
          + (f" id={row['config_id']}" if row["config_id"] else ""))
    return 0


def cmd_known(a) -> int:
    cfg = json.loads(a.config)
    cid = config_id(cfg)
    for r in load(a.ledger):
        if r.get("config_id") == cid:
            print(json.dumps(r, indent=2, sort_keys=True))
            return 0
    print(f"[ledger] not measured: {cid}")
    return 1


def cmd_refuted(a) -> int:
    rows = [r for r in load(a.ledger) if str(r.get("verdict", "")).upper() == "REFUTED"]
    if not rows:
        print("[ledger] no refutations recorded")
        return 0
    print(f"{len(rows)} refuted hypothes{'is' if len(rows) == 1 else 'es'} "
          f"-- do NOT re-test these without new information:\n")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r.get('hypothesis') or r.get('label')}")
        if r.get("killed_by"):
            for line in str(r["killed_by"]).splitlines():
                print(f"    killed by: {line}" if line is str(r["killed_by"]).splitlines()[0]
                      else f"               {line}")
        if r.get("note"):
            print(f"    note:      {r['note']}")
        print()
    return 0


def _passes(r: dict, gate_psnr: Optional[float]) -> bool:
    if gate_psnr is None:
        return True
    v = r.get("psnr_db")
    return v is not None and v >= gate_psnr


def cmd_best(a) -> int:
    rows = [r for r in load(a.ledger)
            if r.get("kind") == "config" and r.get("latency_ms") is not None]
    passing = [r for r in rows if _passes(r, a.gate_psnr)]
    failing = [r for r in rows if not _passes(r, a.gate_psnr)]
    if not passing:
        print(f"[ledger] no config passes the gate (psnr >= {a.gate_psnr})")
        return 1
    best = min(passing, key=lambda r: r["latency_ms"])
    print(f"BEST passing config ({len(passing)} pass, {len(failing)} fail the gate):\n")
    print(json.dumps(best, indent=2, sort_keys=True))
    if failing:
        print("\nfailed the gate (kept so nobody re-runs them):")
        for r in sorted(failing, key=lambda r: r.get("latency_ms") or 0):
            print(f"  {r.get('label'):<44} {r.get('latency_ms')} ms  "
                  f"psnr {r.get('psnr_db')}")
    return 0


def cmd_report(a) -> int:
    rows = load(a.ledger)
    cfgs = [r for r in rows if r.get("kind") == "config"]
    print(f"{'label':<46} {'8core ms':>9} {'1core ms':>9} {'PSNR':>8} {'cos':>10} "
          f"{'gate':>6}")
    print("-" * 96)
    for r in sorted(cfgs, key=lambda r: r.get("latency_ms") or 1e9):
        # Distinguish "ran and missed the gate" from "never produced a number".
        if r.get("psnr_db") is None:
            gate = str(r.get("verdict", "?"))[:7]
        elif a.gate_psnr is None:
            gate = "-"
        else:
            gate = "PASS" if _passes(r, a.gate_psnr) else "FAIL"
        print(f"{str(r.get('label'))[:46]:<46} "
              f"{r.get('latency_ms', '-')!s:>9} {r.get('latency_1core_ms', '-')!s:>9} "
              f"{r.get('psnr_db', '-')!s:>8} {r.get('cos', '-')!s:>10} {gate:>6}")
    nref = sum(1 for r in rows if str(r.get("verdict", "")).upper() == "REFUTED")
    print(f"\n{len(cfgs)} config(s) measured, {nref} refuted hypothes"
          f"{'is' if nref == 1 else 'es'}. Run `refuted` before proposing anything.")
    return 0


def cmd_axes(a) -> int:
    vals: dict[str, set] = {}
    for r in load(a.ledger):
        for k, v in (r.get("config") or {}).items():
            vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True))
    if not vals:
        print("[ledger] no configs recorded")
        return 0
    print("knobs varied so far (an axis with ONE value is effectively untested):\n")
    for k in sorted(vals):
        seen = sorted(vals[k])
        flag = "   <-- only one value tried" if len(seen) == 1 else ""
        print(f"  {k:<32} {len(seen):>2} value(s): "
              f"{', '.join(s.strip(chr(34)) for s in seen)}{flag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="perf_ledger.jsonl")
    # --ledger is accepted BOTH before and after the subcommand, because shell drivers
    # naturally write `perf_ledger.py add --ledger X ...`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", dest="ledger_sub", default=None,
                        help="path to the JSONL ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", parents=[common],
                       help="record a measurement or refutation")
    p.add_argument("--kind", choices=("config", "hypothesis"), default="config")
    p.add_argument("--label", required=True)
    p.add_argument("--config", default=None, help="JSON dict of the knobs")
    p.add_argument("--verdict", default="MEASURED",
                   help="MEASURED | CONFIRMED | REFUTED | FAILED | CRASHED")
    p.add_argument("--latency-ms", type=float, default=None)
    p.add_argument("--latency-1core-ms", type=float, default=None)
    p.add_argument("--psnr-db", type=float, default=None)
    p.add_argument("--cos", type=float, default=None)
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--killed-by", default=None, help="the measurement that settled it")
    p.add_argument("--note", default=None)
    p.add_argument("--profile", default=None, help="JSON dict of profile fingerprint")
    p.add_argument("--env", default=None, help="hardware / sdk string")
    p.add_argument("--allow-dup", action="store_true")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("known", parents=[common],
                       help="exit 0 if this config was already measured")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=cmd_known)

    p = sub.add_parser("refuted", parents=[common], help="list refuted hypotheses")
    p.set_defaults(fn=cmd_refuted)

    p = sub.add_parser("best", parents=[common], help="best config passing the gate")
    p.add_argument("--gate-psnr", type=float, default=None)
    p.set_defaults(fn=cmd_best)

    p = sub.add_parser("report", parents=[common], help="full table")
    p.add_argument("--gate-psnr", type=float, default=None)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("axes", parents=[common], help="which knobs have been varied")
    p.set_defaults(fn=cmd_axes)

    a = ap.parse_args()
    # --ledger is accepted on either side of the subcommand; the subcommand-level value
    # wins when supplied, otherwise the top-level one (which carries the default).
    if getattr(a, "ledger_sub", None):
        a.ledger = a.ledger_sub
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
