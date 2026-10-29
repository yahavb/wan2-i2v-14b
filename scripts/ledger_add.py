#!/usr/bin/env python3
"""Append ledger rows from a JSON file through perf_ledger.py add, so JSON never goes through
shell quoting (a quote inside a --config or --note string is the usual way a hand-typed
`perf_ledger.py add` breaks).

Usage: ledger_add.py rows.json [--ledger perf_ledger.jsonl] [--env "hardware, SDK versions"]
rows.json holds a list of objects whose keys are perf_ledger.py add options with underscores:
kind, label, config, verdict, latency_ms, latency_1core_ms, psnr_db, cos, hypothesis, killed_by,
note, profile, env, allow_dup. Dict values (config, profile) are passed as JSON. A row without
"env" gets --env.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows")
    ap.add_argument("--ledger", default="perf_ledger.jsonl")
    ap.add_argument("--env", default=None, help="default env string for rows without one")
    a = ap.parse_args()
    rows = json.load(open(a.rows))
    if isinstance(rows, dict):
        rows = [rows]
    status = 0
    for row in rows:
        row = dict(row)
        env = row.pop("env", a.env)
        args = [sys.executable, os.path.join(HERE, "perf_ledger.py"), "add", "--ledger", a.ledger]
        if env:
            args.append(f"--env={env}")
        if row.pop("allow_dup", False):
            args.append("--allow-dup")
        for k, v in row.items():  # --key=value, so a value starting with "-" is not read as an option
            val = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            args.append(f"--{k.replace('_', '-')}={val}")
        r = subprocess.run(args, stdin=subprocess.DEVNULL)
        status = status or r.returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
