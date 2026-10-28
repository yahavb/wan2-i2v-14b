#!/usr/bin/env bash
# AWS Neuron runtime-inspector SYSTEM trace directory -> parquet tables for trace_report.py.
# Needs neuron-explorer (Neuron SDK, Linux). No device is used.
#   neuron_trace_to_parquet.sh <trace_dir>/<instance>_pid_<pid> <out_dir>
#   -> <out_dir>/system_profile/SystemProfileEvents.parquet
# The CLI writes its tables and then keeps running a server, so it is stopped once the output
# has stopped growing for 30 s.
set -u
SRC="${1:?usage: neuron_trace_to_parquet.sh <trace_dir> <out_dir>}"
OUT="${2:?usage: neuron_trace_to_parquet.sh <trace_dir> <out_dir>}"
PROF="$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)"
rm -rf "$OUT"; mkdir -p "$OUT"
"$PROF" view -d "$SRC" --ignore-device-profile --output-format parquet --output-file "$OUT" \
  > "$OUT.log" 2>&1 &
pid=$!
prev=-1; stable=0
for _ in $(seq 1 900); do
  sleep 2
  kill -0 "$pid" 2>/dev/null || { echo "converter exited"; break; }
  cur=$(du -sk "$OUT" | cut -f1)
  f=$(find "$OUT" -name SystemProfileEvents.parquet | head -1)
  if [ -n "$f" ] && [ "$cur" = "$prev" ] && [ "$cur" -gt 2000 ]; then stable=$((stable + 1)); else stable=0; fi
  prev=$cur
  [ "$stable" -ge 15 ] && { echo "output stable at ${cur} KB"; kill "$pid"; break; }
done
find "$OUT" -name "*.parquet" -size +1k
