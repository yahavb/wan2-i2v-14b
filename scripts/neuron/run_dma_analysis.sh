#!/bin/bash
# run_dma_analysis.sh — fully automatic instruction-level DMA analysis of ALL
# preserved heavy NEFF+NTFF pairs. No exec, no interaction. Prints to stdout.
#
# Storage rule: /var/mdl is the S3 PVC — ONLY copy the tar.gz off it. Everything
# else (untar, parquet ingest, queries) is LOCAL on /tmp.
#
# neuron-explorer view --output-format parquet WRITES the parquet then idles on
# a UI server (no exit). So we run it backgrounded, poll for the parquet files,
# then kill it — that is the "completion" signal, not process exit.
set -uo pipefail

WORK=/tmp/ntff_query
TGZ=/var/mdl/wan2-i2v-14b_results.tar.gz
EXPLORER=$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== deps: duckdb (local query engine) ==="
python3 -c "import duckdb" 2>/dev/null || pip install duckdb >/dev/null 2>&1 || { echo "duckdb install FAILED"; exit 1; }

echo "=== copy results tar.gz off the S3 PVC to local /tmp ==="
rm -rf "$WORK"; mkdir -p "$WORK"
cp "$TGZ" "$WORK/results.tar.gz" || { echo "MISSING $TGZ"; exit 1; }
tar -xzf "$WORK/results.tar.gz" -C "$WORK"
PAIRS="$WORK/results/ntff_pairs"
mapfile -t NEFFS < <(ls -S "$PAIRS"/*.neff 2>/dev/null)
echo "found ${#NEFFS[@]} NEFF+NTFF pairs"

ingest() {  # $1=neff $2=ntff $3=outdir — write parquet, then kill the idle server
  local neff="$1" ntff="$2" out="$3"
  rm -rf "$out"; mkdir -p "$out"
  timeout 1200 "$EXPLORER" view -n "$neff" -s "$ntff" \
    --output-format parquet --output-file "$out" \
    --ignore-instruction-hierarchy --ignore-event-trace >/dev/null 2>&1 &
  local pid=$!
  # neuron-explorer creates Summary.parquet early but FILLS IT LAST, then idles
  # on a UI server (never exits). So we wait until the parquet dir size has been
  # STABLE for ~15s (writing finished) — not mere file existence — then kill it.
  local prev=-1 cur stable=0
  for _ in $(seq 1 240); do  # up to 20 min
    kill -0 "$pid" 2>/dev/null || break
    cur=$(du -sb "$out" 2>/dev/null | cut -f1)
    if [ -f "$out/Summary.parquet" ] && [ "$cur" = "$prev" ]; then
      stable=$((stable+1)); [ "$stable" -ge 3 ] && break   # ~15s unchanged
    else
      stable=0
    fi
    prev="$cur"; sleep 5
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  # valid only if Summary.parquet is a real file (parquet footer >= a few hundred bytes)
  [ -f "$out/Summary.parquet" ] && [ "$(stat -c%s "$out/Summary.parquet" 2>/dev/null || echo 0)" -gt 1000 ]
}

for neff in "${NEFFS[@]}"; do
  h=$(basename "$neff" .neff)
  ntff="$PAIRS/$h.ntff"
  out="$WORK/pq_$h"
  echo ""
  echo "########## ingesting $h ($(du -h "$neff" | cut -f1)) ##########"
  if [ ! -f "$ntff" ]; then echo "  no NTFF, skip"; continue; fi
  if ingest "$neff" "$ntff" "$out"; then
    python3 "$HERE/pq_dma_report.py" "$out" "$h"
  else
    echo "  ingest produced no parquet (timeout/too big) — skip"
  fi
done

echo ""
echo "=== DMA analysis complete for all NEFFs ==="
