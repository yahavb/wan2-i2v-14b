#!/bin/bash
# query_ntff.sh — set up the preserved heavy NEFF+NTFF pairs for instruction-level
# SQL querying, ON LOCAL DISK ONLY. Run INSIDE the query pod (kubectl exec).
#
# Storage rule: /var/mdl is the S3-backed PVC — ONLY copy tar.gz to/from it.
# Everything else (untar, ingest, DuckDB, parquet, --data-path) goes to /tmp.
#
# The neuron-explorer view server may HANG on big DMA NEFFs on this image
# (see neuron-3run-benchmark skill). So this script does NOT auto-run the heavy
# ones — it lists the pairs smallest-first and gives you the exact commands to
# run one at a time, so you can kill a hang.
set -uo pipefail

WORK=/tmp/ntff_query
TGZ=/var/mdl/wan2-i2v-14b_results.tar.gz

echo "=== 1. copy tarball off the S3 PVC to local /tmp (only allowed PVC op) ==="
rm -rf "$WORK"; mkdir -p "$WORK"
cp "$TGZ" "$WORK/results.tar.gz" || { echo "MISSING $TGZ"; exit 1; }
tar -xzf "$WORK/results.tar.gz" -C "$WORK"
PAIRS="$WORK/results/ntff_pairs"
echo "pairs dir: $PAIRS"

echo "=== 2. preserved pairs, SMALLEST NEFF first (test the small one first) ==="
ls -lS "$PAIRS"/*.neff 2>/dev/null | tac | awk '{print $5, $9}'

echo ""
echo "=== 3. to query ONE pair (start with the smallest hash above): ==="
cat <<'CMDS'
  H=<hash>                       # from the list above
  D=/tmp/ntff_query/results/ntff_pairs
  D2=/tmp/neuron-profile          # local data-path, NOT /var/mdl
  # start server (background); kill with `kill %1` if it hangs
  neuron-explorer view -n "$D/$H.neff" -s "$D/$H.ntff" \
    --data-path "$D2" --display-name q --disable-ui &
  sleep 15
  # does this profile even have DMA packet rows?
  curl -s -X POST http://localhost:3002/api/v1/db/q/_search \
    -H 'Content-Type: application/json' \
    -d '{"type":"databaseExplorerQuery","tableName":"DmaPacket","query":"SELECT COUNT(*) cnt FROM DmaPacket"}'
  # top DMA by source location — NAMES the op generating the DMA
  curl -s -X POST http://localhost:3002/api/v1/db/q/_search \
    -H 'Content-Type: application/json' \
    -d '{"type":"databaseExplorerQuery","tableName":"DmaPacketAggregated","query":"SELECT bir_debug_info_source_location, COUNT(*) pkts, SUM(size_bytes) bytes FROM DmaPacketAggregated GROUP BY 1 ORDER BY bytes DESC LIMIT 20"}'
CMDS
echo ""
echo "Done. Run the commands above one NEFF at a time."
