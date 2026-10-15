#!/usr/bin/env bash
# capture_attn_pairs.sh -- on a FINISHED stage-job pod, find each ATTN_IMPL stage's
# attention-kernel NEFF, capture its NTFF, and pack NEFF+NTFF+summary+kernel source.
#
# Stage attribution is by NEFF mtime against the job's own log files:
#   stage S compiled its NEFFs during warmup_S, i.e. between the end of the previous
#   stage's clean run (clean_<prev>.txt mtime; provenance.txt for the first stage) and
#   the end of warmup_S (warmup_S.txt mtime). NEFFs are located with find over the
#   local filesystem, not assumed to be in any particular cache dir.
#
# Usage (on the pod):  bash scripts/neuron/capture_attn_pairs.sh
# Output: /tmp/results/attn_pairs_<stage>.tar.gz
set -uo pipefail

R=/tmp/results
OUT=$R/attn_pairs
STAGES=${ATTN_IMPLS:-legacy cte constmax}
SEQ=$(( ((${FRAME_NUM:-21} - 1) / 4 + 1) * (${OUT_H:-368} / 16) * (${OUT_W:-480} / 16) ))
PROF=$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)
WORLD=$(( ${TP_DEGREE:-4} * ${SP_DEGREE:-2} ))
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== all NEFFs written since job start (any dir) =="
find / -xdev -name '*.neff' -newer "$R/provenance.txt" \
     -not -path "$R/*" -not -path '/proc/*' 2>/dev/null > "$OUT/all_neffs.txt"
echo "  total: $(wc -l < "$OUT/all_neffs.txt")"
sed 's|/[^/]*$||' "$OUT/all_neffs.txt" | sort | uniq -c | sort -rn | head -5 | sed 's/^/  /'

prev="$R/provenance.txt"
for S in $STAGES; do
  echo "████ $S ████"
  [ -f "$R/warmup_$S.txt" ] || { echo "  no warmup_$S.txt -- skip"; continue; }
  lo=$(stat -c %Y "$prev"); hi=$(stat -c %Y "$R/warmup_$S.txt")
  D="$OUT/$S"; mkdir -p "$D/neffs"
  while read -r f; do
    m=$(stat -c %Y "$f" 2>/dev/null) || continue
    [ "$m" -ge "$lo" ] && [ "$m" -le "$hi" ] && ln -sf "$f" "$D/neffs/$(echo "$f" | md5sum | cut -c1-8)_$(basename "$f")"
  done < "$OUT/all_neffs.txt"
  echo "  NEFFs compiled in this stage's warmup: $(ls "$D/neffs" | wc -l)  ($(date -d @"$lo" +%T)..$(date -d @"$hi" +%T))"

  python3 "$REPO/scripts/neuron/pick_dit_neffs.py" "$D/neffs" --seq "$SEQ" 2>&1 | tee "$D/identify.txt" | head -30
  # The attention core: attention-shaped kernels, plus every other custom-call with
  # matmul work (the kernel layout differs per ATTN_IMPL; shapes are in identify.txt).
  n=0
  while IFS=$'\t' read -r role ws neff; do
    case "$role" in attn_kernel|nki_kernel) ;; *) continue ;; esac
    n=$((n+1)); h=$(basename "$neff" .neff); P="$D/$(printf '%02d' $n)_${role}_${h}"; mkdir -p "$P"
    cp -L "$neff" "$P/$h.neff" || continue
    [ "$ws" = "0" ] && ws=$WORLD
    extra=(); [ "$ws" -gt 1 ] && extra=(-r "$ws" -i 0)
    ( cd "$P" && timeout 1800 "$PROF" capture -n "$h.neff" -s "$h.ntff" ${extra[@]+"${extra[@]}"} \
        --num-exec=1 --enable-dge-notifs ) > "$P/capture.log" 2>&1 || echo "  capture rc=$?"
    if [ ! -s "$P/$h.ntff" ]; then
      f=$(ls -S "$P/$h"*rank_0*.ntff "$P/$h"*.ntff 2>/dev/null | head -1) || true
      [ -n "$f" ] && [ -s "$f" ] && mv "$f" "$P/$h.ntff" || true
    fi
    rm -f "$P/$h"*rank_*.ntff 2>/dev/null || true
    if [ -s "$P/$h.ntff" ]; then
      timeout 900 "$PROF" view -n "$P/$h.neff" -s "$P/$h.ntff" --output-format summary-json > "$P/summary.json" 2>/dev/null || true
      timeout 900 "$PROF" view -n "$P/$h.neff" -s "$P/$h.ntff" --output-format summary-text > "$P/summary.txt" 2>/dev/null || true
    else
      echo "  !! $h no ntff:"; tail -5 "$P/capture.log"
    fi
  done < <(python3 "$REPO/scripts/neuron/pick_dit_neffs.py" "$D/neffs" --seq "$SEQ" --print-paths 2>/dev/null || true)
  python3 "$REPO/scripts/neuron/attn_kernel_report.py" "$D" "$S" | tee "$D/report.txt"

  # The code that produced this stage's kernel.
  mkdir -p "$D/code"
  cp "$REPO/wan/modules/attention.py" "$REPO/models/sp_attention.py" "$D/code/"
  case "$S" in
    legacy)   cp "$REPO/kernels/self_attention.py" "$REPO/kernels/cross_attention.py" "$D/code/" ;;
    cte)      cp "$(python3 -c 'import nkilib.core.attention.attention_cte as m; print(m.__file__)')" "$D/code/" ;;
    constmax) cp "$REPO"/kernels/nkilib/experimental/attention/{attention_const_max,ring_attention_const_max_fwd}.py "$D/code/" ;;
  esac
  rm -rf "$D/neffs"
  tar -czhf "$R/attn_pairs_$S.tar.gz" -C "$OUT" "$S" && echo "  -> $R/attn_pairs_$S.tar.gz ($(du -h "$R/attn_pairs_$S.tar.gz" | cut -f1))"
  prev="$R/clean_$S.txt"
done
echo "== ATTENTION KERNEL PROGRESSION =="
cat "$OUT"/*/report.txt 2>/dev/null
