#!/bin/bash
# setup.sh - Main entrypoint for Wan2.2-I2V-A14B on Neuron (single chip)
# Installs deps, downloads model, launches torchrun
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Install deps ─────────────────────────────────────────────
cd "${SCRIPT_DIR}"
sed -i '/flash_attn/d' wan_requirements.txt
sed -i '/torchaudio/d' wan_requirements.txt
uv pip install -r wan_requirements.txt
uv pip install "setuptools<81"
uv pip install git+https://github.com/pytorch/vision.git@v0.25.0 --no-deps --no-cache --no-build-isolation
uv pip install imageio imageio-ffmpeg

# ─── Download model directly (too large for tar/S3 cache) ────
MODEL_LOCAL="/tmp/Wan2.2-I2V-A14B"

if [[ -d "$MODEL_LOCAL" && -f "$MODEL_LOCAL/Wan2.1_VAE.pth" ]]; then
  echo "Model already present at $MODEL_LOCAL"
else
  echo "Downloading model from HuggingFace..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='${MODEL_LOCAL}', local_dir_use_symlinks=False)
"
fi
echo "Model weights ready at $MODEL_LOCAL"

# ─── Native PyTorch Neuron profiler (captures NEFF+NTFF pairs) ────
export NEURON_PROFILE_ENABLE=1
export NEURON_PROFILE_DIR=/tmp/neuron_profile
echo "============================================"
echo "  Native Neuron Profiler ENABLED"
echo "============================================"

# ─── Launch with torchrun (TP=2, single chip) ────────────────
export WAN_DIR="${SCRIPT_DIR}"
export MODEL_PATH=/tmp/Wan2.2-I2V-A14B

cd "${SCRIPT_DIR}"
WORLD_SIZE=$(( ${TP_DEGREE:-4} * ${SP_DEGREE:-2} ))
torchrun --nproc_per_node=${WORLD_SIZE} --master_port=29500 \
  "${SCRIPT_DIR}/inference_neuron_i2v.py" 2>&1 || true

# ─── Neuron Explorer: analyze traces ───────────────────────
PROFILE_DEST="/tmp/neuron_profile"
if [[ -d "$PROFILE_DEST" && -n "$(ls -A $PROFILE_DEST 2>/dev/null)" ]]; then
  echo ""
  echo "============================================"
  echo "  NEURON EXPLORER ANALYSIS"
  echo "============================================"

  # Copy NEFFs into session dirs so explorer can match NEFF+NTFF pairs
  if [[ -d "$PROFILE_DEST/neffs" ]]; then
    for session_dir in $(find "$PROFILE_DEST" -name "*.ntff" -printf '%h\n' 2>/dev/null | sort -u); do
      cp "$PROFILE_DEST/neffs/"*.neff "$session_dir/" 2>/dev/null || true
    done
    echo "Copied NEFFs into session directories"
  fi

  NTFF_DIR=$(find "$PROFILE_DEST" -name "*.ntff" -printf '%h\n' 2>/dev/null | head -1)

  echo "=== neuron-explorer view (summary-text) ==="
  neuron-explorer view -d "$NTFF_DIR" \
    --output-format summary-text \
    2>&1 | tee /tmp/neuron_explorer_summary.txt || true

  echo ""
  echo "=== neuron-explorer view (JSON) ==="
  neuron-explorer view -d "$NTFF_DIR" \
    --output-format json \
    --output-file /tmp/neuron_explorer_profile.json \
    2>&1 | tee /tmp/neuron_explorer_view.log || true

  echo ""
  echo "=== NEFF COUNT AND SIZE DISTRIBUTION ==="
  NEFF_COUNT=$(find "$NTFF_DIR" -name '*.neff' | wc -l)
  echo "Total NEFFs: $NEFF_COUNT"
  find "$NTFF_DIR" -name '*.neff' -exec ls -l '{}' ';' | awk '{print $5}' | sort -n | awk '
    BEGIN { count=0; sum=0 }
    { sizes[count++]=$1; sum+=$1 }
    END {
      if (count == 0) { print "No NEFFs found"; exit }
      printf "Total size: %.2f MB\n", sum/1024/1024
      printf "Min: %d bytes\n", sizes[0]
      printf "Max: %d bytes (%.2f MB)\n", sizes[count-1], sizes[count-1]/1024/1024
      printf "Median: %d bytes\n", sizes[int(count/2)]
      printf "Mean: %.0f bytes\n", sum/count
      printf "\nSize buckets:\n"
      small=0; med=0; large=0; xlarge=0
      for(i=0;i<count;i++) {
        if(sizes[i]<10000) small++
        else if(sizes[i]<100000) med++
        else if(sizes[i]<1000000) large++
        else xlarge++
      }
      printf "  <10KB  (tiny):   %d\n", small
      printf "  10-100KB (small): %d\n", med
      printf "  100KB-1MB (med):  %d\n", large
      printf "  >1MB  (large):    %d\n", xlarge
    }'
else
  echo "WARNING: No profile directory found for neuron-explorer"
fi

# ─── Archive results to S3 ───────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="/var/mdl/wan2_2_i2v/runs/${TIMESTAMP}"
echo ""
echo "============================================"
echo "  ARCHIVING to S3: $ARCHIVE_DIR"
echo "============================================"
mkdir -p "$ARCHIVE_DIR"
cp /tmp/wan2_i2v_output.mp4 "$ARCHIVE_DIR/" 2>/dev/null || true
cp /tmp/neuron_explorer_summary.txt "$ARCHIVE_DIR/" 2>/dev/null || true
cp /tmp/neuron_explorer_profile.json "$ARCHIVE_DIR/" 2>/dev/null || true
cp /tmp/neuron_explorer_view.log "$ARCHIVE_DIR/" 2>/dev/null || true
cp -r "$PROFILE_DEST" "$ARCHIVE_DIR/profiles" 2>/dev/null || true
echo "Archive complete!"
ls -la "$ARCHIVE_DIR/" 2>/dev/null || true

echo "═══════════════════════════════════════════"
echo "  Done!"
echo "═══════════════════════════════════════════"
