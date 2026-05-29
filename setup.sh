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

# ─── Model caching (S3-backed PVC tar pattern) ───────────────
MODEL_TAR="/var/mdl/wan2_2_i2v/Wan2.2-I2V-A14B.tar"
MODEL_LOCAL="/tmp/Wan2.2-I2V-A14B"

if [[ -f "$MODEL_TAR" ]]; then
  echo "Copying model tar from S3 cache..."
  cp "$MODEL_TAR" /tmp/Wan2.2-I2V-A14B.tar
  echo "Extracting..."
  tar xf /tmp/Wan2.2-I2V-A14B.tar -C /tmp/
  rm -f /tmp/Wan2.2-I2V-A14B.tar
else
  echo "Downloading model from HuggingFace..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='${MODEL_LOCAL}', local_dir_use_symlinks=False)
"
  echo "Creating tar archive for S3 cache..."
  tar cf /tmp/Wan2.2-I2V-A14B.tar -C /tmp Wan2.2-I2V-A14B
  mkdir -p "$(dirname $MODEL_TAR)"
  cp /tmp/Wan2.2-I2V-A14B.tar "$MODEL_TAR"
  rm -f /tmp/Wan2.2-I2V-A14B.tar
  echo "Cached tar to S3!"
fi
echo "Model weights ready at $MODEL_LOCAL"

# ─── Profiling setup ─────────────────────────────────────────
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_OUTPUT_DIR=/tmp/neuron_profile
export NEURON_RT_INSPECT_DEVICE_PROFILE=session
mkdir -p /tmp/neuron_profile
echo "============================================"
echo "  Profiling ENABLED"
echo "============================================"

# ─── Launch with torchrun (TP=2, single chip) ────────────────
export WAN_DIR="${SCRIPT_DIR}"
export MODEL_PATH=/tmp/Wan2.2-I2V-A14B

cd "${SCRIPT_DIR}"
torchrun --nproc_per_node=${TP_DEGREE:-2} --master_port=29500 \
  "${SCRIPT_DIR}/inference_neuron_i2v.py" 2>&1 || true

# ─── Neuron Explorer analysis ────────────────────────────────
PROFILE_DIR="/tmp/neuron_profile"
NTFF_DIR=$(find "$PROFILE_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | head -1)
if [[ -n "$NTFF_DIR" ]]; then
  echo ""
  echo "============================================"
  echo "  NEURON EXPLORER ANALYSIS"
  echo "============================================"

  neuron-explorer view -d "$NTFF_DIR" \
    --output-format summary-text \
    --ignore-dma-trace 2>&1 | tee /tmp/neuron_explorer_summary.txt || true

  neuron-explorer view -d "$NTFF_DIR" \
    --output-format json \
    --output-file /tmp/neuron_explorer_profile.json \
    --ignore-dma-trace 2>&1 | tee /tmp/neuron_explorer_view.log || true

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
cp -r /tmp/neuron_profile "$ARCHIVE_DIR/profiles" 2>/dev/null || true
echo "Archive complete!"
ls -la "$ARCHIVE_DIR/" 2>/dev/null || true

echo "═══════════════════════════════════════════"
echo "  Done!"
echo "═══════════════════════════════════════════"
