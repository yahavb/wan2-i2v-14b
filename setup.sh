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

# ─── Profiling DISABLED (saves ~5GB HBM on single chip) ──────
# To re-enable: set NEURON_RT_INSPECT_ENABLE=1 in the job manifest
# export NEURON_RT_INSPECT_ENABLE=1
# export NEURON_RT_INSPECT_OUTPUT_DIR=/tmp/neuron_profile
# export NEURON_RT_INSPECT_DEVICE_PROFILE=session
# mkdir -p /tmp/neuron_profile
echo "============================================"
echo "  Profiling DISABLED (HBM conservation)"
echo "============================================"

# ─── Launch with torchrun (TP=2, single chip) ────────────────
export WAN_DIR="${SCRIPT_DIR}"
export MODEL_PATH=/tmp/Wan2.2-I2V-A14B

cd "${SCRIPT_DIR}"
torchrun --nproc_per_node=${TP_DEGREE:-4} --master_port=29500 \
  "${SCRIPT_DIR}/inference_neuron_i2v.py" 2>&1 || true

# ─── Neuron Explorer analysis (only if profiling was enabled) ─
PROFILE_DIR="/tmp/neuron_profile"
if [[ -d "$PROFILE_DIR" && -n "$(ls -A $PROFILE_DIR 2>/dev/null)" ]]; then
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
  fi
else
  echo "Profiling was disabled — skipping neuron-explorer"
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
