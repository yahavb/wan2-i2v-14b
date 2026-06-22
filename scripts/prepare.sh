#!/bin/bash
# prepare.sh - deps + model download only (no run, no profiler).
# Extracted from setup.sh so the 3-run benchmark harness (wan2-i2v-14b-job.yaml)
# can reuse it without triggering the old single-run / neuron-explorer flow.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
