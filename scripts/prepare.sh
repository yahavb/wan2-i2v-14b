#!/bin/bash
# prepare.sh - deps + model download only (no run, no profiler).
# Extracted from setup.sh so the 3-run benchmark harness (wan2-i2v-14b-job.yaml)
# can reuse it without triggering the old single-run / neuron-explorer flow.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Install deps ─────────────────────────────────────────────
cd "${SCRIPT_DIR}"
# Ported from wan2-ti2v-5b-tp8-job.yaml.
# New SDK 2.31 DLC image does NOT auto-activate a venv -> `uv pip install`
# errors with "No virtual environment found". Target the SAME interpreter
# python3 uses (where torch-neuronx already lives) via --python.
UVPY="$(command -v python3)"
echo "uv target interpreter: $UVPY"

sed -i '/flash_attn/d' wan_requirements.txt
sed -i '/torchaudio/d' wan_requirements.txt
# SDK 2.31 image ships numpy 2.x; the SDK's own NKI kernels use np.long (an alias
# only in numpy>=2.0). The `numpy<2` pin downgrades to 1.26.4 -> "module 'numpy'
# has no attribute 'long'" neuronx-cc exit 70. Strip the pin, keep the image's numpy.
sed -i '/numpy/d' wan_requirements.txt
uv pip install --python "$UVPY" -r wan_requirements.txt
uv pip install --python "$UVPY" "setuptools<81"
uv pip install --python "$UVPY" git+https://github.com/pytorch/vision.git@v0.25.0 --no-deps --no-cache --no-build-isolation

# SDK 2.31 DLC ships a python3 built WITHOUT the _lzma stdlib extension.
# torchvision eagerly imports datasets/utils.py -> `import lzma`, which crashes
# `import torchvision`. We never open .xz files, so stub lzma (raises only if used).
"$UVPY" - <<'PY'
import importlib.util as u, pathlib
loc = u.find_spec('torchvision').submodule_search_locations[0]
p = pathlib.Path(loc) / 'datasets' / 'utils.py'
s = p.read_text()
if '_no_lzma' not in s:
    stub = (
        'try:\n'
        '    import lzma\n'
        'except ImportError:\n'
        '    import types as _t\n'
        '    def _no_lzma(*a, **k):\n'
        '        raise RuntimeError("lzma unavailable in this Python build")\n'
        '    lzma = _t.SimpleNamespace(open=_no_lzma, LZMAFile=_no_lzma)\n'
    )
    s = s.replace('import lzma\n', stub, 1)
    p.write_text(s)
    print('Patched torchvision datasets/utils.py: lzma replaced with stub')
else:
    print('torchvision datasets/utils.py already patched')
PY

# einops is absent from the SDK 2.31 image and not in wan_requirements.txt.
uv pip install --python "$UVPY" imageio imageio-ffmpeg einops

# ─── Download model directly (too large for tar/S3 cache) ────
MODEL_LOCAL="/tmp/Wan2.2-I2V-A14B"
if [[ -d "$MODEL_LOCAL" && -f "$MODEL_LOCAL/Wan2.1_VAE.pth" ]]; then
  echo "Model already present at $MODEL_LOCAL"
else
  echo "Downloading model from HuggingFace..."
  "$UVPY" -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-I2V-A14B', local_dir='${MODEL_LOCAL}', local_dir_use_symlinks=False)
"
fi
echo "Model weights ready at $MODEL_LOCAL"
