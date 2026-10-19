#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run on the Linux NPU server only.' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
NPU_BASE="${NPU_BASE:-/opt/buildtools/Python-3.11.4/bin/python3}"
"$NPU_BASE" -c 'import torch,torch_npu; assert torch.npu.is_available(); print(torch.__version__,torch_npu.__version__)'
if [[ ! -e .venv-visual-npu ]]; then
  "$NPU_BASE" -m venv --system-site-packages .venv-visual-npu
fi
VISUAL_PY=.venv-visual-npu/bin/python
[[ -x "$VISUAL_PY" ]] || { echo 'Existing visual environment is not usable.' >&2; exit 2; }
"$VISUAL_PY" -m pip install --no-cache-dir --no-deps 'av==12.3.0' 'Pillow==10.4.0'
"$VISUAL_PY" -c 'import av,PIL,numpy,torch,torch_npu; assert torch.npu.is_available(); assert "/.venv-visual-npu/" not in torch.__file__; assert "/.venv-visual-npu/" not in torch_npu.__file__; print("Visual dependencies ready",av.__version__,PIL.__version__)'
"$VISUAL_PY" -m training.visual.weights --download --path "${WEIGHTS:-models/local/resnet18-f37072fd.pth}" \
  --timeout "${WEIGHTS_TIMEOUT:-300}" --attempts "${WEIGHTS_ATTEMPTS:-3}"
echo 'Visual setup complete. NPU torch is reused; no torchvision or MediaPipe training required.'
