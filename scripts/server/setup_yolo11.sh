#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run on the Linux NPU server only.' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
NPU_BASE=/opt/buildtools/Python-3.11.4/bin/python3
BATCH_SOURCE=${1:-runs/dmd-full-gaze.local.json}
BATCH_TARGET=runs/dmd-full-yolo11.local.json
[[ -f "$BATCH_SOURCE" ]] || { echo "Missing $BATCH_SOURCE" >&2; exit 2; }
[[ ! -e "$BATCH_TARGET" ]] || { echo "Already configured: $BATCH_TARGET (not overwritten)."; exit 0; }
"$NPU_BASE" -c 'import torch, torch_npu; assert torch.__version__.split("+")[0]=="2.7.1"; assert torch_npu.__version__=="2.7.1.post4"; assert torch.npu.is_available(); print("Reusing:",torch.__file__,torch_npu.__file__)'
if [[ ! -e .venv-yolo-npu ]]; then
  "$NPU_BASE" -m venv --system-site-packages .venv-yolo-npu
fi
YOLO_PY=.venv-yolo-npu/bin/python
[[ -x "$YOLO_PY" ]] || { echo 'Existing .venv-yolo-npu is not a usable server environment.' >&2; exit 2; }
export YOLO_AUTOINSTALL=false
"$YOLO_PY" -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu 'torchvision==0.22.1'
"$YOLO_PY" -m pip install --no-cache-dir -r requirements-yolo-npu.txt
"$YOLO_PY" -c 'import torch,torch_npu,torchvision,ultralytics; assert torch.__version__.split("+")[0]=="2.7.1"; assert torch_npu.__version__=="2.7.1.post4"; assert ultralytics.__version__=="8.3.199"; assert torch.npu.is_available(); assert "/.venv-yolo-npu/" not in torch.__file__; assert "/.venv-yolo-npu/" not in torch_npu.__file__; print("Detector dependencies ready; real NPU model warmup happens on start.")'
"$NPU_BASE" -S -m training.dmd.server_setup --config "$BATCH_SOURCE" --output "$BATCH_TARGET"
echo 'Setup completed on server. Original NPU and MediaPipe environments were not installed into.'
