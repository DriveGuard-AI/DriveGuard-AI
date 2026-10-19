# Live composite service implementation plan

User approved assembling the existing latest models and selected live webcam input.
No retraining, threshold search, or commits/pushes are part of this change.

**Goal:** One versioned model bundle and one webcam HTTP entry point, using real
inference for coverage25, gap80/C_fusion and yawn/native-eye/dense_track.

**Architecture:** Export selected artifacts from the existing all-task/recovery
reports. Run PyTorch in the inference environment and perception in a separate
Python subprocess. A single active camera session owns all state. Browser JPEG
frames retain capture timestamps; causal 25 Hz resampling uses only past frames,
and stale samples remain invalid. Output once per second after four-second warmup.
Use existing normalizers, group masks, RGB preprocessing and event decoder.

**Scope:** A local/tunnel-only development HTTP service and bundled webcam debug
page, not the complete business API or a production safety decision service.
All tasks expose available/score/decision/source; body gaze is marked unvalidated.
No camera recording, annotations, training caches, or dataset are needed at runtime.

1. Add tests for task routing, unavailable evidence, exact frame sampling,
   cross-session reset, timestamp rejection, artifact checksum binding and HTTP
   lifecycle. Run `.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_live_*.py`.
2. Add `inference/live/bundle.py`: immutable artifact export/verification with
   relative paths; preserve upstream checksum sidecars and report selections.
3. Add `inference/live/routing.py`, `timeline.py`, `models.py`: route only body
   fatigue/distraction; full-frame and head/eye encoders remain independent;
   retain gap80 fallback and independent branch normalization.
4. Add `inference/live/perception_worker.py` and `bridge.py`: bounded JSONL IPC,
   one ordered stream, subprocess stderr separated from protocol, timeout cleanup.
5. Add `inference/live/server.py` and `camera.html`: loopback HTTP, explicit session
   view/ROI/ownership, one in-flight frame, no unbounded queues, status expiry,
   token-protected requests, browser camera stop/error cleanup.
6. Add server launch/export scripts and deployment runbook. Verify Python tests,
   HTTP integration with synthetic boundary models, shell syntax and JS syntax.
   Report that genuine checkpoint/NPU/camera validation requires the server assets.

**Acceptance:** Frozen branch routing agrees with the existing offline functions;
gaps cannot create closed-eye evidence or SAFE output. Incompatible bundles fail
before serving. Startup and camera errors remain visible and do not return a stale
result as current. Live transport latency is measured separately from capture time.
