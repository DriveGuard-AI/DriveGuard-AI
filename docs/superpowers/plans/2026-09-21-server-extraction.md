# Server extraction implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and TDD. Work in the user-designated model branch. No installs, downloads of models, commits or server access on the local machine.

**Goal:** Configure YOLO11s + MediaPipe gaze extraction and a resumable nohup server workflow with 6/3/3 video concurrency.

**Architecture:** Preserve batch-v2 ROI and feature-v3 contracts. One loopback-only YOLO11 NPU service owns detector weights; CPU video processes call it and run their own stateful MediaPipe tasks. A parent scheduler holds the existing batch lock, dispatches unique jobs with bounded subset quotas, writes per-job logs, and waits for children before releasing the lock. No automatic CPU fallback, no changes to 25Hz or labels. MediaPipe retains 2-face/4-hand ambiguity detection, confidence 0.5, VIDEO mode and transformation matrices.

**Tech Stack:** Existing Python/NumPy/MediaPipe/PyAV; isolated server Ultralytics 8.3.199 environment reusing the proven PyTorch/torch_npu installation; stdlib HTTP loopback IPC with token and bounded payloads. YOLO11s 640 FP32 initially, CPU NMS. Batch=1 service initially; batching is not claimed.

## Task 1: Detector adapter and service (independent implementation)

- [x] Add tests in tests/test_yolo11_service.py for YOLO11 RGB/letterbox preprocessing, COCO category filtering, raw-output dimensions, CPU NMS, remote configuration identity, unauthorized/malformed requests, disconnect and cleanup.
- [x] Implement inference/experts/yolo11_adapter.py, detector_service.py, detector_client.py. Export a lazy adapter factory for variant yolo11s, plus service client API. No vendor imports until local model construction, no automatic downloads.
- [x] Service CLI accepts a verified runtime JSON and writes readiness only after NPU warmup. Requests carry binary BGR frames, not pickle. Bind only 127.0.0.1; finite timeouts, length/dimension caps, authentication, and model/config identity checks. Independent CPU mock-boundary tests, no local weight files.
- [x] Verify with `.venv-training/Scripts/python.exe -m unittest tests.test_yolo11_service -v`; spec review then quality review.

## Task 2: Bounded scheduler and existing pipeline wiring (main)

- [x] Add tests/test_parallel_extraction.py, observe missing capability with `-m unittest tests.test_parallel_extraction -v`.
- [x] Add training/dmd/parallel.py and CLI options `--workers-dist 6 --workers-drows 3 --workers-gaze 3 --threads 1`. Parent alone owns `.batch.lock`; spawned workers process separate videos through existing _inputs/_existing/_extract. Failures are collected, SIGTERM stops new jobs and drains active work, no stale-lock stealing.
- [x] Preserve existing serial defaults and publish; source identity includes new code and detector service identity. Add per-job logs and summary; use record-atomic reuse, no partial video resume.
- [x] Wire runtime config for yolo11s service while retaining legacy strict validation and imports. No automatic modification of original ROI config.

## Task 3: Server-only setup and nohup lifecycle (main)

- [x] Add scripts/server extraction start/status/stop wrappers and setup helper. No execution on Windows; explicit Linux server checks. Dependencies never replace original torch/torch_npu. Fixed official YOLO11s source, local receipt/hash, license recorded AGPL-3.0; downloaded only when user executes setup on server.
- [x] Generate separate runtime and batch config preserving every confirmed ROI and preparation hash. Do not overwrite existing edited configs or model artifacts.
- [x] Start service first, wait for actual readiness with timeout, then run extraction via nohup. Distinct PID/log/status files; repeated starts rejected; stop only exact owned PIDs verified by command/start identity, or scheduler-specific stop request. Never kill broad process names or auto-remove foreign locks.
- [x] Document setup, start, tail logs, status, graceful stop, resume and publish. Explain server execution remains unverified and detector quality is not guaranteed by warmup.

## Task 4: Final verification

- [x] Run full unittest regression, stdlib CLI checks, `git diff --check`; inspect real protocol and spawned-process test results.
- [x] Independent spec review, then quality review; fix issues via regressions.
- [x] Update full training guide and experiment record. Preserve branch and user files; no claim of completed server extraction.
