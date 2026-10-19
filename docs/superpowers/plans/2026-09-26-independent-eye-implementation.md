# Independent eye branch implementation plan

**Goal:** Deliver the approved three-arm, two-seed eye comparison with fixed reference yawn predictions.

**Architecture:** Native-eye extraction and independent frame training are bounded modules. The suite owns reference reproduction, matched cohorts, event composition, reporting, previews and server orchestration. Use subagent-driven development for isolated modules and independent final review.

- [x] Native cache: implement `training/visual/eye_frames.py` and `tests/test_eye_frames.py`, exposing `build_eye_cache(head_cache, dataset_root, output_dir, continue_run=False)` and `EyeCache(root, head_cache)`. Arrays: images uint8[N,3,224,224], slot_indices/source_pts_ms int64[N], valid bool[N], roi_xyxy float32[N,4], native_eye_width float32[N]. Valid is original freshness AND accepted eye geometry; invalid pixels/geometry zero. Manifest binds head cache hash, source/model identity and preprocessing. CLI independent of Torch, record-level resumption, preview metadata.
- [x] Independent trainer: `training/visual/eye_train.py`, `tests/test_eye_train.py`. `EyeDataset(head_cache, eye_cache, labels, split, mode)` unique matched ROI-valid explicitly known eye frames; metadata access and trained-subject sampling. `train_eye(train, val, reference_model, settings, output_dir, provenance, subject_balanced=False, resume=None)` initializes from the SAME selected reference. Return best/last paths. Loader reconstructs strict, hash-checked `eye_v1` model. Exact epoch resume and terminal-stage checks.
- [x] Integration tests first: reference yawn identity under missing eye, original windows and pure-eye denominators, no validation labels in predictions or sampling, shared A/B frame cohort, six matched runs and idempotent completed-stage reuse.
- [x] Suite `training/visual/fatigue_eyes.py`: verify diagnostic/action references, load eye cache, source label integrity; immutable plan, initial audits, cached frozen yawn, six trainings, best/last fixed-threshold eye-only and composed event metrics, by-subject/subtype/pair changes. No promotion.
- [x] Server wrappers for separate perception/NPU stages, one-command sequential launcher, documentation with nohup/resume and environment overrides. Fail promptly on missing prerequisites.
- [x] Run new tests and related regressions, CLI help in each path, Bash syntax, whitespace checks; independent review and fix findings before delivery. Real data/NPU performance remains pending user execution.


## Verification completed

- Training environment: 41 tests passed across eye comparison/training, head adaptation, fatigue actions, eye diagnostics, visual data and encoder.
- Perception environment: 16 native-eye/head-cache tests passed; Torch-free import checked.
- Both CLI help paths, Python compilation, server Bash syntax/LF and tracked/untracked whitespace checks passed.
- Two independent reviews completed. Fixed best-checkpoint selection being coupled to early-stopping tolerance, with a failing-then-passing regression covering resumed cumulative improvement.
- Synthetic suite executes six trainings, retains original window cohorts and reference yawn positives, verifies identical initialization, reuses completed results and rejects inconsistent result metadata.
- Actual source-video crop quality and NPU performance remain pending server execution. Code has not been committed or pushed in this task.
