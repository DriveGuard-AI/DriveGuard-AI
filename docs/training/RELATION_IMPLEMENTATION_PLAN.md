# Relation-gated CPU training implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the isolated model and reviews; controller integrates contracts/training with TDD. Work in the user's existing model branch. No commits, installs, remote runs or data downloads.

**Goal:** Deliver the four-stream relation-gated model, strict v2 feature input, shared CPU/CUDA/NPU training entrypoints, checkpoint reload/resume and an executable CPU synthetic smoke.

**Architecture:** Keep all v1 behavior intact. New v2 config selects relation_gated_v1 and embeds a versioned group profile; v2 manifest binds the same profile/hash and adds observed expert ages. Shared training selects model by schema/type; no MediaPipe/YOLOX import is required for feature-only training.

**Tech Stack:** Existing Python, NumPy, PyTorch; unittest. Optional torch_npu only on explicit NPU initialization.

## 1. Contract and dataset (controller)

Files: training/relation/profile.py, preprocessing.py; training/common/config.py and data.py; configs/training/relation_gated.json; tests/test_relation_data.py.

- [x] RED: v2 config loads; old v1 rejects added fields; manifest v1/v2 cannot be mixed; invalid ages/profile hashes/split leakage fail.
- [x] Implement profile with fixed F/O/HO/FO feature names, objects/face/hands age columns, 80ms expiry, quality threshold0.5 and 100-slot history. Hash equality binds manifest/config/checkpoint.
- [x] Implement pure NumPy preprocessing: propagate stale dependencies, conservatively invalidate history after gaps, normalize using training-only supported channels; quality[B,4,2] and eligible[B,4] derived from masks/ages, never labels. Age unknown=81; old cache cannot fabricate ages.
- [x] Extend bounded-mmap corpus through explicit v2 hooks; dataset returns a dictionary input plus target/label mask. Fit normalizer on dependency-filtered training slots. Preserve old tuple tensor input and fingerprints for v1.
- [x] GREEN: `.venv-perception/Scripts/python.exe -B -m unittest tests.test_relation_data -v`.

## 2. Independent model (subtask)

Files: training/relation/__init__.py, model.py; tests/test_relation_model.py.

- [x] RED: assert import exists, then test shapes, shared parameters, causality, missing masks, all-invalid mixed batch, finite gradients, checkpoint state roundtrip and <100000 parameters.
- [x] Implement `RelationGatedTCN(input_channels=38, group_indices=..., projection_dim=16, gate_hidden_dim=16, channels=(16,)*5, dilations=(1,2,4,8,16), kernel_size=3, dropout=.1, window_steps=100)`.
- [x] `forward(x, quality, eligible, return_aux=False)` accepts normalized [B,T,38], [B,4,2], bool[B,4]. Four independent projections -> one shared causal residual encoder -> four independent 72->16->4 gates -> weighted embeddings -> four logits. Auxiliary dictionary contains logits, weights[B,4,4], available[B]. All-invalid rows have finite zero weights, unavailable=True represented by available=False; masked feature values cannot leak.
- [x] GREEN: `.venv-perception/Scripts/python.exe -B -m unittest tests.test_relation_model -v`.
- [x] Independent spec review, then quality review; fix findings before handoff.

## 3. Shared training, configuration and checkpoints (controller)

Files: training/common/model_factory.py; training/temporal/train.py and cli.py; training/relation/cpu_train.py, gpu_train.py, npu_train.py; tests/test_relation_training.py.

- [x] RED: v2 training writes best/last and CPU tensors; resumed run equals uninterrupted CPU training; changed profile/data/model is rejected; all-input-missing batches do not step Adam; preserve label coverage separately from usable-input metrics.
- [x] Select factory from validated v1/v2 config. Move nested tensor input to selected device; mask supervision by input availability without relabeling unknown ground truth; count unavailable windows and known dangerous labels there. Reject no effective supervision/updates.
- [x] Reuse optimizer, masked BCE, metadata, RNG, content hashes, save and resume logic. Persist group profile/hash and model kind, supported channels and task order with weights. Expose safe checkpoint model reload for feature inference, not production alarm support.
- [x] CPU/GPU/NPU wrappers choose explicit backend and default relation config. `python -S -m training.relation.npu_train --check-config` must not import torch or initialize devices. Runtime paths via flags/env; no platform default fallback.
- [x] GREEN: relation training tests and full unittest suite.

## 4. Traceable v2 extraction and synthetic smoke (controller)

Files: inference/perception/export.py and video.py; training/relation/smoke.py; tests/test_relation_export.py and test_relation_training.py.

- [x] RED: opt-in v2 export adds real per-slot ages/hash; absent/failing experts age81, future observation rejected; labels remain unknown. v1 default unchanged.
- [x] Add `--group-profile` to JSONL/video feature export; embedded extractor profile must agree on fields/rate/max age. Use existing streaming output, no full-video buffering or in-place cache upgrade.
- [x] Smoke generates separate synthetic subjects and binary/unknown labels, missing evidence and unsupported pose; train2 epochs, resume to3, compare continuous3, load best for finite inference, report parameter count/hashes/coverage. Never report synthetic numbers as DMD accuracy.
- [x] Run `.venv-perception/Scripts/python.exe -B -m training.relation.smoke --output-dir runs/relation-smoke-20260919` (new directory only).

## 5. Documentation and final review

Files: docs/training/RELATION_RUNBOOK.md, DATA_CONTRACT.md, IMPLEMENTATION_PLAN.md; docs/architecture/RELATION_GATED_FUSION.md; docs/experiments/20260919-relation-training.md; docs/README.md; .ai-memory handoff/log.

- [x] Document exact CPU smoke and server GPU/NPU commands, hyperparameter/config files, data-root versus manifest, v2 arrays/example and checkpoint usage. Explain that raw DMD paths alone are insufficient: real labels, subject split and exported features are required. No implicit fake labels.
- [x] Mark implemented CPU skeleton separately from unverified GPU/NPU, real-data scores, optional expert-dropout augmentation, production task-support/alarms and optional export optimization.
- [x] Independent final spec/quality review, full unittest on existing CPU environments, configuration-only checks, document/link checks, diff whitespace check. Preserve unrelated worktree changes and leave branch uncommitted.

Acceptance example: `report['synthetic_only'] is True`, `report['resume_matches_continuous'] is True`, finite inference logits, no previous test regression; no assertion of real-data performance or hardware compatibility.

## Verification outcome

128 tests passed in each existing CPU environment (torch2.7.1+cpu and torch2.14.0+cpu). Independent model and integration spec/quality reviews passed after fixing ineligible latent leakage and usable-label coverage. Synthetic CLI report verifies13,524parameters,2→3 resume equality and best reload. No server, real data, install, commit or branch integration performed. Optional missing augmentation and production task-support/alarms remain outside this CPU skeleton delivery.
