# Body visual experiment implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent encoder and probe tasks; parent integrates cache/data, comparison and server workflow. Do not commit automatically.

**Goal:** Run frozen gap80, frozen RGB and RGB+gap80 fusion comparisons from one reproducible visual cache.

**Architecture:** Independent visual experiment package; strict temporal/provenance alignment to the existing corpus. Preserve existing joint training and inference behavior.

**Tech Stack:** Python, NumPy, PyTorch, PIL, existing PyAV decoder and NPU device helper.

---

- [x] Encoder/extraction: `training/visual/encoder.py`, `training/visual/extract.py`, `tests/test_visual_extract.py`; causal grid, letterbox, bounded extraction, resumable per-record caches and atomic exclusive completion markers.
- [x] Probe training: `training/visual/model.py`, `training/visual/train.py`, `tests/test_visual_training.py`; context independence/availability, exact CPU resume and checkpoint provenance.
- [x] Preparation/data: `training/visual/data.py`, `training/visual/prepare.py`, `tests/test_visual_data.py`; frozen checkpoint corpus binding, cached context, train-only normalization and bounded window reads.
- [x] Comparison/suite: `training/visual/suite.py`, `training/visual/report.py`, `tests/test_visual_suite.py`; same sample identities, baseline-H cohorts, all-known-positive denominators and verified epoch resume.
- [x] Server instructions and preflight: explicit official weights+SHA256, isolated inherited NPU environment, resumable pipeline, nohup logs and completion artifacts.
- [x] Independent review, targeted regressions, synthetic end-to-end test and honest real-data limitations.

## Verification evidence (2026-09-24)

- Visual suite/data/training/extract/encoder: 26 tests passed before the final atomic-publication fix.
- Pipeline/weights + existing head_policy/body_dist_experiments/body_dist_suite: 20 tests passed (3 new + 17 existing).
- After atomic-publication fix, extract/data/pipeline: 17 tests passed, including the new interrupted-manifest regression. Total distinct tests across these runs: 47.
- Real official ResNet18 weights loaded strictly on CPU: finite float32 `(1,512)` features, all parameters frozen. No torchvision dependency.
- Both shell scripts passed `bash -n`; training wrapper passed syntax-only `--check-config` with the correct original manifest ID. Official weight full SHA256 verified.
- Independent review findings resolved: source/encoder/marker checks, bounded finite validation, interrupted checkpoint fallback, manifest path typo, and atomic exclusive weight/manifest publication.
- No real-data training, NPU execution or accuracy improvement is claimed by local tests. Server setup/extraction/training remains the user's server run, with commands in `docs/training/BODY_VISUAL_COMPARISON.md`.
