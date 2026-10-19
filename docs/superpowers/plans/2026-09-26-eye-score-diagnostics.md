# Eye score diagnostics implementation plan

> Execute inline in the current authorized task, with test-driven verification and an independent code review before delivery.

**Goal:** Explain the selected action model's missed closed-eye events without another training run.

**Architecture:** Reuse verified head pixels and the selected action checkpoint. Separate frame ranking statistics from causal event replay, preserving original timestamps, masks, windows and duration policy. Training-derived thresholds are exploratory, never independent calibration.

**Tech stack:** Python, NumPy, PyTorch/NPU, unittest, Bash.

## Tasks

- [x] Add `tests/test_eye_diagnostics.py`: perfect ranking below zero, tied scores, one-class and unknown labels; shifted eye threshold must restart/recompute closure duration; yawn-only and eye-only attribution retain all windows.
- [x] Run `.venv-training/Scripts/python.exe -m unittest tests.test_eye_diagnostics -v` and verify the new API is missing.
- [x] Add `training/visual/eye_scores.py`: `score_summary(scores, targets, known)` returns quantiles, fixed-zero counts, grouped ROC/AP and diagnostic thresholds, with null for undefined metrics.
- [x] Extract `replay_events(cache, predictions, rows, resets, closed_eye_min_ms, eye_threshold=0., branches='both')` from existing event evaluation. Keep default replay byte-for-byte equivalent; `eye_threshold=None` disables eye positives. Unknown labels never enter event inference.
- [x] Add `training/visual/eye_diagnostics.py`: verify report/checkpoint/cache/labels, reproduce published event and action metrics, infer train and val once, export frame CSV and score distributions, compare branch-only and training-derived eye thresholds at unchanged yawn threshold/duration. Preserve subtype and subject breakdowns and paired changes.
- [x] Add pipeline tests for report reproduction, stale provenance rejection, output protection, validation labels not selecting exploratory thresholds, and completed report contents.
- [x] Add server wrapper and usage documentation with nohup command and exact output paths. No training, downloads, default threshold changes or test-set use.
- [x] Run new and existing action/head adaptation tests, Python CLI help, Bash syntax and `git diff --check`; review the diff independently and fix findings. Leave publishing for the user's explicit push instruction.

## Acceptance

The report distinguishes frame ranking from fixed-zero classification and sustained-event detection. All curves group ties. Invalid/unknown frames are counted but excluded from binary statistics. Validation-derived sweeps are marked diagnostic only; training-derived candidates never read validation targets. The original selected event metrics must reproduce before a complete result can be emitted. Server results remain pending until execution against the real NPU artifacts.

## Verification evidence

- 37 tests passed across eye diagnostics, actions, head adaptation training/cache/pipeline; subsequent output-protection assertions passed in both pipeline tests.
- CLI help, Bash syntax and git diff checks passed. Independent reviewer found no actionable issues and independently passed all six new tests.
- Uploaded report schema/default variant/epoch4/seed42 and 1000ms duration policy verified against the actual attachment.
- Real server/NPU replay remains pending. No checkpoint or deployment routing was changed, and no commit/push was performed.
