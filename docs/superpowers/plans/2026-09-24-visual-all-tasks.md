# Visual all-task evaluation implementation plan

> **For agentic workers:** Use test-driven-development and verification-before-completion; independent strategy investigation runs in parallel at user request.

**Goal:** Evaluate a frozen joint model plus body-only visual distraction head.

**Architecture:** A read-only runner verifies prepared features and checkpoints,
collects baseline predictions, aligns body probe windows, composes outputs and
exports per-task reports with an invariance audit. No training or checkpoint edits.

**Tech stack:** Existing Python, NumPy, PyTorch and unittest; server Bash wrapper.

- [x] Add `tests/test_visual_all_tasks.py`: behavioral tests for body-only replacement,
  unknown/unavailable accounting, face/gaze preservation, exact window matching,
  threshold identity and a real synthetic baseline/probe integration.
- [x] Run `.venv-training/Scripts/python.exe -m unittest tests.test_visual_all_tasks`;
  verify missing implementation is the failure.
- [x] Add `training/visual/all_tasks.py` and focused reporting helper if needed.
  Reject mismatched cohort/provenance; export summary, task CSV and window scores.
- [x] Add `scripts/server/evaluate_visual_all_tasks.sh` selecting seed42/default,
  seed43/default and seed43/validation-FPR from the existing stability report.
- [x] Run the new tests plus visual data/training/stability regressions, CLI help
  and Bash syntax validation. Review report schema and missing-evidence cases.
- [x] Document invocation and real-data limitations, and review the separate
  body-dist strategy document against actual results.

Verification: 33 tests passed (all_tasks, data, training, stability, early_stop),
CLI help and Git Bash syntax check passed. The evaluator has eight tests, including
real synthetic model inference. Independent review corrected missing gaze CSV
rows and null metrics for unsupported tasks. Real NPU validation is pending
server execution because its checkpoints and feature caches are not local.
