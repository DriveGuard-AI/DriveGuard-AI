# Fatigue head stability implementation plan

**Goal:** Test the approved lower-learning-rate / stronger-regularization direction using verified existing head RGB features.

**Architecture:** Add a separate validation experiment suite. Reuse SpatialData, train_probe, epoch-resume logic and fatigue metrics. Do not change shared training behavior, data, feature extraction or task routing.

**Tech stack:** Python, PyTorch/torch_npu, unittest, Bash.

## Design

The preceding spatial experiment recovered subject7 positives with head-only features, but later epochs increased false positives. Compare four arms at seeds 42 and 43: A_control (published settings), B_slow (learning rate /10), C_regularized (dropout .4, weight decay .01), D_slow_regularized (both). Regularization is a bundled intervention; this experiment cannot isolate dropout from weight decay. All other settings and head-only input remain identical. Initialize each training run afresh, not from the previously selected validation checkpoint.

Reuse the completed spatial comparison report, preparation and head cache. Before creating output, replay both published B_head best checkpoints at their original evaluation batch size and verify identity, settings, provenance, hashes, cohort and all published metrics. Reject source/output overlap. Immutable plan and verified epoch checkpoints support explicit continuation.

Report best and last at fixed logit zero, per subject and overall. Include same-seed changes against control, last-minus-best degradation, and across-seed summaries. Keep FPR<=3% sweeps diagnostic-only, preserve unknown labels and unavailable positives, and do not automatically deploy a winner. Best checkpoint selection remains unweighted validation BCE. Existing training histories provide per-epoch loss and overall confusion matrices.

## Tasks

- [x] Add failing integration tests using the existing real tiny fatigue/spatial fixture: 4 runs per seed, reference replay, parameter isolation, identical cohorts, per-subject CSV, continuation, and tamper rejection.
- [x] Implement `training/visual/fatigue_head_stability.py` with report validation, four arms, verified replay, training/evaluation, paired summaries and CLI.
- [x] Add `scripts/server/train_body_fatigue_head_stability.sh` and operator documentation with one nohup command, 8 sequential runs, no extraction/download, and output locations.
- [x] Run targeted suite tests and related visual regression tests, CLI help and Bash syntax checks. Review diff and report local validation separately from server effectiveness.

Run tests with `.venv-training/Scripts/python.exe -m unittest tests.test_fatigue_head_stability -v`. The first run must fail because the new suite is absent; the completed suite must pass against actual tiny CPU training rather than mocked predictions.

Validation: 34 related tests pass, including eight real tiny training runs across both seeds, exact completed continuation, interrupted-stage reuse, and altered checkpoint/settings/cohort rejection. CLI help, Python compilation and Bash syntax pass. This validates implementation only; real NPU fatigue effectiveness remains to be measured.
