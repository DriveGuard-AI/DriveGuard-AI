# Body fatigue evidence recovery

User authorized investigation and implementation of the highest-priority failure.
This first stage addresses the observed missing-input bottleneck. It cannot claim
classification recovery without real-video and subsequent held-out evaluation.

## Design

Keep the existing face-experiment replay, provenance checks, source clock, isolated
MediaPipe variants and original caches. Add an opt-in fatigue audit that recomputes
the canonical six fatigue features from the candidate face observations. Missing
object/hand packets are explicit; those experts are not needed for these six
features. No new pseudo-labels, interpolation or invented closed-eye evidence.

Use saved diagnostic group profile and normalizer support to measure effective
fatigue features and F eligibility. Window labels use the final slot, just like
training. Window/stride are explicit; older diagnostic summaries do not contain
them. New diagnostic summaries will record them. Per-frame counts and per-window
counts stay separate; count positives, negatives and unknowns independently.

Compared routes: more epochs/threshold tuning cannot restore absent input;
replacing the entire perception model lacks current evidence; replaying controlled
perception changes is the bounded next experiment with falsifiable evidence.
Full-frame VIDEO baseline, IMAGE mode, and three separate confidence changes are
compared. Existing reviewed per-record crop configs remain supported separately.

## Implementation sequence

- [x] Add tests for fatigue label selection, canonical feature reproduction,
  eye-gap history filtering, supported-feature masks, positive window denominators,
  short prefixes, no mutation and refusal of mismatched diagnostic provenance.
- [x] Add `training/dmd/fatigue_replay.py`: bounded temporal feature audit and
  aggregations with raw/effective coverage, feature values, rejection reasons,
  mask/value parity and training-window evidence. No torch or NPU imports.
- [x] Add opt-in integration to `face_experiments.py`; defaults remain the old
  distraction audit. Add `training/dmd/fatigue_recovery.py` CLI requiring the
  fatigue diagnostic summary, selecting all body validation subsets by default.
- [x] Record window geometry in new fatigue diagnostic summaries; old summaries
  require explicit window and stride rather than silently assuming them.
- [x] Add server launcher and runbook: existing perception environment, no
  downloads, no training, new output directory; summary and per-frame/per-window
  CSV plus label-conditioned preview gallery. All subjects selected by default.
- [x] Run new integration tests, existing face experiments/replay/crop and
  fatigue diagnostics tests, CLI help and shell syntax; inspect final diff.

## Acceptance

Tests prove feature and labeling correctness and immutable inputs. Server reports
must separately show baseline reproduction, raw eye recovery, effective temporal
recovery and F eligibility on all original positive windows. Coverage is not
accuracy; a recovered candidate still needs landmark review, a separately
versioned extraction/retraining run, and all-task regression evaluation. If
perception stays absent, report that outcome instead of starting futile training.

## Verification

2026-09-24: fatigue discovery ran 17 tests successfully. Face experiment/replay/
crop/gallery suite ran 28 tests, with 27 passes and one OpenCV test skipped in the
training environment; both gallery tests then passed in the perception environment.
Thus all 45 distinct relevant tests have passed. CLI help, perception-only imports
(no torch), Git diff whitespace checks and Git Bash `bash -n` passed. Local workspace
has no original Dataset; actual recovery and classification performance remain
unmeasured. No production model or feature corpus has been changed.
