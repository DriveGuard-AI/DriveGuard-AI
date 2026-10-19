# Fatigue action supervision implementation plan

Approved direction: preserve D43 as reference, learn eye closure and yawning
from original annotations, compare direct fatigue output with causal event logic.

Goal: test a different supervision strategy with the existing 32-frame pixel
cache and unchanged validation cohort. No new model downloads or pixel extraction.

Architecture: compile masked per-source-frame action labels using existing
OpenLABEL parsing, camera shifts and actual PTS. Add two frame classifiers to
the warm-started adaptation encoder. Train direct fatigue plus masked auxiliary
losses. Evaluate direct and event readouts from the SAME checkpoint. Event logic
reads predicted actions only, with fixed zero logit thresholds and the original
closed-eye duration. It carries closure state along each recording, never across
recordings. Original annotations are used only for loss and evaluation.

First-run scope: reference plus paired direct-only control and auxiliary training
for seeds 42/43; compare auxiliary direct/event readouts. The control is required
to distinguish auxiliary supervision from another round of fine-tuning. Unknown
fatigue windows may carry known auxiliary labels; no unknown is a negative.
No threshold search on validation, no production composite replacement.

- [x] Add `training/visual/action_labels.py` with masked action compilation,
  full cached-fatigue reproduction, action subtype auditing, and source hashes.
  Tests: shifts, gaps, conflicting eyes, unknown yawning, exact duration, and
  future-frame invariance using authored OpenLABEL fixtures.
- [x] Add `training/visual/action_model.py` with warm-start preservation,
  independent per-frame action heads, masked multi-task loss and event decoder.
  Tests: exact fatigue replay, unknown-label gradients, frozen BN, no label
  arguments in inference, gap/reset and duration boundaries.
- [x] Add `training/visual/action_train.py` with matched control, deterministic
  epoch-level resume, auxiliary best checkpoint selected by validation action
  BCE (both readouts share it), separate fatigue-best diagnostic checkpoint,
  train-only class/coverage accounting and clear result metadata.
- [x] Add `training/visual/fatigue_actions.py` CLI: validate existing comparison,
  checkpoint, preparation, pixel cache and annotations; replay reference; run
  four trainings; publish overall/by-subject/subtype/action reports and scored
  windows. Preserve the original window order, hashes, labels and denominators.
- [x] Add server wrapper and documentation including nohup, continuation,
  checkpoint format, fixed thresholds, 8 Hz duration quantization, the fact that
  validation-selected D43 is not an independent test, and evidence limits.
- [x] Run meaningful unit and tiny end-to-end training/resume/replay tests,
  relevant existing head-adaptation/label tests, CLI help, bash syntax and diff check.

Commands: `.venv-training/Scripts/python.exe -m unittest tests.test_fatigue_actions`
then include `tests.test_head_adaptation`, `tests.test_head_adapt_train`,
`tests.test_head_frames`, `tests.test_head_adaptation_pipeline`, `tests.test_dmd_labels`.

Acceptance: inputs are immutable and corrupt/misaligned sources fail; fresh and
resumed tiny runs match; unknown labels do not contribute gradients; causal
event inference works without annotations; every validation row survives;
report explicitly separates eye/yawn/overlap/unresolved positive subtypes.

Verification: 59 related unittest cases passed; CLI help and bash syntax passed. Independent read-only review found no important defects. Full dataset/NPU execution remains a server-side step.
