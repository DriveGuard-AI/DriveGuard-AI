# Body distraction: next experiments plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute the investigation tasks sequentially. Any conditional implementation needs a separate design and implementation plan after its evidence gate passes.

**Goal:** Identify and improve the cause of body distraction false negatives without sacrificing precision through unmeasured changes.

**Architecture:** Start with existing per-window predictions and feature masks. Separate input availability from classifier use of available evidence; evaluate perception replacements separately before changing training data. This document plans investigations and decision gates, not an approved replacement model design.

**Tech Stack:** Existing Python diagnostics, JSON/CSV reports, frozen feature caches and checkpoints; perception environment for bounded video experiments, NPU environment only for justified training experiments.

---

## Implementation update after user requested several directions together

The user subsequently authorized coding multiple comparable directions to accelerate iteration. The bounded implementation is specified in `docs/superpowers/specs/2026-09-23-body-dist-comparison-design.md` and operated through `docs/training/BODY_DIST_COMPARISON.md`.

- [x] Audit existing last-checkpoint CSV: H-ineligible body positives have O-only evidence in 150 windows and O+HO in 2; all 152 are FN. This does not identify an H gate bug.
- [x] Add original train/val input and supervision audits, optional missing-head positive sampling and label-independent full-face evidence dropout.
- [x] Add four-arm suite: unchanged coverage25 baseline, sampling, face_dropout, coverage25+gap80; isolate each factor with the same seed/schedule/data.
- [x] Add best/last diagnostics and provenance-checked comparison JSON/CSV, preserving unavailable-positive denominators and other task metrics.
- [x] Add restart identity checks, retained checkpoint lineage and failure reports containing completed valid results.
- [ ] Run the real NPU suite and judge precision/recall/coverage tradeoffs. Local synthetic tests do not establish real-data improvements.

The remaining investigation gates below still apply to new perception pipelines and any subsequent model/cache promotion.

## Evidence and scope

- The full 448.64-second crop experiment reproduces baseline runtime and masks.
- Head coverage: 2132/11216 baseline, 2350/11216 crop. Positive head coverage declines from 1194/2698 to 1170/2698.
- Under coverage25, positive eligible windows remain 33/107: 11 gained and 11 lost. These are raw-mask proxies, not classifier recall.
- Fixed crop is not promoted to training caches. This is one recording, not a conclusion about every body recording or every crop method.
- Previous coverage25 best/last diagnostics both miss all 152 H-ineligible positive body windows. This correlation motivates auditing other inputs; it does not prove a gate defect or a missingness shortcut.
- Validation data has already influenced development. Final generalization assessment requires subjects/recordings untouched by these choices.

## Task 1: Audit existing H-ineligible predictions first

**Read:** `training/gaze/diagnostics.py`, `training/gaze/head_model.py`, `training/gaze/profile.py`; previously supplied coverage25 best/last `windows.csv` and corresponding summaries.

- [ ] Verify checkpoint, data fingerprint, window keys and known-label denominators before joining best/last rows.
- [ ] Partition body positives and negatives by H eligibility and the exact F/O/HO/FO eligibility pattern. Report counts, TP/FP/FN/TN and unavailable predictions separately; reconcile totals to the source report.
- [ ] Within H-ineligible windows, report effective feature valid ratios, group quality, gate weights and probability distributions. Treat weights as descriptive evidence, not causal feature importance.
- [ ] Compare the same strata in training data using its original split and saved preprocessing, without refitting the normalizer. Check whether missing-head positive examples are scarce or receive no task supervision.
- [ ] Produce a concise audit identifying one of: insufficient alternative inputs; available inputs with weak separation; or a verified availability/supervision inconsistency. Do not claim hand/object cues alone identify every distraction event.

**Decision:** A verified inconsistency gets a focused regression fix. Insufficient inputs prioritize perception. Available but weakly used inputs justify a controlled training experiment after verifying training coverage.

## Task 2: Build a bounded, representative perception comparison

**Read:** `training/dmd/face_experiments.py`, `training/dmd/face_diagnostic_samples.py`, `inference/experts/face_experiment_adapter.py`.

- [ ] Select a fixed review set spanning recordings/subjects, positives/negatives, successful/failed detections and side-facing, downward-looking, forward-leaning and occluded frames. Include temporal segments; isolated IMAGE calls cannot reproduce VIDEO state.
- [ ] Inspect existing failed frames and landmark overlays. Record visible face, approximate face extent, occlusion, blur and whether landmarks fit the face. Do not infer an internal detector failure solely from an empty final Face Landmarker result.
- [ ] Research at most two alternative detection/landmark/head-pose pipelines against official sources. Verify ARM64 CPU/runtime feasibility, model availability, licenses, output conventions and inference cost before selecting one for implementation.
- [ ] Compare the baseline and selected candidate on identical inputs and labels. Report positive/negative coverage, window gains/losses, missing runs, visual geometry failures and runtime by recording.

**Decision:** Continue only if positive eligible windows show a meaningful, repeatable net gain across recordings, with acceptable manually reviewed geometry and runtime. Set the quantitative promotion criterion before candidate runs, after reviewing denominators; do not select solely by aggregate coverage. Confirm coordinate calibration before publishing features.

## Task 3: Run only the training intervention supported by Task 1

**Read:** `training/gaze/head_model.py`, `training/gaze/model.py`, `configs/training/relation_gaze_head_coverage25.json`, and the existing training dataset/loss implementation.

- [ ] If positive missing-head examples are underrepresented and alternative inputs are useful, design one stratified-sampling experiment before considering loss or architectural changes.
- [ ] If branch dependence remains plausible, design a face/head evidence dropout comparison. H duplicates head information also present in F: masking only the H embedding does not simulate absent head evidence. Consistently update affected values, masks, quality, eligibility and task supervision; retain valid alternative evidence and do not invent positive evidence.
- [ ] If alternative inputs are mostly unavailable, defer dropout and reweighting; repair input coverage first.
- [ ] Freeze data, split, seed, schedule and selection rule for each comparison; change one factor at a time. Keep existing best and last checkpoints as references and save checkpoints needed by the new predeclared selection rule.

**Decision:** Training is conditional, not the immediate next job. Define and test the selected intervention separately; this plan does not authorize unreviewed feature semantics or fabricated missing values.

## Task 4: Evaluate the intended behavior and stop unproductive branches

- [ ] Report body precision, recall, F1, FPR, known-input coverage and confusion counts, separately for H-eligible and H-ineligible cohorts. Include all known positives in coverage-aware reporting.
- [ ] Check face-view distraction, gaze and the other behavior tasks for regressions. Compare recall at a predeclared false-positive budget as well as the existing operating threshold.
- [ ] Use only training/development data to choose intervention, thresholds and checkpoints; evaluate the frozen candidate once on an untouched subject-disjoint test set if available. If none exists, label results as development results.
- [ ] Promote a perception candidate only after versioning extraction configuration, coordinate conventions and caches consistently across splits. Refit normalization only on training data; never mix incompatible old/new head-pose features silently.
- [ ] Stop a candidate with no reproducible positive-window or downstream benefit. Do not respond to failed evidence by automatically extending training epochs or adding more threshold/crop variants.

## Immediate deliverable

Execute Task 1 first using existing diagnostic files. Deliver the feature-availability and error breakdown, plus a single justified next experiment. Tasks 2 and 3 describe conditional follow-on work, not simultaneous full-scale training and re-extraction.
