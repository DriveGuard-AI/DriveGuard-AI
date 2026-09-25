# Fatigue spatial comparison implementation plan

**Goal:** Diagnose persistent subject7 fatigue misses and test whether head-region RGB improves them, using the user's approved whole/head/both experiment.

**Architecture:** Export verified old-model window evidence; extract a separate head RGB cache on exactly the original causal sample grid; train matched temporal heads on whole, head, or both frozen embeddings. Original models, labels, caches and other tasks remain immutable.

**Constraints:** Ordinary BCE, two seeds, same epochs/early stopping/threshold. No subject-specific thresholds or label-driven cropping. One shared driver-relative crop `[0.05,0,0.65,0.65]` is a candidate spatial prior, not a claim of complete head coverage. Export its previews and state manual review remains necessary. Test split is never evaluated for selection.

## Tasks and ownership

- [x] Independent diagnostic module `training/visual/fatigue_audit.py` and tests: re-evaluate published RGB baseline with checkpoint/provenance/cohort checks; export all known scores, per-subject distributions and all positive-window contact sheets; verify source hashes and causal PTS, distinguish window labels from frame labels. Explicit identity for resume reuse.
- [x] Root `training/visual/spatial.py`, `extract.py`, `data.py`: versioned crop cache, normalized driver-relative rectangle, source ROI unchanged in provenance, no MediaPipe dependency, exact grid/PTS/validity match; train-only head normalization. Red tests for malformed crops, cache mismatch and future/stale sampling.
- [x] Root paired dataset: 1024 input dimensions in all arms, zero unused half for matched capacity; same masks, labels, subjects, window IDs, temporal model, settings. Tests prove ablation, train-only normalization and no F gate.
- [x] Root spatial suite/pipeline: ordinary BCE, seeds42/43; reproduce previous best before experiments, checkpoint selection by validation BCE; fixed0 threshold metrics per subject/record with total FN; validation-only threshold diagnostics explicitly named. Persist plan and per-arm provenance; resume verified completed stages only.
- [x] Root server command/docs: diagnostic first, separate head extraction (reuse local weights), then six sequential training runs. No downloads or changes to existing production outputs. Expose audit-only mode and normal resume.
- [x] Run new tests red then green, relevant existing extraction/training/data/pipeline/early-stop/all-task regressions, Bash CLI and diff checks, independent review.

Verification: 74 related tests passed in one combined run, including audit-first test ordering, original batch-size reference replay, source-label mismatch rejection, paired cache identity, ordinary BCE, exact resume, and legacy all-task output checks. Bash syntax and wrapper help invocation passed. Independent review's batch-size finding was fixed and covered by a regression test.

## Acceptance

No hardcoded61 denominator; preserve actual unknown labels and all positive windows. Report subject7 andsubject14 separately without treating validation tuning as generalization. A real improvement requires observed TP/FN/FPR from server data; local synthetic tests establish software contracts only. An audit failure stops extraction/training. Mechanical provenance checks cannot prove semantic labels or full head coverage; report this explicitly.
