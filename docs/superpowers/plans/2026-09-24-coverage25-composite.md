# Coverage25 output-base comparison plan

User approved comparing coverage25 outputs plus the existing gap80-context
C_fusion branch, preserving the existing low-FPR thresholds, with parallel
body-fatigue diagnosis. No new training or changes to original caches/weights.

## Design

- Keep the current gap80 baseline and three C_fusion operating points.
- Optionally load coverage25 with its own checkpoint normalizer and config.
  Require identical corpus fingerprints and all window IDs, raw labels and
  metadata. Different H availability is allowed because policies differ.
- Compose current C_fusion body distraction (including its original gap80
  missing-RGB fallback) with coverage25's other behavior outputs and gaze.
  Preserve frozen gap80 H groups for body error analysis. Never pass coverage25
  embeddings to C_fusion. Audit both sources independently.
- Export both standalone bases and six compositions in JSON/CSV/window output.
  No new threshold selection; validation remains development evidence.
- Record per-stage cached-feature evaluation wall time and model tensor bytes;
  state explicitly these are not raw-video end-to-end latency or production
  combined-device peak memory. Document model dependencies of each composition.
- Inspect body fatigue evidence, score distributions, masks and gates using
  original checkpoint preprocessing. Separate measurements from causal hypotheses.

## Implementation / verification

- [x] Add fail-first composition tests for differing H policies, labels/order,
  unchanged C branch including fallback, alternate normalizer and provenance.
- [x] Implement optional output-base checkpoint and a new server wrapper.
- [x] Add bounded profiling helper with honest memory/timing scope and tests.
- [x] Parallel fatigue audit with synthetic tests, export source-bound summaries.
- [x] Run all affected unit/integration tests and Bash syntax/CLI help checks.
- [x] Document commands, expected artifacts and server-only real-data execution.

Existing skills: brainstorming (approved design above), writing-plans, TDD,
parallel-agents, verification-before-completion. Use existing working branch;
do not commit or start server jobs without the actual server artifacts.

Verification: 17 composition/cost tests, 9 fatigue diagnostic tests and 7 existing
diagnostic regressions passed. Both CLI help entries and Git Bash syntax checked.
Root review corrected actual drowsiness subset naming and eliminated duplicate
imported test discovery. Real NPU metrics remain pending server execution.
