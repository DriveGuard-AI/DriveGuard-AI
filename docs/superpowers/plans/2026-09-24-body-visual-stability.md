# Body visual attribution/stability implementation

Apply subagent-driven-development to independent trainer changes and parent orchestration; do not commit automatically.

- [x] Extend VisualProbe with matched context input ablation; test identical initialization and RGB independence.
- [x] Add optional early stopping and persisted restart state; test deterministic stopping/resume and legacy compatibility.
- [x] Generalize stage finalization for early-stopped terminal epochs without changing existing completion behavior.
- [x] Add fixed-global-threshold subgroup reporting and nine-stage stability runner, provenance checks and paired-seed aggregates.
- [x] Add server wrapper and docs, test end-to-end/cache reuse/continuation/config mismatch and run existing visual regressions.
- [x] Review code and publish local validation limits (no NPU or real-data result claims).

Verification: `tests.test_visual_training`, `tests.test_visual_early_stop`, `tests.test_visual_stability`, `tests.test_visual_suite`, `tests.test_visual_pipeline`, `tests.test_visual_data`: 29 tests passed on CPU. Includes six-stage/two-seed synthetic integration, exact resumed model/optimizer/RNG/history, unchanged input files and completed checkpoints on continuation, terminal early-stop recovery, and global-threshold subgroup consistency. Server wrapper passed Bash syntax check and CLI help invocation. Independent review found no significant correctness issues in experiment fairness, reporting or continuation. No real-data/NPU experiment was run locally.
