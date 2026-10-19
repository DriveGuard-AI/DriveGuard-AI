# Eye false-negative breakpoint analysis

User approved the preceding design: diagnose the 11 input-limited pure-eye windows and the one supported but missed window, without training or modifying thresholds, weights, or locator policy.

**Architecture:** `eye_breakpoints.py` verifies the completed recovery report and its fixed sources, reproduces full-validation inference, and exports causal frame traces, per-window diagnoses and cached head/eye playback. `eye_tracking.py` exposes passive `last_trace` telemetry without changing its outputs. `eye_tracking_replay.py` replays only requested recordings from their beginning, checks sampled output against the original dense_track cache, and reports actual source-frame stop reasons. CPU perception and NPU inference remain separate processes.

**Scope:** All validation windows remain in inference and totals. All pure-eye positive windows, including the recovered control, get analysis and playback. Source replay includes their recordings, never starts tracking from a selected positive frame, and never uses labels to locate eyes. Negative/unknown windows are retained in the window table. Existing caches cannot supply historical stop reasons; they must be measured by verified replay, not guessed.

- [x] Write failing causal trace tests: missing pixels, negative score, source reset, exact duration boundary, carried history, repeated PTS, endpoint failure despite earlier event, unknown labels.
- [x] Implement trace and window report with explicit run-duration conventions, frozen full-scope reproduction, provenance and no-overwrite handling.
- [x] Add head/eye animated playback plus seekable HTML frame views for pure-eye windows; show missing inputs as missing and include current score/run time.
- [x] Add passive locator stop telemetry with unchanged return values and tests (delegated using subagent-driven-development).
- [x] Add verified CPU source replay, detailed stop counts and shell wrapper for the two phases.
- [x] Run targeted tests, integration checks, CLI and shell checks; review implementation. Document server command and requested results. No commit/push this turn.

Tests use synthetic causal sequences and small local image fixtures. Real video, NPU reproduction and actual per-window diagnoses require server execution.

## Verification and review

- 56 perception/locator/source replay/cache tests passed, including real translated-image replay, exact cached pixel comparison, first expiration vs subsequent no-seed distinction, provenance/runtime mismatch and damaged-output rejection.
- 35 fixed-inference/trace/event/eye-training regression tests passed. Full-scope synthetic integration retains positives, negatives and unknowns; mismatching published results fail.
- Red/green completion regression verifies entry-page write failure never publishes completion, completed output is reused without inference, and modified artifact hashes fail. Independent review found this completion/resume issue; it is fixed.
- CLI help, Python compilation, Bash syntax/help, Torch-free CPU imports, tracked and new-file whitespace checks passed. Rendered synthetic playback image inspected for readable score/timing/missing-input text.
- Source replay also validates the aggregate stop table on completed reuse. No original policy, return dictionary, cache or weight changed. No commit/push. Real video and NPU evaluation remain pending server execution.
