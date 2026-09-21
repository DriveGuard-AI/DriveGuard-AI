# DMD Annotation Audit Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution; user requested coding in the existing model branch.

**Goal:** Standard-library-only, read-only DMD annotation inventory with actionable JSON/Markdown reports.

**Architecture:** `annotations.py` validates one document and computes inclusive interval coverage; `audit.py` safely inventories local files, pairs recordings, aggregates compact summaries and exposes the CLI. Existing strict JSON reader is reused; no model or video decoder imports.

**Tech Stack:** Python standard library, unittest, existing training.common.config.

## Task 1: Parser (RED → GREEN)

- [x] Add synthetic fixtures and tests in `tests/test_dmd_audit.py`, including inclusive interval union, invalid integers/reversed intervals, action/frame disagreements, unknown IDs, duplicate JSON keys and multi-axis labels.
- [x] Run `.venv-training/Scripts/python.exe -B -m unittest tests.test_dmd_audit -v`; confirm missing implementation assertion failures.
- [x] Create `training/dmd/__init__.py` and `annotations.py`. Return compact, JSON-safe summaries with source hash; do not copy personal attributes or raw frame payloads into reports.
- [x] Repeat tests until passing. Real-world interval discrepancies are diagnostics, not silent label correction.

## Task 2: Scanner and CLI (RED → GREEN)

- [x] Cover multi-recording sessions, gaze/hands pairing, subject aggregation, missing videos, unusual layouts, symlink/URI escape, empty/bad files, existing output protection and exit codes using temporary fixtures.
- [x] Create `audit.py` with `--dataset-root` and `--output-dir`; write `audit.json` and `audit.md` only to a new output directory outside Dataset. Continue after individual file errors; include progress on stderr.
- [x] Test CLI under `python -S` to enforce dependency isolation. Missing optional streams are warnings; malformed annotation/structural failures produce report and exit 2.

## Task 3: Real samples, regression and documentation

- [x] Parse the four user-supplied annotation files without copying them into Git. Record their observed labels, diagnostics and limitations; no actual video decoding claim.
- [x] Run the full unittest suite and `git diff --check`; review safety boundaries and report counts.
- [x] Add `docs/training/DMD_AUDIT.md`, link from training guide and documentation index. Update NPU status explicitly as user-reported synthetic validation, not DMD performance.
- [x] Hand off the exact server command and report paths in the runbook. No new training, dependency installation, branch merge or remote push.

## Acceptance checks

`python -S -m training.dmd.audit --help` works without third-party packages. One corrupt file does not prevent reporting other recordings. Label statistics count inclusive union frames per recording/type, not duplicated per-frame references. File identity, missing streams and sync declarations remain auditable. No automatic labels, split selection or video geometry changes are introduced.
