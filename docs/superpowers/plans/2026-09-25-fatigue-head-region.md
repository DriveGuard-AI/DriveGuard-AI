# Fatigue head-region rejection and recovery

The supplied negative previews contain face meshes on hands. Driver ROI containment
does not verify a face. Add an opt-in, per-record spatial experiment before changing
training caches. Spatial containment itself must never be described as verified eyes.

Design: preserve v1 experiments and the original baseline identity. Add v2 configs
with video-hash/image-size-bound head regions and three arms: original full frame,
full frame plus head-region acceptance gate, and head crop plus the same gate.
Keep detector thresholds unchanged. Gate before canonical feature extraction;
rejected candidates have no landmarks or pose. Do not change hand/object ROI.
An explicitly configured region must contain the whole accepted landmark set;
no interpolation, temporal extrapolation, automatic head boxes, or label-dependent
gates. Crop pose remains experimental. Missing/incorrect record scope fails before
replay output. Report outside-region acceptance/rejection separately from numeric
eye coverage, split by fatigue labels; preserve all window denominators.

Implementation sequence:
- [x] Add failing tests for false-hand rejection, correct-head retention, matrix
  association, crop remapping, baseline parity and scope failures.
- [x] Implement v2 configuration, optional spatial gate and candidate diagnostics.
- [x] Integrate region audit counts, previews and fatigue comparison CSV.
- [x] Add prepare/build CLI using existing preview picker and per-record exports;
  add server runner and operator documentation.
- [x] Run focused and related regression tests, CLI help, shell syntax and diff checks.

Validation: synthetic external detector/decoder APIs exercise real conversion,
canonical feature extraction and report writing. Actual perception quality and
fatigue TP/FP require the server dataset, reviewed head regions and a subsequent
full replay. No production weights or existing caches are rewritten in this change.

Verification: 41 head-region/face experiment/replay/crop/gaze tests, 17 fatigue
tests via discovery (required by their existing fixture imports), and 2 real JPEG
sample tests in the perception environment passed. CLI help, Bash syntax and
diff whitespace checks passed. Server video quality evaluation remains pending.
