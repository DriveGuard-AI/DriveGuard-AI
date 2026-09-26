# Eye continuity recovery implementation plan

User approved continuing diagnosis and recovery after 0/13 pure-eye windows had any continuous eligible input. No further training or publishing is authorized by this plan.

**Goal:** Separate face rejection reasons and compare source-frame detection/short tracking with frozen B seed42 eye weights and frozen original yawn.

**Design:** Preserve legacy native cache reader and extraction by default. Add explicit versioned recovery policies: sampled detection with detailed diagnostics; dense source-frame VIDEO detection; dense detection plus bounded forward/backward optical-flow eye tracking. Never infer closure from missing pixels. Track only previously confirmed eyes, using current-frame pixels, bounded by driver/head ROI, source gap, time since last detection, flow consistency and texture. Do not relax all-face geometry without rejection evidence. Cache all original validation records (including negatives and unknown windows); no label-dependent crop behavior. Restrict new caches to validation to avoid waste; mark exploratory validation explicitly.

**Architecture:** `eye_tracking.py` handles causal localization only. `eye_frames.py` integrates policies, detailed reason counts and immutable cache identities. `eye_recovery.py` verifies old reports/checkpoint, reproduces baseline, audits continuous evidence before fixed-model inference and emits paired per-subject/subtype results. Server wrapper runs CPU extraction then NPU evaluation, zero training.

- [x] Add tested causal locator with granular detector diagnostics and bounded real-pixel flow fallback.
- [x] Extend eye cache with backwards-compatible explicit policies, validation scope, detailed localization provenance and resume checks; test original timestamps/no future state leakage.
- [x] Add frozen-model recovery suite and input eligibility preflight to independent training. Preserve all windows, yawn outputs, source identities; test zero support stops training and fixed model evaluation.
- [x] Add one-command server wrapper, docs, resume support and output locations.
- [x] Run focused regressions/CLI/Bash checks, independent review, fix findings. Real-video recovery quality remains pending server execution.

Implementation uses subagent-driven-development for the bounded locator and independent final review, while root implements cache integration and evaluation. No commit or push unless explicitly requested.

Verification: 45 perception/cache/locator tests and 29 frozen-evaluation/training-support/action tests passed (74 total). Bash syntax and help checked using Git Bash with its Unix utilities on PATH; Python CLI/compilation and tracked diff whitespace checks passed. Independent review findings addressed: training support guard only applies when actually starting/resuming training; fixed B seed42 is enforced. Added real empty-face adapter integration regression and current-frame pixel tracking/cache test; completed caches can be read without MediaPipe/OpenCV.

Actual video recovery and NPU evaluation remain pending server execution. No new model was trained, and no commit/push was performed.
