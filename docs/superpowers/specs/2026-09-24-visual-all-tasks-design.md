# C_fusion combined all-task evaluation

The user approved the preceding proposal: keep the frozen gap80 joint model for
all outputs except body distraction, where a selected C_fusion checkpoint is
used. Evaluate this composite, without retraining, on identical windows. A second
parallel investigation proposes evidence-based body-distraction experiments.

Read existing PreparedData, its exact baseline checkpoint, and fusion checkpoints.
Verify checkpoint hashes, preparation provenance, corpus fingerprint, all body
window identities and labels. Run baseline once and probes in eval mode. Preserve
all original task scores and masks; substitute only body distraction. If RGB is
unavailable, explicitly retain the baseline output and count the fallback.

Publish baseline and composite metrics overall and by view, subject and subset,
with raw labels, input availability, effective metrics, all-positive detection,
and gaze confusion matrices. Report an equality audit for unchanged outputs.
Save window predictions for paired error analysis. Unknown labels remain unknown.

Use seed42 best at logit zero as the default-F1 candidate. Also evaluate seed43
best at zero and at its validation-selected low-FPR threshold, binding threshold
to the source report checkpoint SHA256 and window-label hash. Do not optimize a
threshold on test data. Default split is validation; no deployment claims.

Use a new output directory, never modify caches/checkpoints. Local synthetic
integration verifies execution; real NPU evaluation requires server artifacts.
Historical aggregate recomposition, if shown, is explicitly distinguished from
a fresh end-to-end evaluation.
