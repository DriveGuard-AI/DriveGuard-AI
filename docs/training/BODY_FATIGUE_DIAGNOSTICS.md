# Body/face fatigue evidence diagnostics

`training.gaze.fatigue_diagnostics` evaluates an existing trusted checkpoint on its
original **validation** feature corpus. It does not train, extract video features,
change crops/MediaPipe, select thresholds, or evaluate the test split. Predictions
use the training boundary `fatigue_logit >= 0`, including tiny negative logits
whose float32 sigmoid rounds to 0.5.

Run both checkpoints separately on the Ascend host, from the repository root:

```bash
python -m training.gaze.fatigue_diagnostics \
  --checkpoint /mnt/workspace/DriveGuard-AI/runs/body-dist-comparison-001/gap80/train-001/best.pt \
  --backend npu --device-index 0 --batch-size 64 \
  --output-dir /mnt/workspace/DriveGuard-AI/runs/body-fatigue-gap80-diagnostics-001

python -m training.gaze.fatigue_diagnostics \
  --checkpoint /mnt/workspace/DriveGuard-AI/runs/relation-gaze-head-coverage25-npu-001/best.pt \
  --backend npu --device-index 0 --batch-size 64 \
  --output-dir /mnt/workspace/DriveGuard-AI/runs/body-fatigue-coverage25-diagnostics-001
```

Use the host's existing training Python environment with `torch_npu`. The output
directories must be new and outside the feature data root. Saved manifest/data
paths are used by default. If the corpus moved, append `--manifest /path/manifest.json
--data-root /path/features`; all original corpus fingerprints must still match.
The checksum sidecar beside each trusted checkpoint is required. Both models use
their **own saved configuration and normalizer**; no policy or normalizer is copied
between gap80 and coverage25. `--backend cpu` supports local fixture verification.

Defaults are `--split val --view all --subset all`; `all` views selects body and
face only. `--view body`, `--view face`, and `--subset drowsiness`,
`--subset distraction` or `--subset gaze` narrow the report without changing inference.
The manifest calls fatigue recordings `drowsiness`; `fatigue` is a task name, not
a subset name. Keep all subsets to examine fatigue negatives and unknown labels outside the drowsiness
subset. There is deliberately no threshold option and no test-split option.

## Outputs and denominators

`windows.csv` contains recording/subject/view/subset IDs, inclusive window timing,
raw label known/unknown, target, prediction availability, logit/probability,
outcome, F eligibility/block reasons, all group quality values, and **fatigue**
F/O/HO/FO gate weights. Unavailable predictions are empty, not probability 0.5.
Unknown labels stay unknown even when a score is available.

Raw and effective validity ratios are emitted for all 23 canonical features.
EAR left/right, MAR, PERCLOS, closed duration and blink rate also have current
validity, last valid-at-current-slot values, and window means over valid slots.
These means use the original feature units, not normalized model inputs. Missing
values are empty, never an observed zero. Pair validity requires simultaneous
left/right eye evidence. No new fatigue event definition is inferred.

The manifest's feature definitions are copied into `summary.json`. The canonical
extractor defines EAR from the eye landmark distance ratio, MAR as
`d(13,14)/d(78,308)`, PERCLOS over 100 slots with at least 80% valid evidence,
closed duration from an observed closure onset, and blink rate from completed
80–500 ms events (`count * 15` over four seconds). The diagnostic trusts stored
values/masks and does not recompute these signals. Effective masks come directly
from the training dataset: freshness/dependencies, conservative 100-slot eye
history filtering for PERCLOS/closed duration/blink rate, then saved normalizer
support. Raw extraction validity can therefore exceed model-input validity.

Per-feature mask ratios divide by window slots. F group quality divides by the
number of slots times the number of **supported F features**. F includes eye,
mouth, head and iris features; F eligibility does not prove eye/mouth completeness.
Fatigue availability is any eligible F/O/HO/FO group. The optional H branch serves
distraction only and does not make fatigue available.

`summary.json` groups counts by view, label class, view+label, subject, subset,
view+subset+label, F eligibility and view+label+F eligibility. Each has raw
positive/negative/unknown counts, available and known-unavailable counts,
TP/FP/TN/FN, coverage and precision/recall/F1. Confusion metrics exclude unavailable
and unknown labels, matching training. `detected_fraction_of_all_known_positives`
includes unavailable positives in its denominator.

Available-score/gate quantiles include count, min, p05/p25/p50/p75/p95, max and mean.
Use `groups.view_label['body:positive']` versus `['face:positive']` to inspect
label-conditional evidence, scores and gates. `face_features_absent` and
`fatigue_features_absent` count windows with no effective slot for any feature in
the corresponding set; they do not mean an image-level face detector failed.
`f_eligible_below_zero` identifies available windows still predicted negative
despite F eligibility. Per-feature ratios distinguish that from intact fatigue
signals. `f_ineligible_zero_gate` records enforced F masking;
`f_eligible_zero_gate` and gate quantiles describe eligible-stream weighting.

These are conditional observations, not causal attribution. Low F weights alone
do not establish why fatigue failed, and absent effective features do not identify
the extraction cause. Validation already participated in checkpoint selection;
overlapping windows are not independent events or subjects. Synthetic tests verify
the implementation, not the actual NPU checkpoint accuracy.

New summaries also record `window_steps`, `stride_steps`, and
`manifest_file_sha256`. To investigate and compare recovery of the missing raw
eye/temporal inputs, use the [fatigue recovery replay](BODY_FATIGUE_RECOVERY.md).
That replay measures input evidence, not classification accuracy, and leaves this
checkpoint and its original feature corpus untouched.

## Integrity and resource limits

Existing output directories, output/data overlap, corpus fingerprint differences,
and nonfinite predictions/gates are rejected. Inputs are opened read-only; hashes
for checkpoint, sidecar, manifest and every feature/label array are checked again
before publishing `summary.json`. A directory with only a partial CSV is incomplete
and must not be used as a successful report. No original file is overwritten.

Inference uses bounded batches and the existing four-recording mmap cache. Scalar
scores are retained for exact aggregate quantiles; full feature windows and model
activations are not retained. Full-corpus hash verification adds read I/O before
and after evaluation, including files outside the validation selection; those
other splits are not passed to the model.

```powershell
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p test_fatigue_diagnostics.py
```

Tests cover real synthetic model/corpus inference, raw/effective history masks,
unknown/unavailable cases, tiny negative logits, body/face label grouping, saved
head policies, training-metric agreement, batch tolerances, CLI execution, source
immutability, changed input rejection, output overlap and nonfinite predictions.
