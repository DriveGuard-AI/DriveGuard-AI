# Independent eye branch comparison

User approved three matched directions after eye diagnostics: independent eye supervision on head pixels, native eye pixels, and native eye pixels with subject-balanced sampling. Current reference is `B_action_supervision-seed42-best-event`, epoch 4, 19/61 TP, 0/215 FP. Its pure-eye subtype remains 0/13; validation eye ROC-AUC is 0.515 versus training 0.882.

## Input and isolation

- Keep the selected action checkpoint and all its yawn predictions immutable. New eye models clone its encoder, normalization and eye classifier; their private encoder can change without changing the reference yawn path.
- Extract one eye crop from native decoded video at the existing eight sampled slots per second. Reuse the original source clock, labels, source hashes and prepared windows. Do not upscale the 224px head cache to invent eye detail.
- Use existing face landmarker asset in the CPU perception environment, restricted to the original head crop. Choose the larger projected eye corner span without reading labels, using a square around its corners so closed eyelids do not shrink the crop. Reject absent/ambiguous/outside-head/tiny/invalid geometry. This is a candidate ROI, not human-verified eye visibility.
- Record crop coordinates, native span, validity and detector policy. Missing eyes reset only the replacement eye event; reference yawn predictions stay available. Missing eyes never become negative labels.
- Both head-control and eye arms use the same unique known, ROI-valid frame cohort. This isolates crop content from detector coverage. Report coverage by subject and positive/negative labels, including all original fatigue windows and subtypes.

## Three arms and model selection

`A_head_independent`, `B_eye_native`, `C_eye_subject_balanced`, each seeds 42 and 43, run sequentially on one NPU. A/B differ only in image input; C changes only training subject sampling. All start from the same eye weights, use the same epoch budget, optimize only explicitly labelled eye states, and select on the same per-subject balanced eye BCE. Keep best and last checkpoints and the initial reference diagnostics; no validation threshold search is used to select event results. All class weighting and sampling counts are training-only. No frame/random split across recordings; retain train/validation subjects.

## Evaluation

Reproduce the published reference exactly before training. Cache reference frame logits once; splice only the new eye logits into the causal event decoder with the original 1000ms policy and source gaps. Evaluate eye-only and frozen-yawn-plus-eye at fixed zero threshold, retaining all 61 positives/215 negatives and unknown windows. Preserve recovered/lost positives, new/removed false positives, by-subject, by-subtype and known-frame ROC/AP metrics. An invalid new eye cannot suppress a reference yawn event. The original weak eye contribution is replaced, not accumulated silently.

Export label/crop coverage and previews for subject14 and every other subject. They expose miscropping or timing problems; no claim that ROI quality or higher resolution guarantees better accuracy. Success requires pure-eye recovery with acceptable actual window false positives, not auxiliary loss alone. No deployment routing changes or automatic model promotion.

## Operations

Two environments: perception CPU builds the immutable/resumable native eye cache; visual NPU performs six matched trainings and evaluation. Both stages reuse completed artifacts only after identity/hash checks. Python standard library/NumPy/Pillow cache path must not require Torch in perception environment. Existing originals remain untouched. Publish code only on a subsequent explicit push request.
