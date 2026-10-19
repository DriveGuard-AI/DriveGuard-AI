# Warm-start head visual adaptation implementation plan

**Goal:** Extend the approved best head fatigue checkpoint with controlled encoder fine-tuning and dense temporal inputs, preserving the original checkpoint.

**Architecture:** A new immutable RGB pixel cache supplies 8 or 32 causal frames from the same four-second windows. A separate end-to-end model restores the selected probe and original encoder. A standalone comparison suite replays the reference, starts four arms from identical weights, and reports initial/best/last overall and per subject.

**Tech stack:** PyTorch/torch_npu, PyAV/Pillow, NumPy memmaps, unittest, Bash.

## Experimental protocol

Input is the completed head stability comparison, default C_regularized-seed42-best (13 TP / 2 FP in current validation), plus its original local ImageNet ResNet18 weights and source videos. Never overwrite source checkpoints. Four arms: A_frozen8, B_tune8, C_frozen32, D_tune32. All start from the same selected best probe, not a fresh random probe. Seeds 42/43 govern data order and dropout; these are not independent pretraining seeds. Reset optimizer for this new adaptation phase; separate exact epoch resume handles interruption.

Dense slots each second: [2,5,8,12,15,18,21,24] on the original 25 Hz grid. Sparse frames are dense positions [3,7,11,15,19,23,27,31], exactly the old offsets. Same source hashes/PTS/crop/pixel preprocessing. Train and validation only; unknown labels remain unknown. Report availability for each arm and keep all original validation denominators.

Restore 1024-D head-only probe (whole half zero). Freeze original normalization, early encoder layers and all BN statistics. Fine-tuning updates layer4 only, with separate encoder LR. Sparse and dense models share a gated fourth causal convolution (kernel3,dilation9) initialized with gate zero: warm-start sparse logits match the old three-layer model, while training can expose a 33-frame receptive field. Parameter shapes are identical across arms. This is a common temporal extension, explicitly documented, not a pure sampler-only experiment. Preserve epoch-zero candidates when further training fails to improve BCE.

Default microbatch 2, accumulation 8, frame encoding chunk 16, head LR 1e-4, encoder LR 1e-5, encoder weight decay 1e-4, head decay inherited .01; dropout inherited .4; max 10 adaptation epochs, patience3. No image augmentation in the first controlled run. Validation BCE selects best; fixed logit0 primary, FPR03 diagnostic only. Export confusion matrices by subject and original-miss recoveries/lost positives. Do not auto-deploy or claim other-task improvement.

## Components and interfaces

- `head_frames.py`: build_frame_cache(prepared, crop, dataset_root, output_dir, continue_run=False); FrameCache(root); FrameDataset(prepared, cache, split, frames). Dataset returns ({images:uint8[T,3,224,224], visual_mask:bool[T]}, float32 target, bool known); metadata(index) returns original prepared row. `.frames`, `.metadata`, close resources.
- `head_adapt_train.py`: AdaptModel(probe_payload, encoder_state, frames, tune_encoder, frame_chunk=16); forward(images,visual_mask) returns logits/available. train_adaptation(train,val,probe_payload,encoder_state,settings,output_dir,provenance,frames,tune_encoder,resume=None); load_adaptation(path, backend,device_index); evaluate_adaptation(dataset,checkpoint,backend,device_index,batch_size) returns original metadata + predictions and payload. Standard checksum sidecars, model_kind distinct, self-contained encoder+probe in new checkpoints.
- `head_adaptation.py`: verify report/reference/encoder, build/reuse pixels, run four arms by seed with immutable plan, safe continuation, CSV/JSON summaries and old/new miss comparison. Server script and runbook.

## Tasks

- [x] Test and implement verified pixel cache, causal dense sampling, sparse equivalence, source corruption/continuation rejection.
- [x] Test and implement warm-start model, layer4-only gradients, frozen BN, generalized receptive field, masked BCE/accumulation, epoch-zero best, reload and exact epoch continuation.
- [x] Test and implement reference replay, arm isolation, configuration checks, checkpoint provenance, paired per-subject reporting and continuation orchestration.
- [x] Integrate, run related regression tests, independently review implementation, provide server launch command and accurate resource/quality limitations.

Verification: 67 related tests pass in one run (47.490 seconds). Includes actual ResNet forward/layer4 gradients, eight tiny-encoder training runs through the suite, old-source replay, unchanged source artifacts, exact epoch continuation, initial/best state identity checks, and missing normalization buffer rejection. CLI help and Bash syntax pass. Two component implementers independently reviewed the other components; no blocking findings remain. Actual NPU memory/throughput and real-video accuracy are not measured locally.
