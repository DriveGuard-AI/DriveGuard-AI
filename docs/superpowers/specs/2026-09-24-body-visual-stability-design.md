# Frozen body visual attribution and stability

User approved following the observed C_best gain (105 TP/34 FP under validation FPR <=3%, versus A 89/31) with a context-only control, repeated seeds and smaller/regularized heads. Validation has two subjects; improvements remain development evidence, not held-out performance.

- Reuse the existing prepared directory from body-visual-comparison-001. No download, video decoding, feature extraction, context regeneration or normalization fitting.
- Three paired arms per seed (42,43,44): D_context_control, C_fusion, E_fusion_small. D and C use width64/dropout.1/weight_decay.0001. E changes only width to32; keep other settings identical to isolate capacity. All use unchanged learning rate .001, batch64, max20 epochs.
- D has the exact fusion architecture and initialization, but replaces normalized RGB inputs with zeros before projection. It retains the shared visual availability mask for a strict input ablation. Its constant visual branch may have learned biases but contains no RGB information. This is not a standalone production model for unavailable RGB.
- Early stopping: validation BCE, patience3, min_delta.0001. Strict best BCE determines best.pt; separate improvement anchor/counter handles patience. Save state in every last.pt before termination, resume exactly, and finalize interrupted terminal epochs without extra training. Old checkpoints/settings without early stopping remain valid and behave identically.
- New stability suite isolates outputs, binds settings/prepared identity, executes 9 stages sequentially, resumes verified stages and writes best/last reports. Baseline A is evaluated once. Fixed body windows, labels and baseline H cohorts across all arms.
- Report per seed, per arm mean/std/min/max, paired D-vs-C and C-vs-E differences. No seed cherry-picking. Report default threshold and diagnostic global threshold selected with validation FPR <=3%, with subject/record/H breakdown at that SAME threshold. Thresholds are not held-out estimates or automatically deployed.
- Server wrapper uses existing .venv-visual-npu; defaults prepared runs/body-visual-comparison-001/prepared-001, output runs/body-visual-stability-001. Document exact run count, nohup and continuation.
- Original A/B/C suite, saved models and immutable caches remain compatible. No changes to joint gaze training or production inference.
