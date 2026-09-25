# Body fatigue visual recovery

Implement the user's requested direct attempt to recover fatigue detection when F is unavailable.

1. Test fatigue label selection, RGB availability independent of F, positive weighting, and total-positive reporting.
2. Add explicitly task-bound fatigue preparation using the existing verified RGB cache. Preserve legacy distraction preparation and reject cross-task use in distraction pipelines.
3. Train RGB control, balanced RGB, and balanced fusion with two seeds. Calculate weights from known, RGB-available training windows only; choose checkpoints by unweighted validation BCE. Keep threshold logit zero for primary results.
4. Publish per-window, per-subject, and overall validation counts, including unavailable positives in total FN; keep validation threshold sweeps diagnostic only.
5. Provide a resumable one-command server runner, verify tests and Bash syntax. No downloads, no original-cache changes, no change to existing multitask predictions.

Local synthetic tests establish plumbing correctness only. Actual fatigue TP improvement requires running on the user's NPU server. Existing 0/61 F eligibility is distinct from model TP.
