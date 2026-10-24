# Decision log
One entry per decision, newest at the bottom. Source: team protocol (15-day plan).

## D-001: Branching and PRs
- Branch per member, PRs into main, smoke test must pass.

## D-002: Raw data never committed
- data/raw and data/processed are git-ignored; splits/ is tracked.

## D-003: Signal and sampling rate
- 3-axis accelerometer in g, anti-alias filtered, resampled to 20 Hz.

## D-004: Windows
- 50 samples (2.5 s), stride 25. No window across a gap over 0.5 s, a subject change or an activity change. Single-label windows only.

## D-005: Classes
- 5 classes: walking, upstairs, downstairs, sitting, standing.

## D-006: Domains
- Core: UCI-HAR, WISDM, PAMAP2-hand. Extra: PAMAP2-chest and PAMAP2-ankle (sources with the hand config, and targets).

## D-007: Normalization
- Per-channel standardization with source-train mean/std, gravity kept. Per-window z-score is an optional ablation only.

## D-008: Splits
- 3 fixed subject-wise splits per domain; seed k uses split k. PAMAP2 test sets are disjoint so every subject is tested exactly once (the split script asserts this).

## D-009: Models and size
- CNN, TCN, Tiny Transformer. About 75k params each, all within 10% of each other. Re-check after tuning.

## D-010: Training and tuning
- Adam, batch 128, max 30 epochs, early stop (patience 5) on source-val macro F1, class-weighted loss.
- Tuning: 8 random configs per model per core source, split 0, seed 0, max 15 epochs.

## D-011: Final runs and evaluation
- Seeds 0, 1, 2. Within-dataset test on source test subjects; cross-dataset test on all windows of the target with no retraining.
- Drop = within minus cross (accuracy and macro F1). Headline metric: macro F1.

## D-012: Results format
- One CSV row per run: model, source, target, seed, split, n_windows, n_params, acc, macro_f1, weighted_f1, kappa. Never screenshots.

## D-013: Compute budget
- 117 training runs (72 tuning + 45 final) and 225 evaluations in total.
