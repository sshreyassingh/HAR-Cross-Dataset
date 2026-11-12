# PROTOCOL.md (sprint version, tag: protocol-v1-sprint)

Project: Investigating the Causes of Cross-Dataset Performance Degradation in Human Activity Recognition Systems
Team: Shreya Singh (A), Shanaya Dhamija (B), Sharanya Gupta (C), Sanjana Prasad (D)

This file is FIXED once all four members have signed it. A change needs a dated entry in `docs/decision_log.md` that all four agree to. Every AI assistant used on this project must be given this file and must not change any number or rule in it.

## 0. Why we are doing this

Objectives (from the synopsis):
1. Characterize dataset-level differences between UCI-HAR, WISDM and PAMAP2 (placement, sampling rate, activity definitions, subjects, signal statistics).
2. Quantify cross-dataset performance degradation of lightweight models (CNN, TCN, Tiny Transformer) versus within-dataset performance.
3. Investigate the causes of the degradation (statistics, PCA/t-SNE, ablations) and check whether the lightweight architectures differ in robustness.

Working question: why do HAR models that work well inside one dataset fail on another, and do the architectures differ in how much they degrade?

## 1. Sprint scope

IN: 3 domains (UCI-HAR, WISDM, PAMAP2-hand), 3 models, 3 seeds, 6 directed pairs, shift statistics, PCA/t-SNE, 3 inference-only ablations, descriptive correlation analysis.

STRETCH (only if every core item is finished by the end of Day 5): PAMAP2-chest and PAMAP2-ankle as extra domains (placement study).

OUT for now: leave-one-subject-out, full tuning, retrain-based ablations, few-shot fine-tuning, explainability maps, embedding divergence, mixed-effects statistics, full report.

Simplifications we will state honestly in any presentation or report: tuning on UCI-HAR only (config reused for the other sources), 3 seeds, 3 domains, and PAMAP2 results that are high-variance because each split has only 5 training subjects and a 1-subject validation set.

## 2. Fixed numbers

| Item | Decision |
|---|---|
| Signal | 3-axis accelerometer only, units **g** (divide m/s^2 by 9.80665), anti-alias filtered (polyphase FIR), resampled to **20 Hz** |
| Windows | 50 samples (2.5 s), stride 25 (50% overlap). Never across a gap over 0.5 s, a subject change or an activity change. Single-label windows only |
| Normalization | Per-channel standardization using **source-train** mean and std only, gravity kept. Stats are saved inside every checkpoint and applied unchanged to any target. **Pipelines save raw g, not normalized data** |
| Classes | 0 walking, 1 upstairs, 2 downstairs, 3 sitting, 4 standing (the only 5 classes shared by all three datasets; no jogging, no laying) |
| Domains | Core: UCI-HAR (waist), WISDM (pocket), PAMAP2-hand. Stretch: PAMAP2-chest, PAMAP2-ankle |
| Data file | `data/processed/{uci,wisdm,pamap2_hand}.npz` with `X (N,50,3) float32`, `y (N,)`, `subject_id (N,)`, `meta` |
| Splits | 3 fixed subject-wise splits per domain in `splits/{domain}.json`, generated once with seed 0. Seed k uses split k. UCI 21/4/5, WISDM 25/5/6, PAMAP2 5/1/3 (train/val/test subjects). The 3 PAMAP2 test sets are disjoint and cover all 9 subjects exactly once |
| Model I/O | input `(B,50,3)` to logits `(B,5)`. Each model permutes internally if needed |
| Model size | About 75k parameters each. The binding rule is parity: after tuning, all three models are re-checked and must be within **10%** of each other (max/min <= 1.10); 60k to 90k is only a sanity range. Tuned depth and kernel size change the count, so if parity fails, re-solve the width before any final run |
| Training | Adam, batch 128, max 30 epochs, early stop (patience 5) on source-val macro F1, class-weighted cross-entropy |
| Tuning | 4 random configs per model, on **UCI-HAR only**, split 0, seed 0, max 10 epochs, chosen by source-val macro F1. The best config is reused for WISDM and PAMAP2-hand. Search: lr {3e-4, 1e-3, 3e-3}, dropout {0.1, 0.3, 0.5}, weight decay {0, 1e-4, 1e-3}, kernel size {3, 5, 7} (CNN, TCN) or heads {2, 4} (Transformer). Width is solved from the parameter budget, not searched |
| Final runs | Best config, seeds 0, 1, 2 (seed k uses split k), for each of the 3 sources |
| Within-dataset test | Test subjects of the source split |
| Cross-dataset test | All windows of the target dataset, no retraining, no target statistics, source normalization applied |
| Drop | Within-dataset score minus cross-dataset score, for accuracy and macro F1 |
| Headline metric | Macro F1 (WISDM is imbalanced). Accuracy, weighted F1 and Cohen's kappa reported too |
| Reference floor | Majority-class and random scores computed from label distributions |
| Results CSV | columns: `model, source, target, seed, split, n_windows, n_params, acc, macro_f1, weighted_f1, kappa` (one row per run, never screenshots) |
| Run budget | Per model: 4 tuning runs, 9 final runs, 27 result rows (9 within + 18 cross). Total: 12 tuning runs, 27 final runs, 81 rows |

## 3. Ablations (inference only, seed 0, split 0, the 6 core pairs)

Done with `src/analysis/perturb.py` written by D. Owners must not reimplement it. Each ablation includes an **unperturbed control row** that must match the final result for the same seed.

1. Unit mismatch: multiply target windows by 9.80665 (as if m/s^2 data were given to a g-trained model).
2. Time-stretch of the effective sampling rate: factors 0.75 and 1.25 (resample the 50-sample window to round(50*f) samples, then crop or edge-pad to 50).
3. Random 3D rotation of target windows (fixed seed).

Results go to `results/raw/ablation_{model}.csv` with extra columns `ablation` and `level`.

## 4. Shift analysis (D)

- Per domain and activity: mean, variance, skewness, kurtosis per axis, dominant FFT frequency, spectral energy, signal magnitude area, mean gravity vector.
- Per directed pair (6 pairs): JSD and KL on binned feature distributions (fixed bins from the pooled range), MMD (RBF, median heuristic) on standardized handcrafted features, proxy A-distance (domain classifier), class-distribution JSD.
- Metadata per pair: placement, original sampling rate, units, gravity and filtering status, subject counts.
- PCA and t-SNE on handcrafted features, coloured by domain and by activity, with silhouette score by domain versus by activity.
- Statistics: descriptive. Spearman correlation between the macro-F1 drop and each shift measure across 6 pairs x 3 models = 18 points, reported as **exploratory**, with no significance claims.

## 5. Amendments (agreed before any code is written)

1. Pipelines save un-normalized g values. Normalization happens only in the harness.
2. If the stretch placement domains are built, PAMAP2-chest and -ankle must keep identical window indices and `subject_id` arrays to PAMAP2-hand, and pairs inside PAMAP2 are evaluated only on the source split's test subjects.
3. Tuning runs go to `results/raw/tuning_{model}.csv`. Final runs go to `results/raw/{model}_results.csv`. Ablations go to `results/raw/ablation_{model}.csv`.
4. No analysis is chosen after looking at results. Planned analyses are listed in section 4.
5. UCI-HAR window handling. Preferred: rebuild continuous per-subject signals from the overlapping windows, then window at 20 Hz. Allowed fallback if that takes over half a day: resample each 128-sample window to 20 Hz and crop to 50 samples. If the fallback is used, log it in `docs/decision_log.md` with the date, and state it in the data card and in every slide or report (windows are resampled individually, which slightly changes signal statistics and window overlap, so UCI shift statistics carry that caveat). Whichever method is chosen is used for all three models.

## 6. Rules that never change

- Hyperparameters are tuned on source validation only. Target data is never used for tuning, normalization or any choice.
- No subject appears in more than one of train, validation and test within a domain.
- Every reported number is a mean and standard deviation over 3 seeds, produced by a script that writes the results CSV. No numbers typed in by hand or taken from a chat.
- Nobody edits `PROTOCOL.md`, `splits/*.json` or `src/train/` from their own branch. Raise it with the team, log it, then change it.
- No data files, `.npz` files, checkpoints, tokens or keys are committed to the repo.
- Results are called **preliminary** in any presentation until the full protocol has been run.
- Nobody writes or says "significant" about a result. Use "larger", "smaller" or "exploratory". Correlations and model comparisons are exploratory only.
- PAMAP2 results are reported per seed and split as well as mean and standard deviation, because the 5/1/3 split has high variance.

## 7. Ownership

| Member | Owns |
|---|---|
| Shreya (A) | UCI-HAR pipeline, CNN, repo scaffold and decision log, presentation integration |
| Shanaya (B) | WISDM pipeline, TCN, bibliography (8 to 10 verified papers) |
| Sharanya (C) | PAMAP2 pipeline, Tiny Transformer, efficiency table (params, FLOPs, size, latency) |
| Sanjana (D) | Splits, training harness, data audit, parity check, shift analysis, perturbation code, master results and figures |

## 8. Sign-off

| Member | Signed (name, date) |
|---|---|
| Shreya Singh | |
| Shanaya Dhamija | |
| Sharanya Gupta | |
| Sanjana Prasad | |
