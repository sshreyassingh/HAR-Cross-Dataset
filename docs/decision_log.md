# Decision Log

One entry per decision. Newest at the bottom.  
Source of truth: `PROTOCOL.md` (tag: protocol-v1-sprint).

---

## D-001: Protocol locked
- Date: 2026-11-12
- Decision: PROTOCOL.md (protocol-v1-sprint) is fixed. Any change requires team agreement + new dated entry here.
- Owner: All

## D-002: Branching and PRs
- Date: 2026-11-12
- Decision: Each member works on their own branch. PRs into main. Smoke test must pass.
- Owner: All

## D-003: Raw data never committed
- Date: 2026-11-12
- Decision: `data/raw/` and `data/processed/` are git-ignored. Only `splits/*.json` and small result CSVs are tracked.
- Owner: All

## D-004: Signal & preprocessing (locked by protocol)
- 3-axis accelerometer only, units in **g**
- Anti-alias filter + resample to **20 Hz**
- Windows: 50 samples (2.5 s), stride 25
- Never across gap > 0.5 s, subject change, or activity change
- Classes: 0 walking, 1 upstairs, 2 downstairs, 3 sitting, 4 standing

## D-005: Normalization
- Pipelines save **raw g** only.
- Normalization (source-train mean/std) happens only inside the harness and is saved in the checkpoint.

## D-006: Tuning rule
- 4 random configs per model
- Only on UCI-HAR, split 0, seed 0, max 10 epochs

- Dataset contains 8,355 windows from 30 subjects.
- Input shape: `(8355, 50, 3)`.
- Sampling frequency: 20 Hz.
- Window size: 50 samples.
- Units: g.
- Data is not normalized.
- Preprocessing method: continuous signal stitching.

2026-11-15 (Shreya): UCI-HAR uses the preferred stitch method (amendment 5), not the fallback. Train 7352 rows chained into 600 segments, test 2947 rows into 257. Windows never cross a gap in the overlap. Output: 8355 windows, 30 subjects. See `docs/data_cards/uci_har.md`.

- Best config is reused for WISDM and PAMAP2-hand

2026-11-15 (Shreya): UCI-HAR uses the preferred stitch method (amendment 5), not the fallback. Train 7352 rows chained into 600 segments, test 2947 rows into 257. Windows never cross a gap in the overlap. Output: 8355 windows, 30 subjects. See docs/data_cards/uci_har.md.

- Best config is reused for WISDM and PAMAP2-hand

## WISDM Dataset

- Dataset file: `wisdm.npz`
- SHA-256: `25aa9ab426c0f1f4c662566059321e5a3c5c3d1772c8a4143882c2b65b5c7ac0`
- Dataset contains 29,553 windows from 36 subjects.
- Input shape: `(29553, 50, 3)`.
- Sampling frequency: 20 Hz.
- Window size: 50 samples.
- Units: g.
- Data is not normalized.
- Preprocessing: interpolation within segments; jogging dropped.
- All five activity classes are present.
- Generated `splits/wisdm.json` using seed 0 and real subject IDs.
- All three splits passed subject-count, disjointness and class-presence checks.
- Metadata currently records `git_commit: unknown`. Awaiting confirmation from the dataset owner.

## UCI-HAR Dataset

- Dataset contains 8,355 windows from 30 subjects.
- Input shape: `(8355, 50, 3)`.
- Sampling frequency: 20 Hz.
- Window size: 50 samples.
- Units: g.
- Data is not normalized.
- Preprocessing method: continuous signal stitching.

2026-11-15 (Shreya): UCI-HAR uses the preferred stitch method (amendment 5), not the fallback. Train 7352 rows chained into 600 segments, test 2947 rows into 257. Windows never cross a gap in the overlap. Output: 8355 windows, 30 subjects. See `docs/data_cards/uci_har.md`.

## 2026-11-16 - CNN tuning result (Shreya)
- Tuning run by Sanjana on Colab (`python -m scripts.tune_cnn_uci`, commit 562f15f8adbb1751b01ffa0ca54bdeb16ff44b4e, on main; splits/uci.json git hash 6524095acb690d56912ecfdd1a492b4a1e0490c8). Protocol settings: 4 unique random configs (seed 0) on UCI-HAR, split 0, max 10 epochs, chosen by source-val macro F1. Results in results/raw/tuning_cnn.csv.
- Selected: lr 3e-3, dropout 0.3, weight decay 1e-4, kernel 3, 3 blocks, width 90, 75,335 params (val macro F1 0.937). Saved in configs/cnn_best.json. Reused unchanged for WISDM and PAMAP2-hand.
- Limitation: all 4 drawn configs had weight decay 1e-4 (about 3% chance; Sanjana verified the sampler is correct), so weight decay was not compared for the CNN. We are not rerunning, to keep the protocol's seed-0 draw. Any extra exploration will be a separate, labelled experiment.
- Limitation: the four scores range from 0.930 to 0.937, a gap of under 0.01. With one seed we cannot tell whether that gap is real, so the config was selected by validation F1, not shown to be better. Trial 1 peaked at epoch 9 of 10, so the 10-epoch cap may have limited it.
- README tuning numbers (8 configs, 15 epochs) were stale. PROTOCOL.md (4 configs, UCI only, 10 epochs) is the source of truth, and the README now points to it. No protocol change.

2026-11-16 (Shreya): .gitignore line `results/raw/*` was blocking result CSVs. Added `!results/raw/*.csv`, since PROTOCOL.md requires results CSVs in the repo. Other files in results/raw/ remain ignored.
2026-11-16 (Shreya): .gitignore: allow results/tables/*.csv, since the protocol requires result CSVs in the repo.

## WISDM pipeline decisions (Shanaya)

- Date: 2026-11-16
- Release: WISDM v1.1 (2010). 36 users, phone in pocket, about 20 Hz. The 36 users fit the protocol's 25/5/6 split.
- Dropped: malformed lines (11,734), rows with timestamp 0, duplicate rows (same user, activity, timestamp), and Jogging.
- Timestamps were sorted within each user and activity. A new segment starts after any gap over 0.5 s.
- No anti-alias filter: the data is already about 20 Hz, so it is only interpolated onto a uniform 20 Hz grid.
- Converted to g (divide by 9.80665), not normalized. Code: `src/data/wisdm_preprocess.py`.
- Reply to the `git_commit: unknown` note above: wisdm.npz was built from a local folder that is not a git repository, so no commit could be recorded. The script that produces it is `src/data/wisdm_preprocess.py`.
- Owner: Shanaya

## TCN tuning result (Shanaya)

- Date: 2026-11-16
- UCI-HAR, split 0, seed 0, 4 random configs, max 10 epochs (see D-006).
- Best config: lr 0.001, dropout 0.5, weight decay 1e-4, kernel 7. Width 39, 76,640 parameters. Validation macro F1 0.9306.
- The top two trials were nearly tied (0.9305 and 0.9306), so the choice is preliminary.
- Parameter count sent to Sanjana for the parity check. Against the CNN's 75,335 the ratio is about 1.02.
- Owner: Shanaya


2026-11-15 (Sharanya): PAMAP2-hand uses the hand IMU +/-16 g accelerometer (columns 5-7). Class map from the readme activity list (II.2): walking=4, upstairs=12, downstairs=13, sitting=2, standing=3; activity 0 and all other activities dropped. Only the Protocol folder is read; the Optional folder holds none of the 5 classes.

2026-11-15 (Sharanya): PAMAP2 NaN rule: runs of up to 50 samples (0.5 s at 100 Hz) are filled by linear interpolation inside a single-activity segment; longer runs cut the segment; NaN at segment edges is trimmed. In the real data the longest run is 28 samples, so nothing is cut. Alternative considered: drop every window that contains a NaN.

2026-11-15 (Sharanya): PAMAP2 subject 109 has no windows for the 5 classes (Protocol: rope jumping and transient only; Optional: TV, computer work, car driving, folding laundry, house cleaning, soccer). Usable subjects: 101-108.

2026-11-16 (Sharanya): PAMAP2 split follow-up: the team agreed Sanjana's Option A for 8 usable subjects: three disjoint test sets of 3, 3 and 2 subjects covering all 8 once; train/val/test 4/1/3, 4/1/3 and 5/1/2. PROTOCOL.md still says 5/1/3 and 9 subjects; this log is the record.

2026-11-16 (Sharanya): Tiny Transformer design (choices the protocol does not fix): patch size 5 (10 tokens), 2 encoder layers, learned positional encoding, mean pooling, feed-forward width 2 x d_model, GELU, pre-norm layers with a final LayerNorm. d_model is solved so the real parameter count is nearest 75,000: 2 heads d_model 66 = 73,331 parameters; 4 heads d_model 68 = 77,729 parameters. Shape, parameter-count and gradient tests pass; the overfit test is pending the shared trainer.

2026-11-16 (Sharanya): Tiny Transformer tuning (UCI-HAR, split 0, seed 0, 4 configs, max 10 epochs, shared tuner): best lr 0.001, dropout 0.3, weight decay 0.0001, 4 heads; d_model 68, 77,729 parameters; validation macro-F1 0.9109 (other trials 0.8823, 0.9024, 0.8931). A rerun with the same seed reproduced the CSV and the best config exactly. The selection is preliminary: validation macro-F1 varies by up to about 4 points between epochs. Overfit test on 64 windows with the shared trainer: training accuracy 1.000. Files: results/raw/tuning_transformer.csv, configs/best/transformer.yaml, scripts/tune_transformer.py.

2026-11-16 (Sharanya): Model name settled as tiny_transformer (per Sanjana). Tuning outputs renamed: results/raw/tuning_tiny_transformer.csv, configs/best/tiny_transformer.yaml, docs/tiny_transformer_tuning_summary.txt; the earlier files named transformer were removed. The rerun with the same seed reproduced the same four trials and best config (lr 0.001, dropout 0.3, weight decay 0.0001, 4 heads, d_model 68, 77,729 parameters). Best-config location: configs/best/<model>.yaml, as in the sprint plan.


2026-11-16 (Sharanya): Efficiency table. FLOPs are counted with torch FlopCounterMode on one (1,50,3) window, 2 FLOPs per multiply-add, conv/matmul/attention only (BN, LayerNorm, activations, softmax not counted). A first run gave 1,500,760 FLOPs for the Tiny Transformer; the script cross-check (hand count 1,555,160) showed a gap of 54,400, which is the attention matmuls (QK^T and AV, 2 layers) that the default CPU attention kernel hides from the counter. Fix: count with the math attention backend (sdpa_kernel(SDPBackend.MATH)); the counter then equals the hand count (0.0% apart). Latency still uses the default backend. CNN and TCN FLOPs were unchanged by the fix. Latency is batch 1, 1 thread, shared 2-core CPU; medians varied by about 6% or less across four runs, p95 varies more, so median is the headline.
