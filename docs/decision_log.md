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
- Best config is reused for WISDM and PAMAP2-hand