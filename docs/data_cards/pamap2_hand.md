# Data card: PAMAP2-hand

**Status:** draft, 2026-11-15 (Sharanya). Subject 109 decision recorded under Known limitations.

## Source
- PAMAP2 Physical Activity Monitoring, UCI Machine Learning Repository, dataset ID 231.
- URL: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- DOI 10.24432/C5NW2H. Donor: Attila Reiss.
- Licence: CC BY 4.0 per the UCI page. The dataset readme states the data is "freely available for academic research" with no legal constraints.
- Version: the readme gives no version number; the UCI page lists the dataset as donated in 2012 (UCI ID 231).
- Downloaded 2026-11-15 (`pamap2.zip`, 688,226,940 bytes).
- SHA-256:
  - `pamap2.zip`: 76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352
  - inner `PAMAP2_Dataset.zip`: 6c2282b5b52775cd0588d450d10a315260c94fa3228b9fe453f48c47fda106f4
  - `readme.pdf`: 99a0339f1d754089fc2f7f2c24f5a47888a51dc0646a8b210af60358c00da30d

## Raw data (verified by scripts/pamap2_inspect.py, output in docs/pamap2_audit.txt)
- 54 columns per row, 100 Hz in every file (median step 0.010 s, maximum step 0.010 s, no timestamp gaps or reversals).
- Protocol/: subject101-109 (9 files). Optional/: 101, 105, 106, 108, 109. Only Protocol/ is used.
- Sensor: hand IMU (placed over the wrist of the dominant arm), +/-16 g accelerometer, columns 5-7 (1-based), units m/s^2. The readme recommends this accelerometer over the +/-6 g one, which is not precisely calibrated and saturates.
- At rest the magnitude is 9.76 (sitting) and 9.78 (standing) m/s^2, i.e. about 1.00 g.
- Observed range: -145.4 to 157.8 m/s^2. The maximum is slightly above the nominal 156.9 m/s^2 (16 g). Values were not clipped.
- Missing values: 0.8% of the rows of our 5 classes (6,680 of 836,040). Longest NaN run: 28 samples (0.28 s).

## Classes (PAMAP2 activity ID -> protocol class)
walking 4 -> 0; ascending stairs 12 -> 1; descending stairs 13 -> 2; sitting 2 -> 3; standing 3 -> 4.
Activity 0 (transient) and every other activity are dropped. Mapping source: activity IDs in readme.pdf, section II.2 (checked 2026-11-15).

## Preprocessing (src/data/pamap2.py)
1. Cut each file into single-activity segments (never across an activity change, a timestamp gap > 0.5 s or a NaN run > 0.5 s).
2. NaN runs of up to 50 samples inside a segment: linear interpolation. NaN at segment edges: trimmed.
3. m/s^2 divided by 9.80665 to give g.
4. Polyphase FIR resampling 100 -> 20 Hz (anti-alias filter included).
5. Windows of 50 samples, stride 25, inside each segment. Values are raw g, not normalized.

## Result (data/processed/pamap2_hand.npz, not in git)
- X (6602, 50, 3) float32; y 0-4; subject_id 101-108.
- Cleaning: 57 segments; 4,178 NaN runs interpolated (6,661 samples); 19 edge samples trimmed; no segment cut.

| Subject | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---|---|---|---|---|---|---|---|---|
| Windows | 775 | 893 | 821 | 894 | 853 | 770 | 798 | 798 | 0 |

| Class | Windows | Subjects | Mean abs a (g) |
|---|---|---|---|
| 0 walking | 1899 | 8 | 1.149 |
| 1 upstairs | 913 | 8 | 1.089 |
| 2 downstairs | 813 | 8 | 1.052 |
| 3 sitting | 1469 | 8 | 0.995 |
| 4 standing | 1508 | 8 | 0.997 |

## Tests
`python -m pytest -q -s tests/test_pamap2.py`: 16 passed (12 synthetic, 4 on the real file), 2026-11-15.

## Known limitations
- Subject 109 has no windows for the 5 classes in Protocol or Optional, so only 8 subjects are usable. The readme explains that hardware problems left some activities of some subjects partly or completely missing. The protocol assumed 9 subjects. Team decision (Option A, 2026-11-16): three disjoint test sets of 3, 3 and 2 subjects that cover all 8 subjects once; train/val/test 4/1/3, 4/1/3 and 5/1/2.
- The hand IMU's effective rate is 99.63 Hz (readme); the files are on a regular 0.01 s grid with NaN for dropped samples, and the pipeline treats it as 100 Hz.
- Different parts of one subject's recording are joined during transient (activity 0) stretches (readme). Segments are cut at every activity change and activity 0 is dropped, so no window crosses such a join.
- One placement only (hand, dominant wrist); accelerometer only.
- Downstairs is the smallest class (813 windows).
- Filled NaN values are interpolated, not measured.