"""WISDM v1.1 -> data/processed/wisdm.npz  (raw g, 20 Hz, 50-sample windows, stride 25).
Usage: python wisdm_preprocess.py RAW_TXT OUT_NPZ
"""
import sys, subprocess
import numpy as np, pandas as pd

G = 9.80665                      # m/s^2 per g (protocol)
FS = 20.0                        # target rate, Hz
WIN, STRIDE = 50, 25             # protocol
MAX_GAP = 0.5                    # seconds; break segments at gaps larger than this
# protocol classes; Jogging is dropped
CLASSES = {"Walking": 0, "Upstairs": 1, "Downstairs": 2, "Sitting": 3, "Standing": 4}

def load(path):
    rows, bad = [], 0
    for line in open(path):
        for rec in line.strip().split(";"):      # some lines hold several records
            rec = rec.strip()
            if not rec: continue
            p = rec.split(",")
            try:
                if len(p) != 6: raise ValueError
                rows.append((int(p[0]), p[1], int(p[2]), float(p[3]), float(p[4]), float(p[5])))
            except ValueError:
                bad += 1                         # malformed / missing value -> drop
    df = pd.DataFrame(rows, columns=["user", "activity", "ts", "x", "y", "z"])
    return df, bad

def clean(df):
    n0 = len(df)
    df = df[df.ts > 0]                                            # junk timestamps
    df = df[df.activity.isin(CLASSES)]                            # drop Jogging
    df = df.drop_duplicates(["user", "activity", "ts"])           # duplicate samples
    df = df.sort_values(["user", "activity", "ts"]).reset_index(drop=True)
    df["t"] = df.ts / 1e9                                         # ns -> s
    for c in "xyz": df[c] = df[c] / G                             # m/s^2 -> g
    return df, n0 - len(df)

def windows_from_segment(t, xyz):
    """Interpolate one continuous segment onto a 20 Hz grid, then cut windows."""
    grid = np.arange(t[0], t[-1], 1.0 / FS)
    if len(grid) < WIN: return None
    sig = np.stack([np.interp(grid, t, xyz[:, k]) for k in range(3)], axis=1)
    starts = range(0, len(sig) - WIN + 1, STRIDE)
    return np.stack([sig[s:s + WIN] for s in starts])

def build(df):
    X, y, sid = [], [], []
    for (user, act), d in df.groupby(["user", "activity"]):       # never mixes subjects/activities
        t, xyz = d.t.values, d[["x", "y", "z"]].values
        cuts = np.where(np.diff(t) > MAX_GAP)[0] + 1              # gaps > 0.5 s
        for tt, xx in zip(np.split(t, cuts), np.split(xyz, cuts)):
            w = windows_from_segment(tt, xx)
            if w is None: continue
            X.append(w); y += [CLASSES[act]] * len(w); sid += [user] * len(w)
    return np.concatenate(X).astype(np.float32), np.array(y, np.int64), np.array(sid, np.int64)

if __name__ == "__main__":
    raw, out = sys.argv[1], sys.argv[2]
    df, bad = load(raw)
    df, dropped = clean(df)
    X, y, sid = build(df)
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "unknown"
    meta = dict(dataset="WISDM", version="v1.1 (2010)", original_rate_hz=20, placement="pocket",
                units="g", git_commit=commit)
    np.savez(out, X=X, y=y, subject_id=sid, meta=np.array(str(meta)))
    print(f"malformed lines dropped: {bad}; rows removed in cleaning: {dropped}")
    print("X", X.shape, X.dtype, "| users", len(np.unique(sid)))
    print("class counts:", dict(zip(*np.unique(y, return_counts=True))))
