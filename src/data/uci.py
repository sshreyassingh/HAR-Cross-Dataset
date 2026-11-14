"""UCI-HAR preprocessing, following PROTOCOL.md (sprint version).

Raw input (units already g, 50 Hz, 128-sample windows with 50% overlap):
    <raw_dir>/UCI HAR Dataset/{train,test}/Inertial Signals/total_acc_{x,y,z}_{split}.txt
    <raw_dir>/UCI HAR Dataset/{train,test}/y_{split}.txt         (1 walking ... 6 laying)
    <raw_dir>/UCI HAR Dataset/{train,test}/subject_{split}.txt

Output: data/processed/uci.npz
    X          (N, 50, 3) float32, RAW g (NOT normalized; normalization happens in the harness)
    y          (N,) int64, 0 walking, 1 upstairs, 2 downstairs, 3 sitting, 4 standing (laying dropped)
    subject_id (N,) int64
    meta       JSON string. Read it with: json.loads(str(np.load(path)["meta"]))

Two methods:
    stitch  (preferred) rebuild continuous per-subject signals from the overlapping windows,
            anti-alias filter + resample 50 -> 20 Hz, then cut 50-sample windows with stride 25.
    window  (fallback)  resample each 128-sample window to 20 Hz and crop to 50 samples.
            Log it in docs/decision_log.md if you use it.

Run:  python -m src.data.uci --raw data/raw/uci_har --out data/processed/uci.npz
"""
import argparse
import datetime
import json
import os
import subprocess

import numpy as np
from scipy.signal import resample_poly

FS_IN, FS_OUT = 50, 20
UP, DOWN = 2, 5            # 50 Hz * 2 / 5 = 20 Hz (polyphase FIR, anti-alias filter included)
WIN, STRIDE = 50, 25       # 2.5 s windows, 50% overlap
HALF = 64                  # UCI windows are 128 samples and overlap by 64
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}   # 6 (laying) is dropped
CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def load_split(raw_dir, split):
    base = os.path.join(raw_dir, "UCI HAR Dataset", split)
    X = np.stack(
        [np.loadtxt(os.path.join(base, "Inertial Signals", f"total_acc_{a}_{split}.txt")) for a in "xyz"],
        axis=-1,
    ).astype(np.float64)                                         # (N, 128, 3)
    y = np.loadtxt(os.path.join(base, f"y_{split}.txt")).astype(int)
    s = np.loadtxt(os.path.join(base, f"subject_{split}.txt")).astype(int)
    return X, y, s


def find_segments(X, y, s, atol=1e-6):
    """Chain consecutive rows into continuous segments.

    Row i continues into row i+1 when both have the same subject and label AND the second half of row i
    equals the first half of row i+1 (50% overlap). Returns a list of (start, end_exclusive).
    """
    n = len(X)
    if n == 1:
        return [(0, 1)]
    same = (s[:-1] == s[1:]) & (y[:-1] == y[1:])
    overlap = np.all(np.abs(X[:-1, HALF:, :] - X[1:, :HALF, :]) <= atol, axis=(1, 2))
    link = same & overlap
    segs, start = [], 0
    for i in range(n - 1):
        if not link[i]:
            segs.append((start, i + 1))
            start = i + 1
    segs.append((start, n))
    return segs


def stitch(X, start, end):
    """Rebuild one continuous signal (L, 3) from windows X[start:end] that overlap by 64 samples."""
    w = X[start:end]
    if len(w) == 1:
        return w[0]
    return np.concatenate([w[:-1, :HALF, :].reshape(-1, 3), w[-1]], axis=0)


def to_20hz(sig, axis=0):
    """Polyphase FIR resampling 50 -> 20 Hz. The low-pass (anti-alias) filter is part of the method."""
    return resample_poly(sig, UP, DOWN, axis=axis, padtype="line")


def make_windows(sig):
    """Cut (L, 3) into (n, 50, 3) windows with stride 25. Never crosses the end of the segment."""
    if len(sig) < WIN:
        return np.empty((0, WIN, 3))
    n = (len(sig) - WIN) // STRIDE + 1
    idx = np.arange(WIN)[None, :] + STRIDE * np.arange(n)[:, None]
    return sig[idx]


def build_from_arrays(X, y, s, method="stitch"):
    """X (N,128,3) raw g windows, y labels 1..6, s subject ids (rows in recording order).

    Returns X_out (M,50,3) float32, y_out (M,), s_out (M,), info dict.
    """
    info = {"method": method, "n_rows_in": int(len(X))}
    if method == "stitch":
        segs = find_segments(X, y, s)
        info["n_segments"] = len(segs)
        info["n_links"] = int(len(X) - len(segs))      # consecutive rows that chained (each segment of k rows has k-1 links)
        info["mean_windows_per_segment"] = float(len(X) / len(segs))
        xs, ys, ss = [], [], []
        for a, b in segs:
            label = y[a]
            if label not in LABEL_MAP:
                continue
            w = make_windows(to_20hz(stitch(X, a, b)))
            if len(w):
                xs.append(w)
                ys.append(np.full(len(w), LABEL_MAP[label]))
                ss.append(np.full(len(w), s[a]))
        if not xs:
            raise ValueError("No windows produced. Check the raw data path and files.")
        return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64),
                np.concatenate(ss).astype(np.int64), info)
    if method == "window":
        keep = np.isin(y, list(LABEL_MAP))
        r = to_20hz(X[keep], axis=1)                         # (n, 52, 3)
        r = r[:, 1:1 + WIN, :]                               # centre-crop to 50 samples
        y_out = np.array([LABEL_MAP[v] for v in y[keep]], dtype=np.int64)
        return r.astype(np.float32), y_out, s[keep].astype(np.int64), info
    raise ValueError("method must be 'stitch' or 'window'")


def build(raw_dir, method="stitch"):
    """Load train + test files (together they hold all 30 subjects) and process them."""
    xs, ys, ss, infos = [], [], [], []
    for split in ("train", "test"):
        X, y, s = load_split(raw_dir, split)
        a, b, c, info = build_from_arrays(X, y, s, method)
        xs.append(a); ys.append(b); ss.append(c); infos.append(info)
    info = {"method": method, "per_split": infos}
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), info


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def summary(X, y, s, info):
    print(f"windows: {X.shape}  dtype={X.dtype}  subjects: {len(np.unique(s))}")
    if info["method"] == "stitch":
        for name, i in zip(("train", "test"), info["per_split"]):
            print(f"{name:5s} rows={i['n_rows_in']:5d}  segments={i['n_segments']:4d}  links={i['n_links']:5d}  "
                  f"mean windows/segment={i['mean_windows_per_segment']:.1f}")
    else:
        print("segments: n/a (window method)")
    mag = np.linalg.norm(X, axis=-1).mean(axis=1)
    print("\nclass counts and mean |a| (g):")
    for k, name in enumerate(CLASS_NAMES):
        m = y == k
        print(f"  {k} {name:11s} n={m.sum():6d}  subjects={len(np.unique(s[m])):2d}  mean|a|={mag[m].mean():.3f}")
    print("\nNaN count:", int(np.isnan(X).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/uci_har")
    ap.add_argument("--out", default="data/processed/uci.npz")
    ap.add_argument("--method", choices=["stitch", "window"], default="stitch")
    ap.add_argument("--sha256", default="", help="SHA-256 of the downloaded zip, stored in meta")
    args = ap.parse_args()

    if not args.sha256:
        print("WARNING: --sha256 not given, so meta will hold an empty checksum. Take it from docs/data_checksums.txt.")
    X, y, s, info = build(args.raw, args.method)
    summary(X, y, s, info)
    if args.method == "stitch":
        mean_w = np.mean([i["mean_windows_per_segment"] for i in info["per_split"]])
        if mean_w < 5:
            print("\nWARNING: rows do not chain into long segments (mean windows per segment = %.1f)." % mean_w)
            print("The rows are probably not in recording order. Use --method window and log it.")

    meta = {
        "dataset": "UCI-HAR", "source_url": SOURCE_URL, "raw_zip_sha256": args.sha256,
        "placement": "waist", "units": "g", "fs_original": FS_IN, "fs": FS_OUT,
        "window": WIN, "stride": STRIDE, "method": args.method, "classes": CLASS_NAMES,
        "dropped": ["laying"], "normalized": False, "info": info,
        "git_commit": git_commit(), "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, subject_id=s, meta=np.array(json.dumps(meta)))
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()