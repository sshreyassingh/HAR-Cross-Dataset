"""PAMAP2-hand preprocessing, following PROTOCOL.md (sprint version).

Raw input (one space-separated .dat file per subject, 54 columns, 100 Hz, missing = NaN):
    <raw_dir>/**/Protocol/subject{101..109}.dat
Columns used (0-based): 0 timestamp (s), 1 activityID, 4-6 hand IMU +/-16 g accelerometer (m/s^2).
The Optional/ folder is not read: its files contain none of our 5 classes (checked, see docs/pamap2_audit.txt
and docs/data_cards/pamap2_hand.md).

Output: data/processed/pamap2_hand.npz
    X          (N, 50, 3) float32, RAW g (NOT normalized; normalization happens in the harness)
    y          (N,) int64, 0 walking, 1 upstairs, 2 downstairs, 3 sitting, 4 standing
    subject_id (N,) int64, the PAMAP2 subject number (101..109)
    meta       JSON string. Read it with: json.loads(str(np.load(path)["meta"]))

Steps per file: cut into single-activity segments (never across an activity change, a timestamp gap > 0.5 s,
or a NaN run > 0.5 s) -> fill short NaN runs by linear interpolation -> m/s^2 to g -> anti-alias filter +
resample 100 -> 20 Hz (polyphase FIR) -> cut 50-sample windows with stride 25 inside each segment.

Run:  python -m src.data.pamap2 --raw data/raw/pamap2 --out data/processed/pamap2_hand.npz --sha256 <zip hash>
"""
import argparse
import datetime
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

G = 9.80665                       # m/s^2 per g (protocol)
FS_IN, FS_OUT = 100, 20
UP, DOWN = 1, 5                   # 100 Hz * 1 / 5 = 20 Hz (polyphase FIR, anti-alias filter included)
WIN, STRIDE = 50, 25              # 2.5 s windows, 50% overlap
MAX_GAP_S = 0.5                   # protocol: never across a gap over 0.5 s
MAX_NAN_RUN = int(MAX_GAP_S * FS_IN)   # 50 raw samples: NaN runs up to this are filled, longer ones cut the segment
N_COLS = 54
HAND_ACC16 = [4, 5, 6]            # hand IMU, +/-16 g accelerometer, m/s^2
# PAMAP2 activityID -> protocol class. Source: activity list on the UCI dataset page.
LABEL_MAP = {4: 0, 12: 1, 13: 2, 2: 3, 3: 4}   # walking, ascending, descending stairs, sitting, standing
CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring"


def load_file(path):
    """Read one subject file. Returns timestamps (s), activity IDs, hand acc (m/s^2, may hold NaN), subject id."""
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN")
    if df.shape[1] != N_COLS:
        raise ValueError(f"{Path(path).name}: expected {N_COLS} columns, found {df.shape[1]}")
    t = df[0].to_numpy(dtype=float)
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError(f"{Path(path).name}: timestamps are not strictly increasing")
    if abs(np.median(dt) - 1 / FS_IN) > 1e-4:
        raise ValueError(f"{Path(path).name}: median timestep {np.median(dt):.4f}s is not {1 / FS_IN}s (100 Hz)")
    label = df[1].to_numpy().astype(int)
    acc = df[HAND_ACC16].to_numpy(dtype=float)
    subject = int(re.search(r"subject(\d+)", Path(path).name).group(1))
    return t, label, acc, subject


def nan_runs(mask):
    """Start (inclusive) and end (exclusive) indices of every run of True in a boolean array."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2]


def clean_segment(acc, stats):
    """acc (L, 3) in m/s^2 with NaN, from ONE activity. Returns a list of NaN-free (L_i, 3) pieces.

    - A NaN run longer than MAX_NAN_RUN cuts the segment in two (no piece spans it).
    - NaN at the start or end of a piece is trimmed (there is no value on both sides to interpolate between).
    - Shorter NaN runs inside a piece are filled by linear interpolation between their neighbours.
    """
    bad = np.isnan(acc).any(axis=1)
    starts, ends = nan_runs(bad)
    bounds, prev = [], 0
    for s, e in zip(starts, ends):
        if e - s > MAX_NAN_RUN:
            bounds.append((prev, s))
            prev = e
            stats["long_nan_runs_cut"] += 1
    bounds.append((prev, len(acc)))

    pieces = []
    for a, b in bounds:
        piece, piece_bad = acc[a:b].copy(), bad[a:b]
        good_idx = np.flatnonzero(~piece_bad)
        if len(good_idx) == 0:
            continue
        lo, hi = good_idx[0], good_idx[-1] + 1
        stats["edge_samples_trimmed"] += int(lo + (len(piece) - hi))
        piece, piece_bad = piece[lo:hi], piece_bad[lo:hi]
        if piece_bad.any():
            idx = np.arange(len(piece))
            for k in range(3):
                piece[piece_bad, k] = np.interp(idx[piece_bad], idx[~piece_bad], piece[~piece_bad, k])
            stats["nan_runs_interpolated"] += len(nan_runs(piece_bad)[0])
            stats["nan_samples_interpolated"] += int(piece_bad.sum())
        pieces.append(piece)
    return pieces


def to_20hz(sig):
    """Polyphase FIR resampling 100 -> 20 Hz. The low-pass (anti-alias) filter is part of the method."""
    return resample_poly(sig, UP, DOWN, axis=0, padtype="line")


def make_windows(sig):
    """Cut (L, 3) into (n, 50, 3) windows with stride 25. Never crosses the end of the segment."""
    if len(sig) < WIN:
        return np.empty((0, WIN, 3))
    n = (len(sig) - WIN) // STRIDE + 1
    idx = np.arange(WIN)[None, :] + STRIDE * np.arange(n)[:, None]
    return sig[idx]


def build_from_arrays(t, label, acc, subject, stats=None):
    """One subject file -> (X (M,50,3) float32 in g, y (M,) int64). Segments never mix activities or cross gaps."""
    stats = Counter() if stats is None else stats
    # cut wherever the activity changes or the timestamps jump by more than 0.5 s
    change = (label[1:] != label[:-1]) | (np.diff(t) > MAX_GAP_S)
    cuts = np.flatnonzero(change) + 1
    xs, ys = [], []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if label[a] not in LABEL_MAP:          # activity 0 (transient) and all other activities are dropped
            continue
        stats["segments"] += 1
        for piece in clean_segment(acc[a:b], stats):
            w = make_windows(to_20hz(piece / G))   # m/s^2 -> g, then resample, then window
            if len(w):
                xs.append(w)
                ys.append(np.full(len(w), LABEL_MAP[label[a]]))
    if not xs:
        return np.empty((0, WIN, 3), np.float32), np.empty((0,), np.int64)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def build(raw_dir):
    """Process every Protocol/subject*.dat under raw_dir."""
    files = sorted(Path(raw_dir).rglob("Protocol/subject*.dat"))
    if not files:
        raise ValueError(f"No Protocol/subject*.dat files under {raw_dir}. Check the path and the unzip step.")
    stats, per_subject = Counter(), {}
    xs, ys, ss = [], [], []
    for p in files:
        t, label, acc, subject = load_file(p)
        X, y = build_from_arrays(t, label, acc, subject, stats)
        per_subject[subject] = int(len(X))
        if len(X):
            xs.append(X); ys.append(y); ss.append(np.full(len(X), subject, dtype=np.int64))
    if not xs:
        raise ValueError("No windows produced. Check the raw data and the label map.")
    info = {"files": [p.name for p in files], "windows_per_subject": per_subject, "stats": dict(stats)}
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), info


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def summary(X, y, s, info):
    print(f"windows: {X.shape}  dtype={X.dtype}  subjects with windows: {len(np.unique(s))}")
    print("\nwindows per subject (0 = subject has no window for the 5 classes):")
    for subj, n in sorted(info["windows_per_subject"].items()):
        print(f"  subject {subj}: {n}")
    mag = np.linalg.norm(X, axis=-1).mean(axis=1)
    print("\nclass counts and mean |a| (g):")
    for k, name in enumerate(CLASS_NAMES):
        m = y == k
        print(f"  {k} {name:11s} n={m.sum():6d}  subjects={len(np.unique(s[m])):2d}  mean|a|={mag[m].mean():.3f}")
    print("\nNaN count in X:", int(np.isnan(X).sum()))
    print("cleaning stats:", info["stats"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/pamap2")
    ap.add_argument("--out", default="data/processed/pamap2_hand.npz")
    ap.add_argument("--sha256", default="", help="SHA-256 of the downloaded zip, stored in meta")
    args = ap.parse_args()
    if not args.sha256:
        print("WARNING: --sha256 not given, so meta will hold an empty checksum.")

    X, y, s, info = build(args.raw)
    summary(X, y, s, info)
    meta = {
        "dataset": "PAMAP2-hand", "source_url": SOURCE_URL, "raw_zip_sha256": args.sha256,
        "placement": "hand (IMU over the wrist of the dominant arm)", "sensor": "accelerometer +/-16 g, columns 5-7 of the hand IMU",
        "units": "g", "fs_original": FS_IN, "fs": FS_OUT, "window": WIN, "stride": STRIDE,
        "method": "segment, interpolate NaN runs <= 50 samples, resample_poly 100->20 Hz, window",
        "label_map_pamap2_id_to_class": {str(k): v for k, v in LABEL_MAP.items()}, "classes": CLASS_NAMES,
        "dropped": "activity 0 (transient) and all activities outside the 5 classes; Optional/ folder not read",
        "normalized": False, "info": info, "git_commit": git_commit(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, subject_id=s, meta=np.array(json.dumps(meta)))
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()