"""PAMAP2 raw-data inspection (Day 1). Read-only: it loads the raw .dat files, prints
facts and saves one figure. It does NOT preprocess anything.

Run from the repo root:
    python scripts/pamap2_inspect.py --raw data/raw/pamap2 --fig figures/pamap2_raw_hand_acc.png

Column layout (54 columns per row, from the UCI dataset page; 0-based index here):
    0 timestamp (s), 1 activityID, 2 heart rate,
    3-19 hand IMU: 3 temperature, 4-6 acc +/-16 g (m/s^2), 7-9 acc +/-6 g, 10-12 gyro,
                   13-15 magnetometer, 16-19 orientation (documented as invalid)
    20-36 chest IMU, 37-53 ankle IMU
The script CHECKS this layout against the data (rest magnitude should be about 9.8 m/s^2).
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_COLS = 54
HAND_ACC16 = [4, 5, 6]            # hand IMU, +/-16 g accelerometer, m/s^2 (see layout above)
# Candidate activity IDs for the 5 protocol classes, from the UCI activity list.
# NOT final: the mapping must be confirmed and logged in docs/decision_log.md.
CANDIDATES = {2: "sitting", 3: "standing", 4: "walking", 12: "ascending stairs", 13: "descending stairs"}
NAMES = {0: "other (transient)", 1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
         6: "cycling", 7: "nordic walking", 9: "watching TV", 10: "computer work", 11: "car driving",
         12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
         18: "folding laundry", 19: "house cleaning", 20: "playing soccer", 24: "rope jumping"}


def load(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN")
    if df.shape[1] != N_COLS:
        raise ValueError(f"{path.name}: expected {N_COLS} columns, found {df.shape[1]}")
    return df


def describe_file(path, df):
    t = df[0].to_numpy()
    dt = np.diff(t)
    acc = df[HAND_ACC16].to_numpy()
    nan_rows = int(np.isnan(acc).any(axis=1).sum())
    print(f"{path.parent.name}/{path.name}: rows={len(df)}  duration={t[-1]-t[0]:.1f}s  "
          f"median dt={np.median(dt):.4f}s (rate ~{1/np.median(dt):.1f} Hz)  max dt={dt.max():.3f}s  "
          f"hand-acc NaN rows={nan_rows}  non-increasing timestamps={int((dt <= 0).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/pamap2")
    ap.add_argument("--fig", default="figures/pamap2_raw_hand_acc.png")
    args = ap.parse_args()

    files = sorted(Path(args.raw).rglob("subject*.dat"))
    if not files:
        raise SystemExit(f"No subject*.dat files under {args.raw}. Check the unzip step.")
    print(f"found {len(files)} .dat files\n")

    data = {}                                           # (folder, subject) -> DataFrame
    for p in files:
        subj = int(re.search(r"subject(\d+)", p.name).group(1))
        df = load(p)
        data[(p.parent.name, subj)] = df
        describe_file(p, df)

    subjects = sorted({s for _, s in data})
    print(f"\nsubjects found: {subjects}  (count {len(subjects)})")

    # Rows per activity per subject (Protocol folder only)
    prot = {s: d for (f, s), d in data.items() if f.lower() == "protocol"}
    print("\nrows per activityID per subject (Protocol folder):")
    table = pd.DataFrame({s: d[1].value_counts() for s, d in prot.items()}).fillna(0).astype(int).sort_index()
    table.index = [f"{int(i)} {NAMES.get(int(i), '?')}" for i in table.index]
    print(table.to_string())

    # Units / layout check: rest windows should have magnitude about 9.8 m/s^2 if these are the right columns
    print("\nhand acc (+/-16 g columns) mean magnitude at rest:")
    for aid in (2, 3):
        mags = [np.linalg.norm(d.loc[d[1] == aid, HAND_ACC16].dropna().to_numpy(), axis=1).mean()
                for d in prot.values() if (d[1] == aid).any()]
        print(f"  activity {aid} ({NAMES[aid]}): {np.mean(mags):.2f} m/s^2 over {len(mags)} subjects "
              f"(1 g = 9.80665 m/s^2, so about {np.mean(mags)/9.80665:.2f} g)")

    allacc = np.concatenate([d[HAND_ACC16].to_numpy() for d in prot.values()])
    print(f"\nhand acc overall range: min={np.nanmin(allacc):.1f}  max={np.nanmax(allacc):.1f} m/s^2 "
          f"(+/-16 g would be +/-156.9)")
    print(f"hand acc NaN fraction (Protocol): {np.isnan(allacc).any(axis=1).mean():.4f}")

    # Figure: 5 s (500 samples) from the first stretch of each candidate activity with no NaN
    fig, axes = plt.subplots(len(CANDIDATES), 1, figsize=(9, 11), sharex=True)
    for ax, (aid, name) in zip(axes, CANDIDATES.items()):
        shown = False
        for s, d in sorted(prot.items()):
            acc = d.loc[d[1] == aid, HAND_ACC16].to_numpy()      # rows of this activity, in time order
            for start in range(0, len(acc) - 500 + 1, 100):
                x = acc[start:start + 500]
                if not np.isnan(x).any():
                    ax.plot(np.arange(500) / 100, x, label=["x", "y", "z"])
                    ax.set_title(f"{aid} {name} (subject {s}, samples {start}-{start + 500} of the activity)",
                                 fontsize=9)
                    ax.legend(loc="upper right", ncol=3, fontsize=7)
                    shown = True
                    break
            if shown:
                break
        if not shown:
            ax.set_title(f"{aid} {name}: no NaN-free 500-sample stretch found", fontsize=9)
        ax.set_ylabel("m/s^2")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("PAMAP2 hand IMU +/-16 g accelerometer, raw")
    plt.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig, dpi=150)
    print("\nsaved", args.fig)


if __name__ == "__main__":
    main()