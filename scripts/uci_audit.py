"""UCI-HAR audit: checksums, units, rate, raw-window plot.
Run from the repo root: python3 scripts/uci_audit.py
"""
import datetime, hashlib, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = pathlib.Path("data/raw/uci_har")
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
NAMES = {1: "walking", 2: "upstairs", 3: "downstairs", 4: "sitting", 5: "standing", 6: "laying"}

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()

pathlib.Path("docs").mkdir(exist_ok=True)
pathlib.Path("figures").mkdir(exist_ok=True)

# 1. checksums
zips = sorted(RAW.glob("*.zip"))
assert zips, f"no zip files in {RAW.resolve()}, run from the repo root after downloading"
lines = [
    "UCI-HAR download record",
    f"date_recorded: {datetime.date.today()}",
    f"source_url: {URL}",
    "uci_dataset_id: 240",
    "version: 1.0",
]
lines += [f"{z.name}  sha256={sha256(z)}  bytes={z.stat().st_size}" for z in zips]
pathlib.Path("docs/data_checksums.txt").write_text("\n".join(lines) + "\n")

# 2. load total_acc (train split)
root = next(p for p in RAW.rglob("UCI HAR Dataset") if (p / "train").exists())
sig = root / "train/Inertial Signals"
X = np.stack([np.loadtxt(sig / f"total_acc_{a}_train.txt") for a in "xyz"], axis=-1)
y = np.loadtxt(root / "train/y_train.txt", dtype=int)
subj = np.loadtxt(root / "train/subject_train.txt", dtype=int)

audit = [f"X {X.shape}, y {y.shape}, {np.unique(subj).size} train subjects"]
for k, n in NAMES.items():
    audit.append(f"{n:11s} mean |a| = {np.linalg.norm(X[y == k], axis=-1).mean():.3f}")

ok = [np.allclose(X[i, 64:], X[i + 1, :64])
      for i in range(len(X) - 1) if subj[i] == subj[i + 1] and y[i] == y[i + 1]]
audit.append(f"50% overlap holds in {np.mean(ok):.3f} of {len(ok)} consecutive pairs")

w = X[y == 1][:, :, 1]
w = w - w.mean(1, keepdims=True)
f = np.fft.rfftfreq(128, d=1 / 50)
audit.append(f"walking dominant freq (Hz): {f[np.abs(np.fft.rfft(w)).mean(0).argmax()]:.2f}")

pathlib.Path("docs/uci_audit.txt").write_text("\n".join(audit) + "\n")
print("\n".join(lines)); print(); print("\n".join(audit))

# 3. plot
t = np.arange(128) / 50
rng = np.random.default_rng(0)
fig, axes = plt.subplots(5, 3, figsize=(13, 11), sharex=True)
for r, k in enumerate([1, 2, 3, 4, 5]):
    idx = rng.choice(np.where(y == k)[0], 3, replace=False)
    for c, a in enumerate("xyz"):
        for i in idx:
            axes[r, c].plot(t, X[i, :, c], lw=1)
        if r == 0: axes[r, c].set_title(f"total_acc_{a}")
        if c == 0: axes[r, c].set_ylabel(f"{NAMES[k]}\n(g)")
for c in range(3): axes[-1, c].set_xlabel("time (s)")
fig.suptitle("UCI-HAR raw windows (total_acc, 50 Hz, 3 random windows per activity)")
plt.tight_layout()
plt.savefig("figures/uci_raw_windows.png", dpi=150)