"""Export per-window CNN predictions for the final runs (PROTOCOL.md section 3).

run_epoch in src/train/trainer.py returns only metrics, so this script reruns inference itself, using the
same windows and the same source normalization as scripts/run_experiments.py:
  within: the source split's test subjects, in the order they appear in data/processed/{source}.npz
          (split index = seed, as in the runner)
  cross : ALL windows of the target, in file order

Before anything is written, the metrics recomputed from these predictions must match the matching row of
results/raw/cnn_results.csv (n_windows, n_params exactly; acc and macro_f1 within 1e-4). If they do not,
the script stops and writes nothing for that (source, seed).

Output (kept out of git): predictions/cnn/{source}/seed_{seed}.csv with columns
    kind (within/cross), target, window_index, subject_id, y_true, y_pred
window_index is the row index in data/processed/{target}.npz.

Usage (after the final runs exist):
    python3 -m scripts.export_predictions --final-csv results/raw/cnn_results.csv

Status: not yet run on real checkpoints.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DOMAINS = ["uci", "wisdm", "pamap2_hand"]
MODEL = "cnn"
SEEDS = [0, 1, 2]
COLUMNS = ["kind", "target", "window_index", "subject_id", "y_true", "y_pred"]
TOL = 1e-4


def ckpt_path(ckpt_dir, source, seed):
    return os.path.join(ckpt_dir, MODEL, source, f"seed_{seed}.pt")


def load_cnn(path, config_path, device):
    """Architecture from configs/cnn_best.json; weights and SOURCE normalization from the checkpoint."""
    import torch
    from src.models.cnn import build_cnn

    with open(config_path) as f:
        cfg = json.load(f)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    need = ["model_state_dict", "normalization_mean", "normalization_std"]
    missing = [k for k in need if k not in ck]
    if missing:
        raise KeyError(f"checkpoint {path} lacks {missing}. Keys found: {sorted(ck)}")
    model = build_cnn(kernel=cfg["kernel_size"], dropout=cfg["dropout"], n_blocks=cfg["n_blocks"],
                      width=cfg["width"])
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, ck["normalization_mean"], ck["normalization_std"]


def predict(model, X, device, batch=1024):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def check_against_final(final_df, source, target, seed, n, n_params, y, p):
    """Recomputed numbers must reproduce the reported CSV row, or nothing is written."""
    sel = final_df[(final_df.model == MODEL) & (final_df.source == source) & (final_df.target == target) &
                   (final_df.seed == seed) & (final_df.split == seed)]
    if len(sel) != 1:
        raise RuntimeError(f"expected exactly 1 final row for {source}->{target} seed {seed}, found {len(sel)}")
    ref = sel.iloc[0]
    got = {"acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro")}
    problems = []
    for col, val in got.items():
        if abs(float(ref[col]) - val) > TOL:
            problems.append(f"{col}: CSV {float(ref[col]):.6f} vs export {val:.6f}")
    if int(ref["n_windows"]) != n:
        problems.append(f"n_windows: CSV {int(ref['n_windows'])} vs export {n}")
    if int(ref["n_params"]) != n_params:
        problems.append(f"n_params: CSV {int(ref['n_params'])} vs export {n_params}")
    if problems:
        raise RuntimeError(f"MISMATCH {source}->{target} seed {seed}: " + "; ".join(problems) +
                           ". Check the checkpoint, the device and the window selection. Nothing written.")


def export_one(source, seed, a, final_df):
    from src.data.dataloader import load_split
    from src.data.dataset import load_har_data
    from src.data.normalization import apply_normalization

    model, mean, std = load_cnn(ckpt_path(a.ckpt_dir, source, seed), a.config, a.device)
    n_params = int(model.n_params)
    frames = []
    for target in DOMAINS:
        X, y, sid = load_har_data(os.path.join(a.data_dir, f"{target}.npz"))
        if target == source:
            split = load_split(os.path.join(a.splits_dir, f"{source}.json"), seed)
            idx = np.where(np.isin(sid, np.asarray(split["test"])))[0]
            kind = "within"
        else:
            idx = np.arange(len(X))
            kind = "cross"
        p = predict(model, apply_normalization(X[idx], mean, std), a.device)
        check_against_final(final_df, source, target, seed, len(idx), n_params, y[idx], p)
        frames.append(pd.DataFrame({"kind": kind, "target": target, "window_index": idx,
                                    "subject_id": sid[idx], "y_true": y[idx], "y_pred": p}, columns=COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(a.out_dir, MODEL, source, f"seed_{seed}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"saved {path}: {len(out)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-csv", default="results/raw/cnn_results.csv")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--ckpt-dir", default="checkpoints/final")
    ap.add_argument("--config", default="configs/cnn_best.json")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--device", default="cpu", help="use the same device as the final runs if possible")
    ap.add_argument("--sources", nargs="+", default=DOMAINS, choices=DOMAINS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    final_df = pd.read_csv(a.final_csv)
    for source in a.sources:
        for seed in a.seeds:
            path = os.path.join(a.out_dir, MODEL, source, f"seed_{seed}.csv")
            if os.path.exists(path) and not a.overwrite:
                print(f"exists, skipping {path}")
                continue
            export_one(source, seed, a, final_df)


if __name__ == "__main__":
    main()
