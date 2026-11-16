"""Inference-only ablations for the CNN (PROTOCOL.md section 3).

Seed 0, split 0, the 6 directed core pairs, each with an unperturbed CONTROL row that must match the
final result for the same pair. No retraining. Perturbations come from Sanjana's src/analysis/perturb.py:
nobody reimplements them, so the real run refuses to use anything else.

Output: results/raw/ablation_cnn.csv with the final-results columns plus `ablation` and `level`.

Usage (real run, after the final runs exist):
    python -m scripts.run_ablation_cnn --data-dir data/processed --ckpt-dir checkpoints/final \
        --final-csv results/raw/cnn_results.csv --out results/raw/ablation_cnn.csv

Dry run with the stand-in perturbations (logic check only; writes to a scratch file, never to results/raw):
    python -m scripts.run_ablation_cnn --standin --out /tmp/ablation_dryrun.csv ...

Already matched to the repo: load_model (src/train/trainer.py checkpoint keys, architecture from
configs/cnn_best.json) and perturb_real (src/analysis/perturb.py apply_perturbation).
ckpt_path() follows Sanjana's layout checkpoints/final/{model}/{source}/seed_{seed}.pt.
Labels uci, wisdm and pamap2_hand confirmed by Sanjana (cnn_results.csv and checkpoint folders).
Normalization uses the harness helper src/data/normalization.py apply_normalization.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

DOMAINS = ["uci", "wisdm", "pamap2_hand"]
MODEL = "cnn"
SEED, SPLIT = 0, 0
# (ablation, level). Levels are strings so the CSV is unambiguous.
ABLATIONS = [("unit_mismatch", "9.80665"), ("time_stretch", "0.75"),
             ("time_stretch", "1.25"), ("random_rotation", "seed0")]
COLUMNS = ["model", "source", "target", "seed", "split", "n_windows", "n_params",
           "acc", "macro_f1", "weighted_f1", "kappa", "ablation", "level"]
KEY = ["model", "source", "target", "seed", "split", "ablation", "level"]
CONTROL_TOL = 1e-4          # control macro F1 must match the final result within this


def pairs():
    """The 6 directed core pairs (source != target)."""
    return [(s, t) for s, t in itertools.permutations(DOMAINS, 2)]


# ----------------------------------------------------------------------------------------------
# ADAPTERS: edit these to the real interfaces once Sanjana confirms them.
# ----------------------------------------------------------------------------------------------
def ckpt_path(ckpt_dir, source):
    """Seed-0 checkpoint for this source. Sanjana's runner saves
    checkpoints/final/{model}/{source}/seed_{seed}.pt (confirm the source labels match DOMAINS)."""
    return os.path.join(ckpt_dir, MODEL, source, f"seed_{SEED}.pt")


def load_model(path, config_path="configs/cnn_best.json"):
    """Return (predict, n_params). predict(X_raw_g) -> int labels, applying the SOURCE normalization
    stored in the checkpoint internally.

    src/train/trainer.py saves: model_state_dict, normalization_mean, normalization_std, best_epoch, ...
    It does NOT save kernel or width, so the architecture comes from configs/cnn_best.json."""
    import json

    import torch
    from src.data.normalization import apply_normalization
    from src.models.cnn import build_cnn

    with open(config_path) as f:
        cfg = json.load(f)
    # weights_only=False: these are our own checkpoints, and the normalization stats may be numpy arrays.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    need = ["model_state_dict", "normalization_mean", "normalization_std"]
    missing = [k for k in need if k not in ck]
    if missing:
        raise KeyError(f"checkpoint {path} lacks {missing}. Keys found: {sorted(ck)}")
    model = build_cnn(kernel=cfg["kernel_size"], dropout=cfg["dropout"], n_blocks=cfg.get("n_blocks", 3),
                      width=cfg["width"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    mean, std = ck["normalization_mean"], ck["normalization_std"]

    @torch.no_grad()
    def predict(X):
        # The harness' own helper (src/data/normalization.py), so the formula can never drift from training.
        Xn = apply_normalization(X, mean, std)
        out = []
        for i in range(0, len(Xn), 1024):
            out.append(model(torch.as_tensor(Xn[i:i + 1024])).argmax(1).numpy())
        return np.concatenate(out)

    return predict, int(model.n_params)


def perturb_real(X, ablation, level, seed=SEED):
    """Sanjana's src/analysis/perturb.py. X is RAW g, shape (N, 50, 3); returns float32, same shape.
    apply_perturbation accepts: unit_mismatch, time_stretch (level 0.75 or 1.25), random_rotation, control."""
    from src.analysis.perturb import apply_perturbation

    lvl = float(level) if ablation == "time_stretch" else None
    return apply_perturbation(X, ablation, lvl, seed)


def perturb_standin(X, ablation, level, seed=SEED):
    """STAND-IN for dry runs only. Never used for reported numbers."""
    from scipy.signal import resample
    if ablation == "unit_mismatch":
        return X * float(level)
    if ablation == "time_stretch":
        f = float(level)
        m = int(round(50 * f))
        r = resample(X, m, axis=1)
        if m >= 50:
            return r[:, :50, :]
        pad = 50 - m
        return np.concatenate([r, np.repeat(r[:, -1:, :], pad, axis=1)], axis=1)
    if ablation == "random_rotation":
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        return X @ q.T
    raise ValueError(ablation)


# ----------------------------------------------------------------------------------------------
# Interface-independent logic.
# ----------------------------------------------------------------------------------------------
def load_domain(data_dir, name):
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    return d["X"].astype(np.float32), d["y"].astype(int)


def metrics(y, p):
    return {"acc": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"),
            "kappa": cohen_kappa_score(y, p)}


def row(source, target, n, n_params, m, ablation, level):
    return {"model": MODEL, "source": source, "target": target, "seed": SEED, "split": SPLIT,
            "n_windows": int(n), "n_params": int(n_params), **m, "ablation": ablation, "level": level}


def check_control(control_row, final_df):
    """The unperturbed control must reproduce the final result for the same pair, seed 0."""
    sel = final_df[(final_df.model == MODEL) & (final_df.source == control_row["source"]) &
                   (final_df.target == control_row["target"]) & (final_df.seed == SEED)]
    if len(sel) != 1:
        raise RuntimeError(f"expected exactly 1 final row for {control_row['source']}->{control_row['target']} "
                           f"seed {SEED}, found {len(sel)}")
    ref = sel.iloc[0]
    problems = []
    for col in ("macro_f1", "acc"):
        if abs(float(ref[col]) - control_row[col]) > CONTROL_TOL:
            problems.append(f"{col}: final {float(ref[col]):.6f} vs control {control_row[col]:.6f}")
    for col in ("n_windows", "n_params"):
        if int(ref[col]) != control_row[col]:
            problems.append(f"{col}: final {int(ref[col])} vs control {control_row[col]}")
    if problems:
        raise RuntimeError(
            f"CONTROL MISMATCH {control_row['source']}->{control_row['target']}: " + "; ".join(problems) +
            ". Stop and check the checkpoint, the normalization (it must match the harness exactly) and the "
            "windows before using any ablation number.")


def check_perturbation(X, Xp, ablation, level):
    """Guard against a silent no-op: every real perturbation must change the data, and unit mismatch must
    be exactly the 9.80665 factor."""
    assert Xp.shape == X.shape, f"{ablation} {level} changed shape {X.shape} -> {Xp.shape}"
    assert np.isfinite(Xp).all(), f"{ablation} {level} produced NaN or inf"
    assert not np.allclose(Xp, X), f"{ablation} {level} left the data unchanged (the perturbation did nothing)"
    if ablation == "unit_mismatch":
        assert np.allclose(Xp, X * 9.80665, rtol=1e-4, atol=1e-6), "unit_mismatch is not the 9.80665 factor"


def run(data_dir, ckpt_dir, final_csv, out_csv, perturb_fn, load_model_fn=load_model, domains_loader=load_domain,
        pair_list=None):
    """One control row per directed pair, then the 4 ablation rows for that pair.

    The CSV is rewritten after EVERY pair, so a crash keeps finished pairs and a rerun resumes. A control
    that does not match the final result raises before that pair's rows are written."""
    final_df = pd.read_csv(final_csv)
    out = pd.DataFrame(columns=COLUMNS)
    done = set()
    if os.path.exists(out_csv):
        out = pd.read_csv(out_csv)
        done = {tuple(r) for r in out[KEY].astype(str).itertuples(index=False, name=None)}
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    for source, target in (pair_list or pairs()):
        todo = [(a, l) for a, l in [("control", "none")] + ABLATIONS
                if (MODEL, source, target, str(SEED), str(SPLIT), a, l) not in done]
        if not todo:
            continue
        X, y = domains_loader(data_dir, target)            # ALL target windows, raw g
        predict, n_params = load_model_fn(ckpt_path(ckpt_dir, source))
        pair_rows = []
        for ablation, level in todo:
            if ablation == "control":
                Xp = X
            else:
                Xp = perturb_fn(X, ablation, level, SEED)
                check_perturbation(X, Xp, ablation, level)
            r = row(source, target, len(X), n_params, metrics(y, predict(Xp)), ablation, level)
            if ablation == "control":
                check_control(r, final_df)                  # raises on mismatch, before this pair is saved
            pair_rows.append(r)
        out = pd.concat([out, pd.DataFrame(pair_rows, columns=COLUMNS)], ignore_index=True)
        out.to_csv(out_csv, index=False)
        print(f"saved {source} -> {target}: {len(pair_rows)} rows")
    return out.infer_objects()      # concat onto an empty frame leaves object dtypes; the CSV is unaffected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="checkpoints/final")
    ap.add_argument("--final-csv", default="results/raw/cnn_results.csv")
    ap.add_argument("--out", default="results/raw/ablation_cnn.csv")
    ap.add_argument("--standin", action="store_true", help="dry run with stand-in perturbations")
    a = ap.parse_args()
    if a.standin:
        if os.path.abspath(a.out).startswith(os.path.abspath("results/raw")):
            sys.exit("--standin must not write into results/raw. Choose a scratch --out path.")
        print("WARNING: stand-in perturbations. Dry run only, NOT for reported numbers.")
        fn = perturb_standin
    else:
        fn = perturb_real
    out = run(a.data_dir, a.ckpt_dir, a.final_csv, a.out, fn)
    print(out.groupby(["ablation", "level"]).macro_f1.mean().round(4))
    print("saved", a.out)


if __name__ == "__main__":
    main()
