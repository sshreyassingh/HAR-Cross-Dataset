"""Tidy the CNN tuning CSV and save the winning config.

Reads  results/raw/tuning_cnn.csv   (written by Sanjana's tuning run)
Adds   width and n_params columns, computed with build_cnn (3 blocks, kernel from each row)
Writes results/raw/tuning_cnn.csv   (same rows, two extra columns, safe to re-run)
       configs/cnn_best.json        (the is_best row, plus width and n_params)

Run from the repo root:  python3 -m scripts.tidy_tuning_cnn
"""
import csv
import json
import pathlib

from src.models.cnn import build_cnn

CSV = pathlib.Path("results/raw/tuning_cnn.csv")
BEST = pathlib.Path("configs/cnn_best.json")
N_BLOCKS = 3

rows = list(csv.DictReader(open(CSV, newline="")))
assert len(rows) == 4, f"expected 4 tuning rows, found {len(rows)}"

fields = [f for f in rows[0].keys() if f not in ("width", "n_params")] + ["width", "n_params"]
for r in rows:
    m = build_cnn(kernel=int(r["kernel_size"]), dropout=float(r["dropout"]), n_blocks=N_BLOCKS)
    r["width"], r["n_params"] = m.width, m.n_params

best = [r for r in rows if r["is_best"] == "True"]
assert len(best) == 1, "exactly one row must have is_best=True"
top = max(rows, key=lambda r: float(r["val_macro_f1"]))
assert top is best[0], "is_best row is not the highest val_macro_f1"
b = best[0]

with open(CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

BEST.parent.mkdir(exist_ok=True)
BEST.write_text(json.dumps({
    "model": "cnn",
    "learning_rate": float(b["learning_rate"]),
    "dropout": float(b["dropout"]),
    "weight_decay": float(b["weight_decay"]),
    "kernel_size": int(b["kernel_size"]),
    "n_blocks": N_BLOCKS,
    "width": int(b["width"]),
    "n_params": int(b["n_params"]),
    "selected_by": "source-val macro F1 (UCI, split 0, seed 0, max 10 epochs)",
    "val_macro_f1": float(b["val_macro_f1"]),
    "trial": int(b["trial"]),
}, indent=2) + "\n")

for r in rows:
    print(r["trial"], r["kernel_size"], r["width"], r["n_params"], round(float(r["val_macro_f1"]), 4), r["is_best"])
print("saved", CSV, "and", BEST)