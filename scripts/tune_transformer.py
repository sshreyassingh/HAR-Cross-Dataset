"""Run the four-trial UCI-only Tiny Transformer hyperparameter tuning (PROTOCOL.md section 2).
Same pattern as scripts/tune_tcn.py and tune_cnn_uci.py. Run from the repo root:

    PYTHONPATH=. python scripts/tune_transformer.py | tee docs/tiny_transformer_tuning_summary.txt

Needs: data/processed/uci.npz, splits/uci.json, src/models/tiny_transformer.py
Uses the shared tuner (src/train/tuner.py): 4 random configs, UCI-HAR only, split 0, seed 0, max 10 epochs,
best config chosen by source-validation macro F1. No WISDM or PAMAP2 data is read.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataloader import create_dataloaders
from src.models.tiny_transformer import build_model, count_params
from src.train.tuner import tune_model

MODEL_NAME = "tiny_transformer"    # team decision: used in file names and the CSV
SEED = 0
SPLIT_INDEX = 0

DATA_PATH = Path("data/processed/uci.npz")
SPLIT_PATH = Path("splits/uci.json")
CHECKPOINT_DIR = Path(f"checkpoints/tuning/{MODEL_NAME}_uci")
OUTPUT_PATH = Path(f"results/raw/tuning_{MODEL_NAME}.csv")
BEST_CONFIG_PATH = Path(f"configs/best/{MODEL_NAME}.yaml")


def transformer_factory(cfg):
    """Adapt the tuner configuration (num_heads, dropout) to the Tiny Transformer."""
    return build_model({"num_heads": cfg["num_heads"], "dropout": cfg["dropout"]})


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing tuning CSV / best config")
    args = ap.parse_args()

    for path in [DATA_PATH, SPLIT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if OUTPUT_PATH.exists() and not args.force:
        sys.exit(f"{OUTPUT_PATH} already exists. Tuning is not repeated; use --force to overwrite it.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"git commit: {git_commit()}")
    print(f"device: {device}  torch {torch.__version__}  numpy {np.__version__}  python {sys.version.split()[0]}")

    print("Loading UCI data and split 0...")
    data = create_dataloaders(
        data_path=DATA_PATH,
        split_path=SPLIT_PATH,
        split_index=SPLIT_INDEX,
        batch_size=128,
    )

    print("Starting four Tiny Transformer tuning trials...")
    result = tune_model(
        model_type=MODEL_NAME,
        model_factory=transformer_factory,
        train_loader=data["train"],
        val_loader=data["val"],
        class_weights=data["class_weights"],
        normalization=data["normalization"],
        checkpoint_dir=CHECKPOINT_DIR,
        seed=SEED,
    )

    best_config = result["best_config"]

    # Width and parameter count of the winner (needed for Sanjana's parity check).
    best_model = transformer_factory(best_config)
    width = best_model.embed.out_features          # d_model
    n_params = count_params(best_model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model", "dataset", "split", "seed", "trial",
        "learning_rate", "dropout", "weight_decay", "num_heads",
        "val_macro_f1", "best_epoch", "is_best",
    ]
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for trial in result["trials"]:
            config = trial["config"]
            writer.writerow({
                "model": MODEL_NAME,
                "dataset": "uci",
                "split": SPLIT_INDEX,
                "seed": SEED,
                "trial": trial["trial"],
                "learning_rate": config["learning_rate"],
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "num_heads": config["num_heads"],
                "val_macro_f1": trial["val_macro_f1"],
                "best_epoch": trial["best_epoch"],
                "is_best": config == best_config,
            })

    BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_CONFIG_PATH, "w") as file:
        yaml.safe_dump({**best_config, "width": width, "n_params": n_params}, file)

    print("\nUCI Tiny Transformer tuning completed.")
    print("Best configuration:", best_config)
    print("Best validation macro-F1:", result["best_val_macro_f1"])
    print(f"Best model: d_model {width}, {n_params} parameters (send this to Sanjana)")
    print("Tuning CSV saved to:", OUTPUT_PATH)
    print("Best config saved to:", BEST_CONFIG_PATH)


if __name__ == "__main__":
    main()