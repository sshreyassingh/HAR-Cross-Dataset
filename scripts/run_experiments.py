
"""Run fixed cross-dataset HAR experiments using saved best configs."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloader import create_dataloaders, load_split
from src.data.dataset import HARWindowDataset, load_har_data
from src.data.normalization import apply_normalization
from src.train.trainer import train_model, run_epoch
from src.train.results import write_result


ALIASES = {"uci_har": "uci"}

EXPECTED = {
    "uci": (30, (21, 4, 5)),
    "wisdm": (36, (25, 5, 6)),
    "pamap2_hand": (8, None),
}

PAMAP2_SPLIT_SIZES = [
    (4, 1, 3),
    (4, 1, 3),
    (5, 1, 2),
]

BEST_CONFIG_PATHS = {
    "cnn": "configs/cnn_best.json",
    "tcn": "configs/tcn.yaml",
    "tiny_transformer": "configs/best/tiny_transformer.yaml",
}


def resolve_path(path):
    """Resolve relative output paths against the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_configs(config_path):
    config_path = resolve_path(config_path)

    with open(config_path) as f:
        experiment_config = yaml.safe_load(f)

    base_path = config_path.parent / experiment_config["_base_"]

    with open(base_path) as f:
        base = yaml.safe_load(f)

    return base, experiment_config["experiment"]


def get_domains(experiment):
    return [
        ALIASES.get(domain, domain)
        for domain in experiment["core_domains"]
    ]


def load_best_configs():
    configs = {}

    for model_name, relative_path in BEST_CONFIG_PATHS.items():
        path = ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Missing best configuration: {path}"
            )

        with open(path) as f:
            config = yaml.safe_load(f)

        for key in ("learning_rate", "weight_decay"):
            if key not in config:
                raise ValueError(
                    f"{model_name}: missing '{key}' in {relative_path}"
                )

        configs[model_name] = config

    return configs


def validate_datasets(domains, base):
    print("\nChecking datasets and subject splits...")

    for domain in domains:
        data_path = ROOT / base["data"]["processed_file"].format(
            domain=domain
        )
        split_path = ROOT / base["data"]["splits_file"].format(
            domain=domain
        )

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        if not split_path.exists():
            raise FileNotFoundError(split_path)

        _, _, subject_ids = load_har_data(data_path)
        ids = set(int(x) for x in np.unique(subject_ids))

        expected_count, expected_sizes = EXPECTED[domain]

        if len(ids) != expected_count:
            raise ValueError(
                f"{domain}: expected {expected_count} subjects, "
                f"found {len(ids)}."
            )

        test_subjects = []

        for index in range(3):
            split = load_split(split_path, index)

            train = set(int(x) for x in split["train"])
            val = set(int(x) for x in split["val"])
            test = set(int(x) for x in split["test"])

            actual_sizes = (len(train), len(val), len(test))

            expected_fold_sizes = (
                PAMAP2_SPLIT_SIZES[index]
                if domain == "pamap2_hand"
                else expected_sizes
            )

            if actual_sizes != expected_fold_sizes:
                raise ValueError(
                    f"{domain}, split {index}: expected "
                    f"{expected_fold_sizes}, got {actual_sizes}."
                )

            if train & val or train & test or val & test:
                raise ValueError(
                    f"{domain}: overlapping subjects in split {index}."
                )

            if train | val | test != ids:
                raise ValueError(
                    f"{domain}: split {index} does not cover all subjects."
                )

            test_subjects.extend(test)

        if domain == "pamap2_hand":
            if len(set(test_subjects)) != 8:
                raise ValueError(
                    "PAMAP2 test folds must cover all 8 subjects "
                    "exactly once."
                )

        print(f"{domain}: OK ({len(ids)} subjects)")


def build_model(model_name, config):
    config = dict(config)

    if model_name == "cnn":
        from src.models.cnn import build_cnn

        return build_cnn(
            kernel=config["kernel_size"],
            dropout=config["dropout"],
            n_blocks=config["n_blocks"],
            width=config["width"],
        )

    if model_name == "tcn":
        from src.models.tcn import build_model as build_tcn

        return build_tcn(config)

    if model_name == "tiny_transformer":
        from src.models.tiny_transformer import build_model

        config["target_params"] = 75000
        return build_model(config)

    raise ValueError(f"Unknown model: {model_name}")


def count_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def validate_best_configs(models, configs):
    print("\nChecking saved model configurations...")

    parameter_counts = {}

    for model_name in models:
        config = configs[model_name]
        model = build_model(model_name, config)

        actual = count_params(model)
        expected = int(config["n_params"])

        if actual != expected:
            raise ValueError(
                f"{model_name}: saved parameter count is {expected}, "
                f"but built model has {actual}."
            )

        parameter_counts[model_name] = actual

        print(
            f"{model_name}: {actual:,} parameters | "
            "configuration verified"
        )

    ratio = (
        max(parameter_counts.values())
        / min(parameter_counts.values())
    )

    print(f"\nParameter parity ratio: {ratio:.4f}")

    if ratio > 1.10:
        raise RuntimeError(
            "Parameter parity failed: max/min must be <= 1.10."
        )

    print("Parameter parity passed.")
    return parameter_counts


def make_target_loader(
    domain, normalization, batch_size, base, split_index=0
):
    """Load all target windows using source-training normalization."""

    path = ROOT / base["data"]["processed_file"].format(
        domain=domain
    )

    X, y, subject_id = load_har_data(path)

    # Cross-dataset evaluation uses ALL target windows.
    # No target split or test-subject filtering is applied.

    X = apply_normalization(
        X,
        normalization["mean"],
        normalization["std"],
    )

    dataset = HARWindowDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def existing_result_keys(output_path):
    if not output_path.exists():
        return set()

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)

        return {
            (
                row["source"],
                row["target"],
                row["seed"],
                row["split"],
            )
            for row in reader
        }


def save_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def run_final_experiments(
    models,
    domains,
    seeds,
    best_configs,
    parameter_counts,
    base,
    device,
    results_dir,
    checkpoints_dir,
    max_epochs=None,
):
    batch_size = base["train"]["batch_size"]
    patience = base["train"]["early_stopping_patience"]
    epochs = max_epochs or base["train"]["max_epochs"]

    for model_name in models:
        print("\n" + "=" * 60)
        print(f"EXPERIMENTS: {model_name}")
        print("=" * 60)

        config = best_configs[model_name]
        n_params = parameter_counts[model_name]

        output_path = (
            results_dir / "raw" / f"{model_name}_results.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        completed = existing_result_keys(output_path)

        for source in domains:
            for seed in seeds:
                print(
                    f"\nModel={model_name} | "
                    f"Source={source} | Seed={seed}"
                )

                all_results_exist = all(
                    (source, target, str(seed), str(seed)) in completed
                    for target in domains
                )

                if all_results_exist:
                    print("All target results already exist. Skipping.")
                    continue

                set_seed(seed)

                source_data = ROOT / base["data"][
                    "processed_file"
                ].format(domain=source)

                source_split = ROOT / base["data"][
                    "splits_file"
                ].format(domain=source)

                loaders = create_dataloaders(
                    data_path=source_data,
                    split_path=source_split,
                    split_index=seed,
                    batch_size=batch_size,
                )

                model = build_model(model_name, config)

                checkpoint = (
                    checkpoints_dir
                    / "final"
                    / model_name
                    / source
                    / f"seed_{seed}.pt"
                )

                trained = train_model(
                    model=model,
                    train_loader=loaders["train"],
                    val_loader=loaders["val"],
                    class_weights=loaders["class_weights"],
                    normalization=loaders["normalization"],
                    checkpoint_path=checkpoint,
                    learning_rate=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                    max_epochs=epochs,
                    patience=patience,
                    device=device,
                )

                model = trained["model"]
                model.eval()

                metadata = {
                    "model": model_name,
                    "source": source,
                    "seed": seed,
                    "split": seed,
                    "best_config": config,
                    "n_params": n_params,
                    "best_epoch": trained["best_epoch"],
                    "best_val_macro_f1": trained[
                        "best_val_macro_f1"
                    ],
                    "git_commit": get_git_commit(),
                }

                save_metadata(
                    checkpoint.parent
                    / f"seed_{seed}_metadata.json",
                    metadata,
                )

                criterion = nn.CrossEntropyLoss()

                for target in domains:
                    key = (
                        source,
                        target,
                        str(seed),
                        str(seed),
                    )

                    if key in completed:
                        print(
                            f"{source} -> {target}: "
                            "already recorded, skipping."
                        )
                        continue

                    if target == source:
                        target_loader = loaders["test"]
                    else:
                        target_loader = make_target_loader(
                            target,
                            loaders["normalization"],
                            batch_size,
                            base,
                            split_index=seed,
                        )

                    metrics = run_epoch(
                        model=model,
                        loader=target_loader,
                        criterion=criterion,
                        device=device,
                    )

                    row = {
                        "model": model_name,
                        "source": source,
                        "target": target,
                        "seed": seed,
                        "split": seed,
                        "n_windows": len(target_loader.dataset),
                        "n_params": n_params,
                        "acc": metrics["acc"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "kappa": metrics["kappa"],
                    }

                    write_result(row, output_path)
                    completed.add(key)

                    print(
                        f"{source} -> {target} | "
                        f"Macro-F1: {metrics['macro_f1']:.4f}"
                    )

        print(f"\nResults saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed cross-dataset HAR experiments."
    )

    parser.add_argument(
        "--config",
        default="configs/exp_cross_dataset.yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate datasets, splits and model configs without training.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(BEST_CONFIG_PATHS),
        default=None,
        help="Optional subset of models to run.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(EXPECTED),
        default=None,
        help="Optional subset of domains to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of split/seed indices (0, 1, 2).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root directory. Relative paths are under the project root.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Checkpoint root directory. Relative paths are under the project root.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional epoch limit for smoke tests.",
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no GPU is available."
        )

    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1.")

    base, experiment = load_configs(args.config)

    configured_domains = get_domains(experiment)
    configured_models = experiment["models"]
    configured_seeds = base["final"]["seeds"]

    domains = args.domains or configured_domains
    models = args.models or configured_models
    seeds = args.seeds or configured_seeds

    if any(seed not in (0, 1, 2) for seed in seeds):
        raise ValueError("Seeds/split indices must be 0, 1, or 2.")

    if not set(domains).issubset(configured_domains):
        raise ValueError("Requested domains are not in the experiment config.")

    if not set(models).issubset(configured_models):
        raise ValueError("Requested models are not in the experiment config.")

    results_dir = resolve_path(args.results_dir)
    checkpoints_dir = resolve_path(args.checkpoints_dir)

    print("Domains:", domains)
    print("Models:", models)
    print("Seeds:", seeds)
    print("Device:", args.device)
    print("Results directory:", results_dir)
    print("Checkpoints directory:", checkpoints_dir)
    print("Epoch limit:", args.max_epochs or "config default")

    validate_datasets(domains, base)

    all_best_configs = load_best_configs()
    best_configs = {
        model: all_best_configs[model]
        for model in models
    }

    parameter_counts = validate_best_configs(
        models,
        best_configs,
    )

    if args.preflight_only:
        print("\nPreflight checks passed.")
        return

    print("\nUsing saved best configurations.")
    print("Hyperparameter tuning will NOT be repeated.")

    run_final_experiments(
        models=models,
        domains=domains,
        seeds=seeds,
        best_configs=best_configs,
        parameter_counts=parameter_counts,
        base=base,
        device=args.device,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        max_epochs=args.max_epochs,
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
