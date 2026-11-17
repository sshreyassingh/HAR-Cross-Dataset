"""Common efficiency table for the CNN, the TCN and the Tiny Transformer (Member C).

All three models are built from their tuned configs and measured the same way:
  parameters      trainable parameters (sum of p.numel() for p.requires_grad)
  flops           torch FlopCounterMode on one (1, 50, 3) window: 2 FLOPs per multiply-add for convolutions,
                  matrix multiplies and attention. BatchNorm, LayerNorm, activations, softmax and pooling are NOT counted.
  model_size_mb   serialized state_dict (torch.save to memory, decimal MB = 1e6 bytes; includes BatchNorm buffers)
  cpu_latency_ms  median time of one forward pass, batch 1, eval mode, inference_mode, 1 CPU thread, after warm-up

Run from the repo root:
    PYTHONPATH=. python scripts/efficiency.py | tee docs/efficiency_summary.txt
Needs the tuned configs (paths can be changed with --cnn-config / --tcn-config / --transformer-config).
"""
import argparse
import csv
import io
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from src.models import cnn as cnn_mod
from src.models import tcn as tcn_mod
from src.models import tiny_transformer as tt_mod

INPUT_SHAPE = (1, 50, 3)            # batch 1, 50 samples, 3 axes
THREADS, WARMUP, ITERS, ROUNDS, SEED = 1, 20, 200, 5, 0
OUTPUT = Path("results/tables/efficiency.csv")


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}. Pass the right path with the matching --*-config option.")
    with open(path) as f:
        return json.load(f) if path.suffix == ".json" else yaml.safe_load(f)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(args):
    """Build each model from its tuned config and check the parameter count against the one recorded with the config."""
    c, t, x = load_config(args.cnn_config), load_config(args.tcn_config), load_config(args.transformer_config)
    models = {
        "cnn": (cnn_mod.build_cnn(kernel=c["kernel_size"], dropout=c["dropout"], n_blocks=c["n_blocks"], width=c["width"]), c),
        "tcn": (tcn_mod.build_model({"kernel_size": t["kernel_size"], "dropout": t["dropout"], "width": t["width"]}), t),
        "tiny_transformer": (tt_mod.build_model({"num_heads": x["num_heads"], "dropout": x["dropout"]}), x),
    }
    for name, (model, cfg) in models.items():
        if count_params(model) != cfg["n_params"]:
            raise ValueError(f"{name}: built model has {count_params(model)} parameters, config records {cfg['n_params']}")
    if models["tiny_transformer"][0].embed.out_features != x["width"]:
        raise ValueError("transformer: built d_model differs from the config's width")
    return {name: model.eval() for name, (model, _) in models.items()}


def state_dict_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def count_flops(model, x):
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops()


def transformer_matmul_flops(model):
    """Hand count for the Tiny Transformer, used only as a cross-check of the counter (matmuls and attention)."""
    d, T, L = model.embed.out_features, tt_mod.N_TOKENS, tt_mod.N_LAYERS
    ff = tt_mod.FF_RATIO * d
    macs = T * tt_mod.PATCH * tt_mod.N_CHANNELS * d                  # patch embedding
    macs += L * (T * (3 * d * d + d * d + 2 * d * ff) + 2 * T * T * d)   # qkv + out projection + 2 feed-forward linears; QK^T and AV
    macs += d * tt_mod.N_CLASSES                                     # head (one pooled vector)
    return 2 * macs


def measure_latency(models, x):
    """Round-robin over the models so that slow drift of a shared CPU hits all of them equally."""
    torch.set_num_threads(THREADS)
    samples = {name: [] for name in models}
    for _ in range(ROUNDS):
        for name, model in models.items():
            with torch.inference_mode():
                for _ in range(WARMUP):
                    model(x)
                for _ in range(ITERS):
                    start = time.perf_counter()
                    model(x)
                    samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: (float(np.median(v)), float(np.percentile(v, 95))) for name, v in samples.items()}


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-config", default="configs/cnn_best.json")
    ap.add_argument("--tcn-config", default="configs/tcn.yaml")
    ap.add_argument("--transformer-config", default="configs/best/tiny_transformer.yaml")
    ap.add_argument("--fastpath", action="store_true", help="leave PyTorch's fused Transformer fast path on (default: off, so all models run plain eager ops)")
    args = ap.parse_args()

    # The fused fast path is not visible to the FLOP counter, so it is switched off for FLOPs and latency alike.
    fastpath_note = "MHA fast path left on"
    if not args.fastpath:
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            fastpath_note = "MHA fast path off"
        except AttributeError:
            fastpath_note = "WARNING: could not switch the MHA fast path off, Transformer FLOPs may be undercounted"
    print(f"git commit: {git_commit()}")
    print(f"torch {torch.__version__}  numpy {np.__version__}  cpu: {cpu_name()}  logical cores: {os.cpu_count()}  {fastpath_note}")

    models = build_models(args)
    x = torch.randn(INPUT_SHAPE, generator=torch.Generator().manual_seed(SEED))
    for name, model in models.items():
        assert model(x).shape == (1, 5), f"{name}: output shape is not (1, 5)"

    flops = {name: count_flops(model, x) for name, model in models.items()}
    expected = transformer_matmul_flops(models["tiny_transformer"])
    gap = abs(flops["tiny_transformer"] - expected) / expected
    check = (f"counter {flops['tiny_transformer']} vs hand count {expected} ({gap:.1%} apart)"
             + ("" if gap <= 0.01 else "  WARNING: more than 1% apart"))
    print("transformer FLOP cross-check:", check)

    latency = measure_latency(models, x)
    device = f"CPU {cpu_name()}, {THREADS} thread"
    method = (f"batch 1, input {INPUT_SHAPE}, eval, inference_mode, {THREADS} thread, {WARMUP} warm-up, "
              f"{ROUNDS}x{ITERS} timed passes per model (round-robin), median and p95; "
              f"flops = torch FlopCounterMode (2 per multiply-add; conv/matmul/attention only; {fastpath_note}; flops counted with math attention backend, latency uses the default backend); "
              f"size = serialized state_dict, decimal MB")
    notes = {"cnn": "", "tcn": "", "tiny_transformer": f"flop cross-check: {check}"}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "parameters", "flops", "model_size_mb", "cpu_latency_ms", "latency_p95_ms",
               "measurement_device", "measurement_method", "notes"]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, model in models.items():
            row = {"model": name, "parameters": count_params(model), "flops": flops[name],
                   "model_size_mb": round(state_dict_mb(model), 4), "cpu_latency_ms": round(latency[name][0], 4),
                   "latency_p95_ms": round(latency[name][1], 4), "measurement_device": device,
                   "measurement_method": method, "notes": notes[name]}
            writer.writerow(row)
            print(f"{name:12s} params={row['parameters']:>7d}  flops={row['flops']:>9d}  size={row['model_size_mb']:.4f} MB  "
                  f"latency median={row['cpu_latency_ms']:.3f} ms  p95={row['latency_p95_ms']:.3f} ms")
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()