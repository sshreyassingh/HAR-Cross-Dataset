"""Tests for src/models/tiny_transformer.py.  Run:  python -m pytest -q -s tests/test_tiny_transformer.py"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.tiny_transformer import TARGET_PARAMS, build_model, count_params


def make(heads=2, **cfg):
    torch.manual_seed(0)
    return build_model({"num_heads": heads, **cfg})


def test_output_is_5_logits_per_window():
    out = make()(torch.randn(7, 50, 3))
    assert out.shape == (7, 5) and out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(7), atol=1e-3)   # not softmax probabilities


def test_parameter_count_is_60k_to_90k_and_within_5_percent_of_75k():
    for heads in (2, 4):
        n = count_params(make(heads))
        print(f"\nheads={heads}: {n} parameters ({(n - TARGET_PARAMS) / TARGET_PARAMS:+.2%} vs 75,000)")
        assert 60_000 <= n <= 90_000
        assert abs(n - TARGET_PARAMS) / TARGET_PARAMS <= 0.05


def test_width_is_divisible_by_heads():
    for heads in (2, 4):
        assert make(heads).embed.out_features % heads == 0


def test_same_seed_gives_same_model_output():
    x = torch.randn(4, 50, 3, generator=torch.Generator().manual_seed(1))
    a, b = make().eval(), make().eval()
    assert torch.equal(a(x), b(x))


def test_build_model_ignores_keys_for_other_models():
    assert make(4, dropout=0.3, kernel_size=5)(torch.randn(2, 50, 3)).shape == (2, 5)


def test_wrong_input_shape_raises():
    with pytest.raises(ValueError):
        make()(torch.randn(2, 40, 3))


def test_gradients_reach_every_parameter():
    model = make().train()
    model(torch.randn(8, 50, 3)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_overfits_64_windows_with_the_shared_trainer(tmp_path):
    # Uses Sanjana's train_model (no private training loop). Skipped until her trainer is on this branch.
    trainer = pytest.importorskip("src.train.trainer", reason="shared trainer (and scikit-learn) not available yet")
    torch.manual_seed(0)
    y = torch.arange(64) % 5
    X = torch.randn(5, 1, 3)[y] * 2 + 0.1 * torch.randn(64, 50, 3)   # each class has its own offset
    loader = DataLoader(TensorDataset(X, y), batch_size=64)
    model = make(2, dropout=0.0)
    norm = {"mean": np.zeros((1, 1, 3), np.float32), "std": np.ones((1, 1, 3), np.float32)}
    trainer.train_model(model, loader, loader, torch.ones(5), norm, tmp_path / "ckpt.pt",
                        learning_rate=3e-3, max_epochs=300, patience=300, device="cpu")
    model.eval()
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == y).float().mean().item()
    print(f"\ntraining accuracy on 64 windows: {acc:.3f}")
    assert acc > 0.95