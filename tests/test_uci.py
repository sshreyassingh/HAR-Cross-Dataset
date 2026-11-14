"""Tests for the UCI-HAR pipeline. Run from the repo root:  python -m pytest -q tests/test_uci.py

The synthetic tests always run. The real-data tests are skipped until data/processed/uci.npz exists.
"""
import json
import os

import numpy as np
import pytest

from src.data import uci

REAL = "data/processed/uci.npz"


def make_windows_from_signal(sig):
    """Cut a continuous (L,3) signal into 128-sample windows with 50% overlap (the UCI layout)."""
    n = (len(sig) - 128) // 64 + 1
    return np.stack([sig[i * 64:i * 64 + 128] for i in range(n)])


def synthetic_dataset(offsets, n_windows=12):
    """Each (subject, label) block is a continuous constant-offset signal, so a window that mixes two
    blocks is detectable (its values would differ). offsets: list of (subject, label, constant)."""
    X, y, s = [], [], []
    for subj, label, const in offsets:
        L = 64 * (n_windows + 1)
        sig = np.full((L, 3), const, dtype=float)
        w = make_windows_from_signal(sig)
        X.append(w); y += [label] * len(w); s += [subj] * len(w)
    return np.concatenate(X), np.array(y), np.array(s)


def test_stitch_reconstructs_signal():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(64 * 9, 3))
    w = make_windows_from_signal(sig)                       # 8 windows
    assert np.allclose(uci.stitch(w, 0, len(w)), sig)


def test_segments_break_on_subject_and_label_change():
    rng = np.random.default_rng(2)
    blocks = [(1, 1), (1, 2), (2, 1)]                       # (subject, label) with random signals
    Xs, ys, ss = [], [], []
    for subj, label in blocks:
        w = make_windows_from_signal(rng.normal(size=(64 * 6, 3)))   # 5 windows each
        Xs.append(w); ys += [label] * len(w); ss += [subj] * len(w)
    X, y, s = np.concatenate(Xs), np.array(ys), np.array(ss)
    assert uci.find_segments(X, y, s) == [(0, 5), (5, 10), (10, 15)]


def test_segments_break_on_gap_same_subject_and_label():
    """Two separate bouts of the same subject and activity: only the missing overlap can break the chain."""
    rng = np.random.default_rng(5)
    a = make_windows_from_signal(rng.normal(size=(64 * 5, 3)))   # 4 windows
    b = make_windows_from_signal(rng.normal(size=(64 * 5, 3)))   # separate bout, same subject and label
    X = np.concatenate([a, b])
    y = np.full(len(X), 1)
    s = np.full(len(X), 7)
    assert uci.find_segments(X, y, s) == [(0, 4), (4, 8)]


def test_no_output_window_crosses_a_segment():
    offsets = [(1, 1, 1.0), (1, 4, 2.0), (2, 1, 3.0), (2, 5, 4.0)]
    X, y, s = synthetic_dataset(offsets)
    Xo, yo, so, info = uci.build_from_arrays(X, y, s, "stitch")
    assert Xo.shape[1:] == (50, 3) and Xo.dtype == np.float32
    assert info["n_segments"] == 4
    spread = Xo.max(axis=(1, 2)) - Xo.min(axis=(1, 2))
    assert spread.max() < 1e-3, "a window mixes two segments"
    # each window's value must match the constant of its own (subject, label) block
    expected = {(1, 0): 1.0, (1, 3): 2.0, (2, 0): 3.0, (2, 4): 4.0}
    for subj, lab in set(zip(so, yo)):
        vals = Xo[(so == subj) & (yo == lab)].mean(axis=(1, 2))
        assert np.allclose(vals, expected[(subj, lab)], atol=1e-3)


def test_window_count_formula():
    X, y, s = synthetic_dataset([(1, 1, 1.0)], n_windows=12)       # stitched length 64*13 = 832 -> 333 at 20 Hz (ceil(832*2/5))
    Xo, *_ = uci.build_from_arrays(X, y, s, "stitch")
    length_20hz = len(uci.to_20hz(np.zeros((64 * 13, 3))))
    assert len(Xo) == (length_20hz - 50) // 25 + 1


def test_laying_dropped_and_labels_mapped():
    offsets = [(1, k, float(k)) for k in range(1, 7)]
    X, y, s = synthetic_dataset(offsets)
    for method in ("stitch", "window"):
        Xo, yo, so, _ = uci.build_from_arrays(X, y, s, method)
        assert set(np.unique(yo)) == {0, 1, 2, 3, 4}
        assert Xo.shape[1:] == (50, 3)


def test_window_fallback_shapes():
    X, y, s = synthetic_dataset([(1, 4, 1.0), (2, 4, 1.0)], n_windows=5)
    Xo, yo, so, info = uci.build_from_arrays(X, y, s, "window")
    assert len(Xo) == len(X) and info["method"] == "window"
    assert np.abs(Xo - 1.0).max() < 1e-3                    # constant signal stays constant


def test_antialias_filter_passes_walking_and_removes_alias_band():
    t = np.arange(0, 20, 1 / 50)
    low = np.sin(2 * np.pi * 2.0 * t)[:, None]               # 2 Hz: walking band, must survive
    high = np.sin(2 * np.pi * 15.0 * t)[:, None]             # 15 Hz would alias to 5 Hz at 20 Hz
    a = uci.to_20hz(low)[50:-50]
    b = uci.to_20hz(high)[50:-50]
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    assert abs(rms(a) - 1 / np.sqrt(2)) < 0.02            # a sine has RMS 0.7071; it must be preserved
    assert rms(b) < 0.02                                   # the 15 Hz tone must be filtered out, not aliased


# ----------------------------- real data (skipped until the file exists) -----------------------------
needs_real = pytest.mark.skipif(not os.path.exists(REAL), reason="data/processed/uci.npz not built yet")


@needs_real
def test_real_file_schema():
    d = np.load(REAL)
    X, y, s = d["X"], d["y"], d["subject_id"]
    assert X.ndim == 3 and X.shape[1:] == (50, 3) and X.dtype == np.float32
    assert len(X) == len(y) == len(s)
    assert not np.isnan(X).any()
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}
    meta = json.loads(str(d["meta"]))
    assert meta["units"] == "g" and meta["normalized"] is False


@needs_real
def test_real_both_splits_chained():
    """Rows must chain into long segments in BOTH the train and the test files. If not, the pipeline silently
    behaved like the fallback (about 1 window per segment)."""
    meta = json.loads(str(np.load(REAL)["meta"]))
    if meta["method"] != "stitch":
        pytest.skip("window fallback was used")
    for name, info in zip(("train", "test"), meta["info"]["per_split"]):
        assert info["mean_windows_per_segment"] > 5, f"{name}: rows are not chaining ({info})"
        assert info["n_links"] == info["n_rows_in"] - info["n_segments"]


@needs_real
def test_real_subjects():
    s = np.load(REAL)["subject_id"]
    assert len(np.unique(s)) == 30, "UCI-HAR has 30 subjects"


@needs_real
def test_real_gravity_check_at_rest():
    d = np.load(REAL)
    mag = np.linalg.norm(d["X"], axis=-1).mean(axis=1)
    for k, name in ((3, "sitting"), (4, "standing")):
        m = mag[d["y"] == k].mean()
        assert 0.9 < m < 1.1, f"{name} mean magnitude {m:.3f} g is not near 1 g (wrong units?)"