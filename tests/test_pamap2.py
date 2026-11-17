"""Tests for src/data/pamap2.py.

Part 1 uses small synthetic recordings (constant or ramp signals) so each rule can be checked exactly.
Part 2 checks the real data/processed/pamap2_hand.npz and is skipped until that file exists.
Run:  python -m pytest -q -s tests/test_pamap2.py      (-s shows the printed counts)
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data import pamap2
from src.data.pamap2 import G, LABEL_MAP, MAX_NAN_RUN, build, build_from_arrays, clean_segment, load_file

REAL = Path("data/processed/pamap2_hand.npz")
needs_data = pytest.mark.skipif(not REAL.exists(), reason="data/processed/pamap2_hand.npz not built yet")


def recording(blocks, fs=100):
    """blocks: (activity_id, n_samples, constant in m/s^2 [, seconds of gap before the block]).
    Returns timestamps, activity ids and a (n, 3) accelerometer array."""
    ts, labels, accs, start = [], [], [], 0.0
    for b in blocks:
        act, n, const = b[:3]
        start += b[3] if len(b) > 3 else 0.0
        ts.append(start + np.arange(n) / fs)
        labels.append(np.full(n, act))
        accs.append(np.full((n, 3), float(const)))
        start = ts[-1][-1] + 1 / fs
    return np.concatenate(ts), np.concatenate(labels), np.concatenate(accs)


def windows_are_constant(X):
    return float((X.max(axis=(1, 2)) - X.min(axis=(1, 2))).max()) < 1e-4


# ---------------------------------------------------------------- part 1: synthetic checks
def test_label_mapping_and_no_window_crosses_activity_boundary():
    # 5 activities back to back, each a constant 1..5 g. A window that crossed a boundary would not be constant.
    ids = [4, 12, 13, 2, 3]                                  # walking, up, down, sitting, standing
    t, lab, acc = recording([(a, 600, (k + 1) * G) for k, a in enumerate(ids)])
    X, y = build_from_arrays(t, lab, acc, subject=101)
    assert X.shape == (15, 50, 3)                            # 600 samples -> 120 at 20 Hz -> 3 windows each
    assert windows_are_constant(X)
    assert (y == np.round(X.mean(axis=(1, 2))).astype(int) - 1).all()   # constant k+1 g must carry class k
    assert [LABEL_MAP[a] for a in ids] == [0, 1, 2, 3, 4]


def test_values_are_converted_to_g():
    t, lab, acc = recording([(4, 600, G)])
    X, _ = build_from_arrays(t, lab, acc, subject=101)
    assert np.allclose(X, 1.0, atol=1e-5)


def test_transient_and_other_activities_are_dropped():
    t, lab, acc = recording([(0, 600, G), (1, 600, G), (5, 600, G)])   # transient, lying, running
    X, y = build_from_arrays(t, lab, acc, subject=101)
    assert len(X) == 0 and len(y) == 0


def test_no_window_crosses_a_time_gap():
    # same activity on both sides of a 1 s timestamp jump, with different constants
    t, lab, acc = recording([(4, 600, 1 * G), (4, 600, 2 * G, 1.0)])
    X, _ = build_from_arrays(t, lab, acc, subject=101)
    assert X.shape[0] == 6 and windows_are_constant(X)
    assert set(np.round(X.mean(axis=(1, 2)))) == {1.0, 2.0}


def test_short_nan_run_is_interpolated():
    acc = np.linspace(0, 10, 300)[:, None] * np.ones((1, 3))
    clean = acc.copy()
    acc[100:110] = np.nan
    stats = Counter()
    pieces = clean_segment(acc, stats)
    assert len(pieces) == 1 and pieces[0].shape == (300, 3)
    assert np.allclose(pieces[0], clean)                     # linear interpolation of a ramp is exact
    assert stats["nan_runs_interpolated"] == 1 and stats["nan_samples_interpolated"] == 10


def test_nan_run_boundary_is_half_a_second():
    for run, expected_pieces in [(MAX_NAN_RUN, 1), (MAX_NAN_RUN + 1, 2)]:   # 50 samples filled, 51 cuts
        acc = np.full((600, 3), 5.0)
        acc[300:300 + run] = np.nan
        assert len(clean_segment(acc, Counter())) == expected_pieces


def test_long_nan_run_cuts_the_segment_and_no_window_spans_it():
    t, lab, _ = recording([(4, 660, 0.0)])
    acc = np.full((660, 3), 5.0)
    acc[360:] = 7.0
    acc[300:360] = np.nan                                    # 60 NaN samples = 0.6 s > 0.5 s
    X, _ = build_from_arrays(t, lab, acc, subject=101)
    assert X.shape[0] == 2 and windows_are_constant(X) and not np.isnan(X).any()
    assert np.allclose(sorted(X.mean(axis=(1, 2))), [5 / G, 7 / G], atol=1e-4)


def test_nan_at_segment_edges_is_trimmed_not_extrapolated():
    acc = np.full((300, 3), 5.0)
    acc[:5] = np.nan
    acc[-3:] = np.nan
    stats = Counter()
    pieces = clean_segment(acc, stats)
    assert len(pieces) == 1 and pieces[0].shape == (292, 3) and not np.isnan(pieces[0]).any()
    assert stats["edge_samples_trimmed"] == 8


def test_stride_is_25_and_windows_have_50_samples():
    rng = np.random.default_rng(0)
    t, lab, _ = recording([(4, 600, 0.0)])
    X, _ = build_from_arrays(t, lab, rng.normal(0, 3, (600, 3)), subject=101)
    assert X.shape[1] == 50 == pamap2.WIN and pamap2.STRIDE == 25
    assert np.array_equal(X[1:, :25], X[:-1, 25:])           # 50% overlap = stride 25


def test_output_dtypes_and_no_nan():
    rng = np.random.default_rng(1)
    t, lab, _ = recording([(2, 800, 0.0)])
    X, y = build_from_arrays(t, lab, rng.normal(0, 2, (800, 3)), subject=101)
    assert X.dtype == np.float32 and y.dtype == np.int64 and X.shape[1:] == (50, 3)
    assert np.isfinite(X).all() and set(y) == {3}


def write_dat(path, t, lab, acc):
    """Write a fake 54-column PAMAP2 file."""
    rows = np.zeros((len(t), 54))
    rows[:, 0], rows[:, 1], rows[:, 4:7] = t, lab, acc
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.6f")


def test_build_reads_protocol_folder_only_and_keeps_subjects_apart(tmp_path):
    root = tmp_path / "PAMAP2_Dataset"
    write_dat(root / "Protocol" / "subject101.dat", *recording([(4, 600, 1 * G)]))
    write_dat(root / "Protocol" / "subject102.dat", *recording([(4, 600, 3 * G)]))
    write_dat(root / "Optional" / "subject105.dat", *recording([(4, 600, 2 * G)]))   # must be ignored
    X, y, s, info = build(tmp_path)
    assert set(s) == {101, 102}
    assert np.allclose(X[s == 101], 1.0, atol=1e-4) and np.allclose(X[s == 102], 3.0, atol=1e-4)
    assert info["windows_per_subject"] == {101: 3, 102: 3}


def test_load_file_fails_loudly_on_wrong_rate_or_columns(tmp_path):
    t = np.arange(300) / 50                                   # 50 Hz instead of 100 Hz
    write_dat(tmp_path / "subject101.dat", t, np.full(300, 4), np.zeros((300, 3)))
    with pytest.raises(ValueError):
        load_file(tmp_path / "subject101.dat")
    (tmp_path / "subject102.dat").write_text("0.0 4 1 2 3\n0.01 4 1 2 3\n")
    with pytest.raises(ValueError):
        load_file(tmp_path / "subject102.dat")


# ---------------------------------------------------------------- part 2: the real processed file
def load_real():
    d = np.load(REAL, allow_pickle=False)
    return d["X"], d["y"], d["subject_id"], json.loads(str(d["meta"]))


@needs_data
def test_real_shapes_dtype_labels_and_no_nan():
    X, y, s, _ = load_real()
    assert X.ndim == 3 and X.shape[1:] == (50, 3) and X.dtype == np.float32
    assert len(X) == len(y) == len(s)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}


@needs_data
def test_real_subjects_and_counts():
    X, y, s, meta = load_real()
    print("\nwindows per subject:", {int(k): int((s == k).sum()) for k in np.unique(s)})
    print("windows per class:", {pamap2.CLASS_NAMES[k]: int((y == k).sum()) for k in range(5)})
    # subject 109 has no windows of the 5 classes in Protocol or Optional (docs/pamap2_audit.txt)
    assert set(np.unique(s)) == set(range(101, 109)), "expected subjects 101-108 (109 has no usable windows)"
    for subj in range(101, 109):
        assert set(y[s == subj]) == {0, 1, 2, 3, 4}, f"subject {subj} is missing a class"


@needs_data
def test_real_sitting_and_standing_are_near_1g():
    X, y, _, _ = load_real()
    mag = np.linalg.norm(X, axis=-1).mean(axis=1)
    for k in (3, 4):
        m = float(mag[y == k].mean())
        print(f"\nmean |a| of {pamap2.CLASS_NAMES[k]}: {m:.3f} g")
        assert 0.9 < m < 1.1


@needs_data
def test_real_values_are_raw_g_not_normalized():
    X, _, _, meta = load_real()
    assert meta["normalized"] is False and meta["units"] == "g" and meta["fs"] == 20
    assert 0.5 < float(np.linalg.norm(X, axis=-1).mean()) < 3.0   # m/s^2 would give about 10; z-scores about 1.7