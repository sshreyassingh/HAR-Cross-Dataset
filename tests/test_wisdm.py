import numpy as np
d = np.load("data/processed/wisdm.npz")
X, y, sid = d["X"], d["y"], d["subject_id"]

def test_shape_dtype():
    assert X.ndim == 3 and X.shape[1:] == (50, 3) and X.dtype == np.float32
    assert len(X) == len(y) == len(sid)
def test_no_nan():
    assert not np.isnan(X).any()
def test_labels():
    assert set(np.unique(y)) <= {0, 1, 2, 3, 4}
def test_subjects():
    assert 1 <= sid.min() and sid.max() <= 36
def test_rest_is_1g():
    m = np.linalg.norm(X[(y == 3) | (y == 4)], axis=2).mean()
    print("sitting/standing mean magnitude (g):", m)
    assert 0.9 < m < 1.1
def test_class_table():
    print({int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
