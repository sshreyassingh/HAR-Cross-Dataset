cat > scripts/verify_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Quick sanity check for the three processed .npz files.

Checks:
- Files exist
- Required keys present
- Shape of X is (N, 50, 3)
- Subject counts match expected values
- Optional SHA-256 verification if you pass --hash
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED = {
    "uci": {
        "n_subjects": 30,
        "min_windows": 8000,
        "sha256": "a00a25802753e944b2d2912cc3d890aee45cfc4a48f07b12c25f136d83f775d2",
    },
    "wisdm": {
        "n_subjects": 36,
        "min_windows": 29000,
        "sha256": "25aa9ab426c0f1f4c662566059321e5a3c5c3d1772c8a4143882c2b65b5c7ac0",
    },
    "pamap2_hand": {
        "n_subjects": 8,          # updated 2026-11-16 (subject 109 has 0 windows)
        "min_windows": 6500,
        "sha256": None,           # fill in once Sharanya/Sanjana confirms
    },
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def check_one(name: str, path: Path, check_hash: bool) -> bool:
    print(f"\n=== {name} ===")
    if not path.exists():
        print(f"ERROR: missing {path}")
        return False

    data = np.load(path, allow_pickle=True)
    for key in ["X", "y", "subject_id", "meta"]:
        if key not in data:
            print(f"ERROR: missing key '{key}'")
            return False

    X = data["X"]
    y = data["y"]
    subject_id = data["subject_id"]

    print(f"X shape: {X.shape}  dtype: {X.dtype}")
    print(f"y shape: {y.shape}")
    print(f"subjects: {len(np.unique(subject_id))}")

    if X.ndim != 3 or X.shape[1:] != (50, 3):
        print(f"ERROR: expected X shape (N, 50, 3), got {X.shape}")
        return False

    exp = EXPECTED[name]
    n_subj = len(np.unique(subject_id))
    if n_subj != exp["n_subjects"]:
        print(f"ERROR: expected {exp['n_subjects']} subjects, found {n_subj}")
        return False
    if len(X) < exp["min_windows"]:
        print(f"ERROR: only {len(X)} windows (expected >= {exp['min_windows']})")
        return False

    if check_hash and exp["sha256"] is not None:
        got = sha256_file(path)
        if got != exp["sha256"]:
            print(f"ERROR: SHA-256 mismatch\n  expected {exp['sha256']}\n  got      {got}")
            return False
        print("SHA-256 OK")

    print("OK")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--hash", action="store_true", help="Also verify SHA-256")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ok = True
    for name in ["uci", "wisdm", "pamap2_hand"]:
        path = data_dir / f"{name}.npz"
        if not check_one(name, path, args.hash):
            ok = False

    if ok:
        print("\nAll datasets passed.")
        return 0
    print("\nSome checks failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
EOF