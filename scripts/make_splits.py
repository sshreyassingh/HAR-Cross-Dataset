"""Make fixed subject-wise splits (train/val/test) for each dataset.

Run:  python scripts/make_splits.py          (fake IDs, for testing today)
      python scripts/make_splits.py --real   (real IDs from data/processed/*.npz)
Output: splits/{domain}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEED = 0          # fixed by PROTOCOL.md
N_SPLITS = 3      # seed k uses split k

# (train, val, test) number of subjects, fixed by PROTOCOL.md
SIZES = {
    "uci":         (21, 4, 5),   # 30 subjects
    "wisdm":       (25, 5, 6),   # 36 subjects
    "pamap2_hand": (5, 1, 3),    # 9 subjects
}


def get_ids(domain, real):
    """Return the sorted list of subject ids for a domain."""
    expected = sum(SIZES[domain])
    if real:
        ids = np.unique(np.load(f"data/processed/{domain}.npz")["subject_id"])
    else:
        ids = np.arange(1, expected + 1)   # fake ids 1..N for testing
    assert len(ids) == expected, f"{domain}: expected {expected} subjects, got {len(ids)}"
    return sorted(int(i) for i in ids)


def make_splits(domain, ids):
    """Build the 3 splits for one domain."""
    n_train, n_val, n_test = SIZES[domain]
    rng = np.random.default_rng(SEED)   # fresh generator per domain
    splits = []

    if domain.startswith("pamap2"):
        # Shuffle once, then cut into 3 groups of 3 -> test sets never overlap
        # and together cover all 9 subjects exactly once.
        order = [int(s) for s in rng.permutation(ids)]
        for k in range(N_SPLITS):
            test = order[k * n_test:(k + 1) * n_test]
            rest = [s for s in ids if s not in test]
            val = [int(s) for s in rng.choice(rest, size=n_val, replace=False)]
            train = [s for s in rest if s not in val]
            splits.append({"split": k, "train": sorted(train),
                           "val": sorted(val), "test": sorted(test)})
    else:
        # UCI / WISDM: a new random shuffle for each split
        for k in range(N_SPLITS):
            order = [int(s) for s in rng.permutation(ids)]
            test = order[:n_test]
            val = order[n_test:n_test + n_val]
            train = order[n_test + n_val:]
            splits.append({"split": k, "train": sorted(train),
                           "val": sorted(val), "test": sorted(test)})
    return splits


def check(domain, ids, splits):
    """Fail loudly if any rule from PROTOCOL.md is broken."""
    n_train, n_val, n_test = SIZES[domain]
    assert len(splits) == N_SPLITS
    for s in splits:
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert (len(tr), len(va), len(te)) == (n_train, n_val, n_test), "wrong sizes"
        assert not (tr & va), "subject in both train and val"
        assert not (tr & te), "subject in both train and test"
        assert not (va & te), "subject in both val and test"
        assert tr | va | te == set(ids), "some subject is missing"
    if domain.startswith("pamap2"):
        all_test = sorted(sum((s["test"] for s in splits), []))
        assert all_test == sorted(ids), "PAMAP2 test sets must be disjoint and cover all subjects once"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use real subject ids from the .npz files")
    args = parser.parse_args()

    Path("splits").mkdir(exist_ok=True)
    for domain in SIZES:
        ids = get_ids(domain, args.real)
        splits = make_splits(domain, ids)
        check(domain, ids, splits)
        out = {"domain": domain, "seed": SEED, "splits": splits}
        with open(f"splits/{domain}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"{domain}: OK ({len(ids)} subjects)")
        for s in splits:
            print(f"  split {s['split']}: test={s['test']}")


if __name__ == "__main__":
    main()
