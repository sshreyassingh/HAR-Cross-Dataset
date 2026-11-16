cat > scripts/check_results_csv.py << 'EOF'
#!/usr/bin/env python3
"""Validate a final results CSV against the protocol-v1-sprint rules.

Expected for one model:
- Exactly 27 rows (9 within-dataset + 18 cross-dataset)
- Columns (in any order): model, source, target, seed, split, n_windows, n_params, acc, macro_f1, weighted_f1, kappa
- seeds 0,1,2 present
- n_windows and n_params filled (no NaN)
"""

import sys
import pandas as pd
from pathlib import Path

REQUIRED_COLS = {
    "model", "source", "target", "seed", "split",
    "n_windows", "n_params", "acc", "macro_f1", "weighted_f1", "kappa"
}

def main(csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    # 1. Columns
    missing = REQUIRED_COLS - set(df.columns)
    extra = set(df.columns) - REQUIRED_COLS
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}")
        return 1
    if extra:
        print(f"WARNING: unexpected extra columns: {sorted(extra)}")

    # 2. Row count
    if len(df) != 27:
        print(f"ERROR: expected exactly 27 rows, found {len(df)}")
        return 1
    print("OK: 27 rows")

    # 3. Seeds
    seeds = sorted(df["seed"].unique())
    if seeds != [0, 1, 2]:
        print(f"ERROR: expected seeds [0,1,2], found {seeds}")
        return 1
    print("OK: seeds 0,1,2 present")

    # 4. No missing values in key columns
    for col in ["n_windows", "n_params", "acc", "macro_f1"]:
        if df[col].isna().any():
            print(f"ERROR: NaN values found in column '{col}'")
            return 1
    print("OK: n_windows, n_params, acc, macro_f1 have no NaNs")

    # 5. Within vs cross counts
    within = df[df["source"] == df["target"]]
    cross = df[df["source"] != df["target"]]
    print(f"OK: {len(within)} within-dataset rows, {len(cross)} cross-dataset rows")
    if len(within) != 9 or len(cross) != 18:
        print("ERROR: expected 9 within + 18 cross")
        return 1

    print("\nAll checks passed.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 -m scripts.check_results_csv path/to/results.csv")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
EOF