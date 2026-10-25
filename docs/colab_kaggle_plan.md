# Shared Colab / Kaggle plan

**Rule:** code lives in GitHub, data and results live in shared storage, notebooks stay thin (clone, install, run, save).

## Shared storage
- One Google Drive folder `har-project/` shared with all four members, containing `data/raw/` and `outputs/`.
- Upload WISDM and PAMAP2 there once so nobody downloads them twice.
- For Kaggle, upload the same data as a **private Kaggle Dataset**.
- Never put the shared data folder inside the git repo.

## Which platform for what
| Task | Where |
|---|---|
| Data prep, splits, debugging, smoke test | Colab |
| Long training (tuning and final runs) | Kaggle ("Save & Run All" keeps running after you close the tab) |
| Analysis and figures | Colab or local |

Check the current GPU quota and session limits on both before planning, since free-tier limits change.

## Start of every notebook
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/sshreyassingh/HAR-Cross-Dataset.git
%cd HAR-Cross-Dataset
!git checkout member/<yourname>
!pip install -q -r requirements.txt

# point the repo at the shared data instead of re-downloading
!rm -rf data/raw
!ln -s /content/drive/MyDrive/har-project/data/raw data/raw

!pytest -q tests/test_smoke.py
```
Each member first adds a shortcut to the shared `har-project` folder in their own My Drive (right-click it under "Shared with me" → Organize → Add shortcut).

## Splitting the work (from the protocol: 117 training runs in total)
- Each member trains their own model: tuning on the 3 core sources, then final runs on the 3 core sources plus the 2 extra PAMAP2 sources (best PAMAP2-hand config, no tuning), seeds 0, 1, 2.
- Then test every trained model on the other 4 domains.
- One model per person so work never duplicates.

## After every run
- Copy the result CSVs to the shared Drive `outputs/` folder right away, because sessions can die.
- Commit only the small CSV tables to GitHub, never checkpoints or raw data.

## 15-day checkpoints
| Days | Goal |
|---|---|
| 1-2 | Repo ready, smoke test passes for everyone, datasets downloaded to shared storage |
| 3-5 | Preprocessing to `data/processed/{domain}.npz`, subject-wise splits, label mapping decided |
| 6-8 | Models built (about 75k params each), tuning runs |
| 9-12 | Final runs (seeds 0, 1, 2) and all cross-dataset evaluations |
| 13-14 | Drop analysis, figures, tables |
| 15 | Paper polish, final clean run |
