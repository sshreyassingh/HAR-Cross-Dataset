# Cross-Dataset Degradation in Human Activity Recognition

**Goal:** Measure and explain why human activity recognition (HAR) models lose accuracy when tested on a different dataset than they were trained on, using UCI-HAR, WISDM and PAMAP2 with CNN, TCN and Tiny Transformer models.

> Status: work in progress. Paper link / arXiv: _TBD_

## Team

| Name | GitHub |
|---|---|
| Shreya Singh | [@sshreyassingh](https://github.com/sshreyassingh) |
| Shanaya Dhamija | [@shanaya006](https://github.com/shanaya006) |
| Sharanya Gupta | _add handle_ |
| Sanjana Prasad | _add handle_ |

## Folder layout

| Folder | Purpose |
|---|---|
| `configs/` | YAML configs (`base.yaml`, `datasets.yaml`, one file per experiment) |
| `data/raw/` | Raw datasets. **Not tracked by git.** |
| `data/processed/` | Windowed data. **Not tracked by git.** |
| `splits/` | Subject-wise train/val/test split files (small, tracked) |
| `src/` | Code: data loading, models, training, evaluation, utilities |
| `experiments/` | One folder per experiment run (config copy and logs) |
| `results/` | Final result tables used in the paper |
| `figures/` | Final figures used in the paper |
| `tests/` | Smoke test and unit tests |
| `docs/` | Decision log and Colab/Kaggle plan |

## Quick start

Run these in a Google Colab (or Kaggle) notebook, one cell at a time.

**1. Clone and switch to your branch**
```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!git checkout member/<yourname>
```

**2. Install requirements**
```python
!pip install -q -r requirements.txt
```
For a quick check you only need `!pip install -q pyyaml pytest`.

**3. Download UCI-HAR**
```python
!mkdir -p data/raw/uci_har
!wget -q -O data/raw/uci_har/uci_har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
!unzip -q data/raw/uci_har/uci_har.zip -d data/raw/uci_har
!ls data/raw/uci_har
```
If you do not see a folder called `UCI HAR Dataset`, there is an inner zip: run
`!unzip -q "data/raw/uci_har/UCI HAR Dataset.zip" -d data/raw/uci_har`.

**4. Run the smoke test**
```python
!pytest -q tests/test_smoke.py
```
Everyone should see it pass before running anything bigger.

## Get the data

| Dataset | How to get it |
|---|---|
| UCI-HAR | Download with the commands above (Colab/Kaggle) |
| WISDM | Download manually and place in `data/raw/wisdm/` |
| PAMAP2 | Download manually and place in `data/raw/pamap2/` |

The exact dataset versions we use are recorded in `docs/DECISIONS.md`.
**Raw data is never committed to git.** The `.gitignore` blocks it. Keep a shared copy in the team's Google Drive folder or a private Kaggle Dataset (see `docs/colab_kaggle_plan.md`).

## How we work

- **Branches:** `main` always stays runnable and nobody commits to it directly. Each member works on their own branch named `member/<name>`.
- **Pull requests:** open a PR into `main`, get one teammate to review it, make sure the smoke test passes, then squash and merge. After a merge, everyone updates their branch from `main`.
- **Decisions:** every choice we make (label mapping, window size, sensors, dataset versions) goes into the [decision log](docs/DECISIONS.md).
- **Compute:** see the [Colab/Kaggle plan](docs/colab_kaggle_plan.md).
- **Rules:** run every experiment from a YAML config; use subject-wise splits only; fit normalisation on training data only; never commit raw data, checkpoints or secrets.

## Citation

```bibtex
@misc{har_cross_dataset_2026,
  title  = {TBD},
  author = {Singh, Shreya and Dhamija, Shanaya and Gupta, Sharanya and Prasad, Sanjana},
  year   = {2026}
}
```

## License

Code license: _TBD (MIT is a common choice; add a `LICENSE` file)_.
Paper text and figures: _TBD (CC BY 4.0 once published)_.
