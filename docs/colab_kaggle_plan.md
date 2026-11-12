# Colab / Kaggle Plan (protocol-v1-sprint)

**Rule:** Code lives in GitHub. Data and large outputs live in shared storage. Notebooks stay thin (clone → install → run script → save results).

## Shared storage
- One Google Drive folder `har-project/` shared with all four members.
  - Contains `data/raw/` and `outputs/`.
- Upload UCI-HAR, WISDM and PAMAP2 once so nobody re-downloads.
- For Kaggle: upload the same data as a **private Kaggle Dataset**.
- Never put the shared data folder inside the git repo.

## Which platform for what
| Task                          | Platform                                      |
|-------------------------------|-----------------------------------------------|
| Data prep, debugging, smoke   | Colab                                         |
| Tuning + final training runs  | Kaggle (Save & Run All survives tab close)    |
| Analysis & figures            | Colab or local                                |

## Start of every notebook
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/sshreyassingh/HAR-Cross-Dataset.git
%cd HAR-Cross-Dataset
!git checkout <your-branch-name>
!pip install -q -r requirements.txt

# Point to shared data (do not re-download)
!rm -rf data/raw
!ln -s /content/drive/MyDrive/har-project/data/raw data/raw

!pytest -q tests/test_smoke.py