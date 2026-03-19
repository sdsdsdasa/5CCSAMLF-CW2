# 5CCSAMLF-CW2

KCL Machine Learning Coursework 2 — implementation of the **TPC RP** active learning algorithm from the paper *"Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets"*, evaluated on CIFAR-10.

## Algorithm

**TPC RP** (TypiClust with Representation learning + clustering):
1. Train **SimCLR** (self-supervised) on all data to learn a feature space
2. Run **K-means** with `|L| + B` clusters; identify uncovered clusters (no labeled samples)
3. Query the most **typical** sample from each of the B largest uncovered clusters

Typicality = inverse of mean distance to K=20 nearest neighbours in embedding space.

## Structure

```
tpcrp/
├── dataset.py      # CIFAR-10 loading and augmentation pipelines
├── simclr.py       # SimCLR model, NT-Xent loss, training, embedding extraction
├── typicality.py   # KNN-based typicality scoring
├── clustering.py   # K-means, uncovered cluster detection, query selection
├── classifier.py   # Linear probe on frozen SimCLR encoder
└── active_loop.py  # Full TPC RP active learning loop

main.py             # Entry point — runs experiment, saves CSV + plot
requirements.txt    # Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Outputs saved to `results/`:
- `results.csv` — test accuracy at each AL iteration
- `accuracy_curve.png` — accuracy vs number of labeled examples

> For full accuracy, run on a GPU (Google Colab recommended). CIFAR-10 data is downloaded automatically to `data/`.
