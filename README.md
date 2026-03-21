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
source venv/Scripts/activate        # Windows Git Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically to `data/` on first run.

## Run

**Quick sanity check** (3 SimCLR epochs, 1 AL iteration, ~minutes):
```bash
python test_run.py
```
Outputs saved to `results/test/`: `results_test.csv`, `accuracy_curve_test.png`

**Full experiment** (500 SimCLR epochs × 6 AL iterations, requires GPU):
```bash
python main.py
```
Outputs saved to `results/`: `results.csv`, `accuracy_curve.png`

## Training Config (paper's Appendix F)

| Component | Setting |
|-----------|---------|
| SimCLR backbone | ResNet-18, 3×3 first conv, no maxpool |
| SimCLR training | SGD lr=0.4, momentum=0.9, cosine schedule, 500 epochs |
| Clustering | K-means, up to 500 clusters, min cluster size = 5 |
| Typicality | Inverse mean distance to K=20 nearest neighbours |
| Classifier | Frozen encoder + linear layer, SGD lr=2.5, Nesterov, 200 epochs |
| Initial labeled set | 1 sample per class → \|L₀\| = 10 |
| Budget per iteration | B = 10 |
| Iterations | 6 (stopping at \|L\| = 70) |

> A GPU is required for the full run. Tested on RTX 4070 SUPER with CUDA 12.8.
