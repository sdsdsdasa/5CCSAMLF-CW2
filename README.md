# 5CCSAMLF-CW2

KCL Machine Learning Coursework 2 — implementation of the **TPC RP** active learning algorithm from the paper *"Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets"*, evaluated on CIFAR-10, with two proposed modifications.

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
├── clustering.py   # K-means, uncovered cluster detection, typicality-based query selection
├── classifier.py   # Linear probe on frozen SimCLR encoder + uncertainty scoring
└── active_loop.py  # Full TPC RP active learning loop (supports warm-start and uncertainty weighting)

main.py                  # Original TPC RP experiment
main_warmstart.py        # Modification 1: warm-started SimCLR (500 + 100 fine-tune epochs)
main_uncertainty.py      # Modification 2: uncertainty-weighted typicality scoring
compare.py               # Compare 2+ result CSVs with table + plot
test_run.py              # Sanity-check run (3 SimCLR epochs, 1 AL iteration)
requirements.txt         # Python dependencies
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

**Quick sanity check** (~minutes):
```bash
python test_run.py
```

**Original TPC RP** (500 SimCLR epochs, 200 CLF epochs, 5 AL iterations):
```bash
python main.py
```

**Modification 1 — Warm-started SimCLR** (500 epochs iter 1, 100 fine-tune epochs iter 2+):
```bash
python main_warmstart.py
```

**Modification 2 — Uncertainty-weighted typicality** (score = typicality × classifier entropy):
```bash
python main_uncertainty.py
```

**Compare results:**
```bash
python compare.py            # interactive multi-select
python compare.py --all      # compare all CSVs under results/
```

All outputs saved to `results/<run-name>/`: CSV + accuracy curve PNG.

## Results

| Method | Final Accuracy (|L|=60) | vs Original |
|--------|------------------------|-------------|
| Original (O-500+200) | 77.43% | — |
| Warm-start (W-500+100+200) | 78.18% | +0.75 pp |
| Uncertainty-weighted (U-50+20) | 55.94% | −21.5 pp |

Warm-start reduces total SimCLR training by **64%** (2,500 → 900 epochs) with a mean accuracy gain of **+1.34 pp** across all iterations.

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
| Iterations | 5 (stopping at \|L\| = 60) |

> GPU required. Tested on RTX 4070 SUPER with CUDA 12.8.
