"""
test_run.py — Quick end-to-end test of the TPC RP pipeline.

Uses minimal epochs and only 2 AL iterations to verify everything
works without waiting for a full training run.

Usage:
    python test_run.py

Outputs saved to results/test/:
    results_test.csv
    accuracy_curve_test.png
"""

import os
import csv
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tpcrp.dataset import get_cifar10, initial_labeled_set
from tpcrp.active_loop import run_tpcrp

# ── Test config (intentionally low for speed) ────────────────────────────────
BUDGET          = 10   # samples queried per AL iteration
MAX_LABELED     = 30   # 10 initial + 2 iterations → quick 2-iteration test
SIMCLR_EPOCHS   = 3    # low epochs just to verify the pipeline runs
CLF_EPOCHS      = 5
SEED            = 42
DATA_DIR        = './data'
OUT_DIR         = './results/test'
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading CIFAR-10...")
    train_ds, test_ds = get_cifar10(DATA_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=2)

    labeled_idx, unlabeled_idx = initial_labeled_set(
        train_ds, n_per_class=1, seed=SEED)
    print(f"Initial labeled: {len(labeled_idx)} | Unlabeled: {len(unlabeled_idx)}")

    results = run_tpcrp(
        train_dataset=train_ds,
        test_loader=test_loader,
        device=device,
        budget=BUDGET,
        max_labeled=MAX_LABELED,
        simclr_epochs=SIMCLR_EPOCHS,
        classifier_epochs=CLF_EPOCHS,
        initial_labeled_idx=labeled_idx,
        initial_unlabeled_idx=unlabeled_idx,
        seed=SEED,
    )

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'results_test.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_labeled', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    # Save plot
    n_labeled  = [r['n_labeled']      for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(n_labeled, accuracies, marker='o', linewidth=2, label='TPC RP (test run)')
    plt.xlabel('Number of Labeled Examples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('TPC RP — Test Run (low epochs, sanity check only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'accuracy_curve_test.png')
    plt.savefig(plot_path, dpi=150)

    print()
    print("=== TEST RUN COMPLETE ===")
    for r in results:
        print(f"  |L|={r['n_labeled']:3d}  acc={r['accuracy']*100:.2f}%")
    print(f"\nCSV  -> {csv_path}")
    print(f"Plot -> {plot_path}")
    print("\nAll components working. Run main.py for the full experiment.")
