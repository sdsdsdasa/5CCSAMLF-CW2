"""
TPC RP — main entry point.

Runs the TPC RP active learning experiment on CIFAR-10 and saves:
  - results/results.csv     accuracy at each AL iteration
  - results/accuracy_curve.png  accuracy vs number of labeled samples
"""

import os
import csv
import torch
import matplotlib.pyplot as plt

from tpcrp.dataset import get_cifar10, initial_labeled_set
from tpcrp.active_loop import run_tpcrp


def main():
    # ── Config (paper's actual settings, Appendix F) ────────────────────────
    BUDGET          = 10      # B — samples queried per AL iteration
    MAX_LABELED     = BUDGET*6     # stop when |L| reaches this (paper: 6 iternations)
    INITIAL_PER_CLS = 1       # 1 labeled example per class → |L0| = 10
    SIMCLR_EPOCHS   = 50     # paper: SimCLR trained for 500 epochs per iteration
    CLF_EPOCHS      = 20     # paper: linear eval uses 2x base epochs (200)
    SEED            = 42
    DATA_DIR        = './data'
    OUT_DIR         = './results'
    # ────────────────────────────────────────────────────────────────────────

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    train_dataset, test_dataset = get_cifar10(DATA_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # Initial labeled / unlabeled split
    labeled_idx, unlabeled_idx = initial_labeled_set(
        train_dataset, n_per_class=INITIAL_PER_CLS, seed=SEED)
    print(f"Initial labeled: {len(labeled_idx)} | Unlabeled: {len(unlabeled_idx)}")

    # Run TPC RP
    results = run_tpcrp(
        train_dataset=train_dataset,
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
    csv_path = os.path.join(OUT_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_labeled', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Plot
    n_labeled  = [r['n_labeled']      for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(n_labeled, accuracies, marker='o', linewidth=2, label='TPC RP')
    plt.xlabel('Number of Labeled Examples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('TPC RP Active Learning on CIFAR-10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'accuracy_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
