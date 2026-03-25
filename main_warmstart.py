"""
TPC RP — Warm-Start SimCLR modification (Task 3).

Identical to main.py except SimCLR is fine-tuned from the previous
iteration's checkpoint (10 epochs) instead of retrained from scratch
(500 epochs) on iterations 2+.

Total SimCLR epochs: 500 + 4x100 = 900  (vs 50x5 = 2500 in original)

Saves results to results/warmstart/:
    results_warmstart.csv
    accuracy_curve_warmstart.png
"""

import os
import csv
import time
import torch
import matplotlib.pyplot as plt

from tpcrp.dataset import get_cifar10, initial_labeled_set
from tpcrp.active_loop import run_tpcrp


def main():
    # ── Config ──────────────────────────────────────────────────────────────
    BUDGET          = 10
    MAX_LABELED     = BUDGET * 6      # 60 — matches original run
    INITIAL_PER_CLS = 1               # |L0| = 10
    SIMCLR_EPOCHS   = 500             # iteration 1: train from scratch
    WARMUP_EPOCHS   = 100             # iterations 2+: fine-tune from checkpoint
    CLF_EPOCHS      = 200
    SEED            = 42
    DATA_DIR        = './data'
    OUT_DIR         = './results/warmstart'
    # ────────────────────────────────────────────────────────────────────────

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading CIFAR-10...")
    train_dataset, test_dataset = get_cifar10(DATA_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=2)

    labeled_idx, unlabeled_idx = initial_labeled_set(
        train_dataset, n_per_class=INITIAL_PER_CLS, seed=SEED)
    print(f"Initial labeled: {len(labeled_idx)} | Unlabeled: {len(unlabeled_idx)}")
    print(f"Warm-start config: {SIMCLR_EPOCHS} epochs (iter 1), "
          f"{WARMUP_EPOCHS} epochs (iters 2+)")

    t0 = time.time()
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
        warmup_epochs=WARMUP_EPOCHS,
    )
    total_time = time.time() - t0
    print(f"\nTotal wall-clock time: {total_time/60:.1f} min")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'results_warmstart.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_labeled', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

    # Plot — overlay original results if available
    n_labeled  = [r['n_labeled']      for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]

    plt.figure(figsize=(8, 5))

    # Try to load original results for comparison
    orig_csv = './results/500+200epchos/results.csv'
    if os.path.exists(orig_csv):
        import csv as csv_mod
        orig_n, orig_acc = [], []
        with open(orig_csv) as f:
            for row in csv_mod.DictReader(f):
                orig_n.append(int(row['n_labeled']))
                orig_acc.append(float(row['accuracy']) * 100)
        plt.plot(orig_n, orig_acc, marker='s', linewidth=2,
                 linestyle='--', label='TPC RP (original, 500 epochs/iter)')

    plt.plot(n_labeled, accuracies, marker='o', linewidth=2,
             label=f'TPC RP (warm-start, {SIMCLR_EPOCHS}+{WARMUP_EPOCHS} epochs)')
    plt.xlabel('Number of Labeled Examples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('TPC RP: Original vs Warm-Started SimCLR on CIFAR-10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'accuracy_curve_warmstart.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Print comparison table
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'|L|':>5}  {'Warm-Start':>12}  {'Original':>10}  {'Delta':>8}")
    orig_map = {}
    if os.path.exists(orig_csv):
        import csv as csv_mod
        with open(orig_csv) as f:
            for row in csv_mod.DictReader(f):
                orig_map[int(row['n_labeled'])] = float(row['accuracy']) * 100
    for r in results:
        nl  = r['n_labeled']
        ws  = r['accuracy'] * 100
        ori = orig_map.get(nl, float('nan'))
        delta = ws - ori if nl in orig_map else float('nan')
        print(f"{nl:>5}  {ws:>11.2f}%  {ori:>9.2f}%  {delta:>+7.2f} pp")
    print(f"\nTotal time: {total_time/60:.1f} min")


if __name__ == '__main__':
    main()
