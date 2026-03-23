"""
compare.py — Compare two TPC RP result CSVs.

Usage:
    # Pick interactively from discovered results:
    python compare.py

    # Pass paths directly:
    python compare.py results/50+20epchos/results.csv results/warmstart/results_warmstart.csv

    # Optionally give custom labels:
    python compare.py results/50+20epchos/results.csv results/warmstart/results_warmstart.csv \
        --labels "Original (50 ep)" "Warm-Start (10 ep)"

Output:
    - Printed comparison table
    - results/comparison.png
"""

import os
import csv
import sys
import glob
import argparse
import matplotlib.pyplot as plt


def find_csvs(root='./results'):
    """Recursively find all CSV files under results/."""
    return sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True))


def load_csv(path):
    """Return (n_labeled list, accuracy_pct list) from a results CSV."""
    n_labeled, accuracies = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            n_labeled.append(int(row['n_labeled']))
            accuracies.append(float(row['accuracy']) * 100)
    return n_labeled, accuracies


def pick_csv(csvs, prompt):
    """Interactive numbered menu to pick a CSV."""
    print(f"\n{prompt}")
    for i, path in enumerate(csvs):
        print(f"  [{i}] {path}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 0 <= choice < len(csvs):
                return csvs[choice]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")


def print_table(n_a, acc_a, label_a, n_b, acc_b, label_b):
    """Print side-by-side comparison table."""
    all_n = sorted(set(n_a) | set(n_b))
    map_a = dict(zip(n_a, acc_a))
    map_b = dict(zip(n_b, acc_b))

    col = 14
    print()
    print(f"{'|L|':>5}  {label_a:>{col}}  {label_b:>{col}}  {'Delta':>9}")
    print("-" * (5 + 2 + col + 2 + col + 2 + 9))
    for n in all_n:
        a = map_a.get(n)
        b = map_b.get(n)
        a_str = f"{a:.2f}%" if a is not None else "     -    "
        b_str = f"{b:.2f}%" if b is not None else "     -    "
        if a is not None and b is not None:
            delta = b - a
            d_str = f"{delta:+.2f} pp"
        else:
            d_str = "     -"
        print(f"{n:>5}  {a_str:>{col}}  {b_str:>{col}}  {d_str:>9}")
    print()

    # Summary
    shared = [n for n in all_n if n in map_a and n in map_b]
    if shared:
        last = shared[-1]
        delta_final = map_b[last] - map_a[last]
        avg_delta = sum(map_b[n] - map_a[n] for n in shared) / len(shared)
        wins_b = sum(1 for n in shared if map_b[n] > map_a[n])
        print(f"  Final |L|={last}: {label_b} vs {label_a}  ->  {delta_final:+.2f} pp")
        print(f"  Average delta across {len(shared)} shared points: {avg_delta:+.2f} pp")
        print(f"  {label_b} wins {wins_b}/{len(shared)} iterations")


def main():
    parser = argparse.ArgumentParser(description="Compare two TPC RP result CSVs.")
    parser.add_argument('csv_a', nargs='?', help='First CSV path (baseline)')
    parser.add_argument('csv_b', nargs='?', help='Second CSV path (to compare)')
    parser.add_argument('--labels', nargs=2, metavar=('LABEL_A', 'LABEL_B'),
                        help='Legend labels for the two curves')
    parser.add_argument('--out', default='results/comparison.png',
                        help='Output plot path (default: results/comparison.png)')
    args = parser.parse_args()

    csvs = find_csvs()
    if not csvs:
        print("No CSV files found under results/. Run main.py or main_warmstart.py first.")
        sys.exit(1)

    # Resolve paths
    if args.csv_a and args.csv_b:
        path_a, path_b = args.csv_a, args.csv_b
    else:
        print("Available result files:")
        path_a = pick_csv(csvs, "Select FIRST result (baseline):")
        path_b = pick_csv(csvs, "Select SECOND result (to compare):")

    def auto_label(path):
        parts = path.replace('\\', '/').split('/')
        return parts[-2] if len(parts) >= 2 else path

    label_a = args.labels[0] if args.labels else auto_label(path_a)
    label_b = args.labels[1] if args.labels else auto_label(path_b)

    n_a, acc_a = load_csv(path_a)
    n_b, acc_b = load_csv(path_b)

    print_table(n_a, acc_a, label_a, n_b, acc_b, label_b)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_a, acc_a, marker='s', linewidth=2, linestyle='--', label=label_a)
    plt.plot(n_b, acc_b, marker='o', linewidth=2, label=label_b)
    plt.xlabel('Number of Labeled Examples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('TPC RP — Result Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved to {args.out}")
    plt.show()


if __name__ == '__main__':
    main()
