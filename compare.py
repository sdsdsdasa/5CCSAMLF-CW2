"""
compare.py — Compare TPC RP result CSVs (2 or more).

Usage:
    # Interactive multi-select from discovered results:
    python compare.py

    # Compare all CSVs found under results/ automatically:
    python compare.py --all

    # Pass 2+ paths directly:
    python compare.py results/50+20epchos/results.csv results/warmstart/results_warmstart.csv

    # Optionally give custom labels (must match number of CSVs):
    python compare.py results/50+20epchos/results.csv results/warmstart/results_warmstart.csv \
        --labels "Original (50 ep)" "Warm-Start (100 ep)"

Output:
    - Printed comparison table (deltas relative to first/baseline curve)
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


def auto_label(path):
    parts = path.replace('\\', '/').split('/')
    return parts[-2] if len(parts) >= 2 else path


def pick_csvs(csvs):
    """Interactive numbered menu — enter comma-separated indices or 'all'."""
    print("\nAvailable result files:")
    for i, path in enumerate(csvs):
        print(f"  [{i}] {path}")
    print("\nEnter indices to compare (comma-separated, e.g. 0,2,3) or 'all':")
    while True:
        try:
            raw = input("> ").strip()
            if raw.lower() == 'all':
                return list(csvs)
            indices = [int(x.strip()) for x in raw.split(',')]
            if len(indices) < 2:
                print("  Select at least 2 files.")
                continue
            if all(0 <= i < len(csvs) for i in indices):
                return [csvs[i] for i in indices]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid input, try again.")


def print_table(curves):
    """
    Print comparison table for N curves.
    curves: list of (label, n_list, acc_list)
    Deltas are shown relative to the first curve (baseline).
    """
    all_n = sorted(set(n for _, ns, _ in curves for n in ns))
    maps  = [dict(zip(ns, accs)) for _, ns, accs in curves]
    labels = [label for label, _, _ in curves]

    col = 13
    header = f"{'|L|':>5}" + "".join(f"  {lbl:>{col}}" for lbl in labels)
    if len(curves) > 1:
        for lbl in labels[1:]:
            header += f"  {'Δ vs base':>10}"
    print()
    print(header)
    print("-" * len(header))

    for n in all_n:
        row = f"{n:>5}"
        vals = [m.get(n) for m in maps]
        for v in vals:
            row += f"  {f'{v:.2f}%' if v is not None else '-':>{col}}"
        base = vals[0]
        for v in vals[1:]:
            if base is not None and v is not None:
                row += f"  {v - base:>+9.2f} pp"
            else:
                row += f"  {'  -':>10}"
        print(row)
    print()

    # Summary stats vs baseline
    if len(curves) > 1:
        base_map = maps[0]
        for label, _, _ in curves[1:]:
            i = labels.index(label)
            shared = [n for n in all_n if n in base_map and n in maps[i]]
            if not shared:
                continue
            last = shared[-1]
            delta_final = maps[i][last] - base_map[last]
            avg_delta   = sum(maps[i][n] - base_map[n] for n in shared) / len(shared)
            wins        = sum(1 for n in shared if maps[i][n] > base_map[n])
            print(f"  [{label}] vs [{labels[0]}]:")
            print(f"    Final |L|={last}: {delta_final:+.2f} pp")
            print(f"    Avg delta over {len(shared)} points: {avg_delta:+.2f} pp")
            print(f"    Wins {wins}/{len(shared)} iterations")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare 2+ TPC RP result CSVs.")
    parser.add_argument('csvs', nargs='*', help='CSV paths to compare')
    parser.add_argument('--all', action='store_true',
                        help='Load every CSV found under results/ automatically')
    parser.add_argument('--labels', nargs='+', metavar='LABEL',
                        help='Legend labels (must match number of CSVs)')
    parser.add_argument('--out', default='results/comparison.png',
                        help='Output plot path (default: results/comparison.png)')
    args = parser.parse_args()

    discovered = find_csvs()
    if not discovered:
        print("No CSV files found under results/. Run an experiment first.")
        sys.exit(1)

    # Resolve which CSVs to use
    if args.all:
        paths = discovered
    elif args.csvs:
        paths = args.csvs
    else:
        paths = pick_csvs(discovered)

    if len(paths) < 2:
        print("Need at least 2 CSVs to compare.")
        sys.exit(1)

    # Assign labels
    if args.labels:
        if len(args.labels) != len(paths):
            print(f"--labels count ({len(args.labels)}) must match CSV count ({len(paths)}).")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(p) for p in paths]

    # Load data
    curves = []
    for path, label in zip(paths, labels):
        ns, accs = load_csv(path)
        curves.append((label, ns, accs))
        print(f"  Loaded [{label}]: {path}  ({len(ns)} points)")

    print_table(curves)

    # Plot
    markers = ['s', 'o', '^', 'D', 'v', 'P', 'X', '*']
    linestyles = ['--', '-', '-.', ':', '--', '-', '-.', ':']

    plt.figure(figsize=(8, 5))
    for i, (label, ns, accs) in enumerate(curves):
        plt.plot(ns, accs,
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=2, label=label)

    plt.xlabel('Number of Labeled Examples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('TPC RP — Result Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")
    plt.show()


if __name__ == '__main__':
    main()
