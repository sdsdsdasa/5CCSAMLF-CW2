"""
TPC RP active learning loop.

Each iteration:
  1. Train SimCLR on all data (labeled + unlabeled) — 500 epochs, SGD lr=0.4
  2. Extract L2-normalised embeddings for all data
  3. K-means with n_clusters = min(|L| + B, 500)
  4. Iteratively select B samples: pick largest uncovered cluster (size > 5),
     compute per-cluster typicality with min(20, size) neighbours, query best
  5. Move selected from U → L
  6. Train fresh linear classifier on L — 200 epochs, SGD lr=2.5 (Nesterov)
  7. Evaluate on test set
"""

import numpy as np
import torch

from tpcrp.simclr import SimCLRModel, train_simclr, extract_embeddings
from tpcrp.clustering import run_kmeans, select_query_indices, MAX_CLUSTERS
from tpcrp.classifier import LinearClassifier, train_classifier, evaluate
from tpcrp.dataset import (make_simclr_loader, make_embed_loader,
                            make_classifier_loader)


def run_tpcrp(train_dataset, test_loader, device,
              budget=10,
              max_labeled=100,
              simclr_epochs=500,
              classifier_epochs=200,
              initial_labeled_idx=None,
              initial_unlabeled_idx=None,
              seed=42):
    """
    Full TPC RP active learning experiment.

    Args:
        train_dataset:         raw CIFAR-10 train dataset (no transform)
        test_loader:           DataLoader for test set evaluation
        device:                torch device
        budget:                B — samples queried per iteration
        max_labeled:           stop when |L| reaches this
        simclr_epochs:         epochs to train SimCLR each iteration (paper: 500)
        classifier_epochs:     epochs to train linear classifier (paper: 200)
        initial_labeled_idx:   list of initial labeled indices
        initial_unlabeled_idx: list of initial unlabeled indices
        seed:                  random seed

    Returns:
        results: list of dicts with keys 'n_labeled' and 'accuracy'
    """
    labeled_idx   = list(initial_labeled_idx)
    unlabeled_idx = list(initial_unlabeled_idx)
    results = []

    iteration = 0
    while len(labeled_idx) < max_labeled:
        iteration += 1
        print(f"\n=== AL Iteration {iteration} | Labeled: {len(labeled_idx)} ===")

        all_idx = labeled_idx + unlabeled_idx

        # --- Step 1: Train SimCLR from scratch ---
        print("  Training SimCLR...")
        simclr_model = SimCLRModel(proj_dim=128)
        simclr_loader = make_simclr_loader(train_dataset, all_idx, batch_size=512)
        simclr_model = train_simclr(simclr_model, simclr_loader,
                                    epochs=simclr_epochs, device=device)

        # --- Step 2: Extract L2-normalised embeddings ---
        print("  Extracting embeddings...")
        embed_loader = make_embed_loader(train_dataset, all_idx, batch_size=512)
        embeddings, _ = extract_embeddings(simclr_model, embed_loader, device)

        # --- Step 3: K-means (capped at max_clusters) ---
        n_clusters = min(len(labeled_idx) + budget, MAX_CLUSTERS)
        print(f"  Running K-means with {n_clusters} clusters...")
        cluster_labels = run_kmeans(embeddings, n_clusters, seed=seed)

        # --- Step 4: Select B queries iteratively ---
        unlabeled_positions = list(range(len(labeled_idx), len(all_idx)))
        selected_positions = select_query_indices(
            cluster_labels=cluster_labels,
            embeddings=embeddings,
            labeled_indices=labeled_idx,
            all_indices=all_idx,
            unlabeled_positions=unlabeled_positions,
            budget=budget,
        )
        selected_orig_idx = [all_idx[p] for p in selected_positions]
        print(f"  Selected {len(selected_orig_idx)} samples to query")

        # --- Step 5: Update labeled / unlabeled sets ---
        labeled_idx   = labeled_idx + selected_orig_idx
        unlabeled_idx = list(set(unlabeled_idx) - set(selected_orig_idx))

        # --- Step 6: Train fresh classifier (re-initialised weights) ---
        print("  Training classifier...")
        clf_loader = make_classifier_loader(train_dataset, labeled_idx,
                                            batch_size=min(128, len(labeled_idx)))
        classifier = LinearClassifier(simclr_model.encoder)
        classifier = train_classifier(classifier, clf_loader,
                                      epochs=classifier_epochs, device=device)

        # --- Step 7: Evaluate ---
        acc = evaluate(classifier, test_loader, device)
        print(f"  Test accuracy: {acc*100:.2f}%  (|L|={len(labeled_idx)})")
        results.append({'n_labeled': len(labeled_idx), 'accuracy': acc})

    return results
