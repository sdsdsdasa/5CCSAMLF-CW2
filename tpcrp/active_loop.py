"""
TPC RP active learning loop.

Each iteration:
  1. Train SimCLR on all data (labeled + unlabeled)
  2. Extract embeddings for all data
  3. Compute typicality scores
  4. K-means with n_clusters = |L| + B
  5. Find uncovered clusters
  6. Select B most typical samples from B largest uncovered clusters
  7. Move selected from U → L
  8. Train classifier on L, evaluate on test set
"""

import copy
import numpy as np
import torch

from tpcrp.simclr import SimCLRModel, train_simclr, extract_embeddings
from tpcrp.typicality import compute_typicality
from tpcrp.clustering import run_kmeans, find_uncovered_clusters, select_query_indices
from tpcrp.classifier import LinearClassifier, train_classifier, evaluate
from tpcrp.dataset import (make_simclr_loader, make_embed_loader,
                            make_classifier_loader)


def run_tpcrp(train_dataset, test_loader, device,
              budget=10,
              max_labeled=100,
              simclr_epochs=100,
              classifier_epochs=100,
              initial_labeled_idx=None,
              initial_unlabeled_idx=None,
              seed=42):
    """
    Full TPC RP active learning experiment.

    Args:
        train_dataset:        raw CIFAR-10 train dataset (no transform)
        test_loader:          DataLoader for test set evaluation
        device:               torch device
        budget:               B — samples queried per iteration
        max_labeled:          stop when |L| reaches this
        simclr_epochs:        epochs to train SimCLR each iteration
        classifier_epochs:    epochs to train linear classifier
        initial_labeled_idx:  list of initial labeled indices
        initial_unlabeled_idx: list of initial unlabeled indices
        seed:                 random seed

    Returns:
        results: list of dicts with keys 'n_labeled' and 'accuracy'
    """
    labeled_idx   = list(initial_labeled_idx)
    unlabeled_idx = list(initial_unlabeled_idx)
    results = []

    iteration = 0
    while len(labeled_idx) < max_labeled:
        iteration += 1
        n_labeled = len(labeled_idx)
        print(f"\n=== AL Iteration {iteration} | Labeled: {n_labeled} ===")

        # All indices used for SimCLR and clustering (labeled + unlabeled)
        all_idx = labeled_idx + unlabeled_idx

        # --- Step 1: Train SimCLR ---
        print("  Training SimCLR...")
        simclr_model = SimCLRModel(proj_dim=128)
        simclr_loader = make_simclr_loader(train_dataset, all_idx,
                                           batch_size=256)
        simclr_model = train_simclr(simclr_model, simclr_loader,
                                    epochs=simclr_epochs, device=device)

        # --- Step 2: Extract embeddings for all data ---
        print("  Extracting embeddings...")
        embed_loader = make_embed_loader(train_dataset, all_idx, batch_size=512)
        embeddings, _ = extract_embeddings(simclr_model, embed_loader, device)

        # --- Step 3: Typicality scores ---
        print("  Computing typicality scores...")
        typicality = compute_typicality(embeddings, k=20)

        # --- Step 4: K-means ---
        n_clusters = len(labeled_idx) + budget
        print(f"  Running K-means with {n_clusters} clusters...")
        cluster_labels = run_kmeans(embeddings, n_clusters, seed=seed)

        # --- Step 5: Uncovered clusters ---
        # Positions of labeled samples within all_idx
        labeled_positions = list(range(len(labeled_idx)))   # labeled come first in all_idx
        uncovered = find_uncovered_clusters(
            cluster_labels,
            labeled_indices=labeled_idx,
            all_indices=all_idx
        )
        print(f"  Uncovered clusters: {len(uncovered)}")

        # --- Step 6: Select queries ---
        # Positions of unlabeled samples in all_idx array
        unlabeled_positions = list(range(len(labeled_idx), len(all_idx)))
        selected_positions = select_query_indices(
            cluster_labels, typicality, uncovered,
            unlabeled_positions, budget
        )

        # Convert positions back to original dataset indices
        selected_orig_idx = [all_idx[p] for p in selected_positions]

        # --- Step 7: Update labeled / unlabeled sets ---
        labeled_idx   = labeled_idx + selected_orig_idx
        unlabeled_set = set(unlabeled_idx) - set(selected_orig_idx)
        unlabeled_idx = list(unlabeled_set)

        # --- Step 8: Train classifier and evaluate ---
        print("  Training classifier...")
        clf_loader = make_classifier_loader(train_dataset, labeled_idx,
                                            batch_size=min(128, len(labeled_idx)))
        classifier = LinearClassifier(simclr_model.encoder)
        classifier = train_classifier(classifier, clf_loader,
                                      epochs=classifier_epochs, device=device)
        acc = evaluate(classifier, test_loader, device)
        print(f"  Test accuracy: {acc*100:.2f}%  (|L|={len(labeled_idx)})")

        results.append({'n_labeled': len(labeled_idx), 'accuracy': acc})

    return results
