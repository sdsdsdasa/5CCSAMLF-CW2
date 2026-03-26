"""
ResNet-18 classifier trained on the labeled set.
Uses the SimCLR encoder as a frozen backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LinearClassifier(nn.Module):
    """Linear probe on top of a frozen SimCLR encoder."""
    def __init__(self, encoder, feat_dim=512, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
        return self.fc(h)


def train_classifier(model, train_loader, epochs=200, lr=2.5, device='cpu'):
    """Train the linear classifier on the labeled set.
    Paper (linear eval): SGD, lr=2.5, Nesterov momentum=0.9, 200 epochs, cosine scheduler.
    Weights are re-initialised each AL iteration by constructing a new LinearClassifier.
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def evaluate(model, test_loader, device='cpu'):
    """Return top-1 accuracy on the test set."""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


import numpy as np

@torch.no_grad()
def compute_uncertainty_scores(model, loader, device='cpu'):
    """
    Compute per-sample prediction entropy from the classifier.
    Returns a numpy array of shape (N,) normalised to [0, 1],
    aligned with the loader's sample order (no shuffle assumed).

    Entropy H(x) = -sum_c p_c * log(p_c), max = log(num_classes).
    Higher value = classifier is more uncertain about the sample.
    """
    model.eval()
    model.to(device)
    entropies = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1)
        # clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=1)
        entropies.append(entropy.cpu())
    entropies = torch.cat(entropies).numpy()

    # Normalise to [0, 1] globally so it can be combined with typicality
    lo, hi = entropies.min(), entropies.max()
    if hi > lo:
        entropies = (entropies - lo) / (hi - lo)
    else:
        entropies = np.ones_like(entropies)
    return entropies
