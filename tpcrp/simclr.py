"""
SimCLR self-supervised representation learning for TPC RP.
Backbone: ResNet-18 trained from scratch on CIFAR-10 (no labels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm import tqdm


class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        # ResNet-18 backbone — adapt first conv for 32x32 CIFAR images
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        feat_dim = 512

        # 2-layer MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)   # (B, 512)
        z = self.projector(h)             # (B, proj_dim)
        return h, z

    def encode(self, x):
        """Return L2-normalised encoder embeddings (no projection head).
        Paper uses L2-normalised 512-dim penultimate layer as embedding space.
        """
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
            h = F.normalize(h, dim=1)
        return h


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalised temperature-scaled cross-entropy loss (NT-Xent).
    z1, z2: (B, D) — two views of the same batch.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate all representations: [z1; z2] shape (2B, D)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B),
                         torch.arange(0, B)]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_simclr(model, loader, epochs=500, lr=0.4, device='cpu'):
    """Train SimCLR on unlabeled+labeled data (pairs of augmented views).
    Paper: SGD, lr=0.4, momentum=0.9, weight_decay=1e-4, cosine scheduler, 500 epochs.
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4,
                                nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(loader)
            print(f"  SimCLR epoch {epoch:3d}/{epochs} | loss {avg:.4f}")

    return model


@torch.no_grad()
def extract_embeddings(model, loader, device='cpu'):
    """
    Extract encoder embeddings for all samples in loader.
    Returns (embeddings, labels) as numpy arrays.
    """
    model.eval()
    model.to(device)
    all_embs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        embs = model.encode(imgs)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return (torch.cat(all_embs).numpy(),
            torch.cat(all_labels).numpy())
