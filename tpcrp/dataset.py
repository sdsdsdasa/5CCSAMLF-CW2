"""
CIFAR-10 data loading and augmentation pipelines for TPC RP.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# SimCLR augmentation: two random views of each image
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# Standard eval transform (for classifier training and embedding extraction)
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Augmented transform for classifier training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class PairDataset(Dataset):
    """Wraps a dataset to return two augmented views (for SimCLR)."""
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        v1, v2 = self.transform(img)
        return v1, v2


def get_cifar10(data_dir='./data'):
    """Return raw CIFAR-10 train and test datasets (no transform applied)."""
    train = datasets.CIFAR10(data_dir, train=True, download=True,
                             transform=None)
    test  = datasets.CIFAR10(data_dir, train=False, download=True,
                             transform=eval_transform)
    return train, test


def initial_labeled_set(train_dataset, n_per_class=1, seed=42):
    """
    Select n_per_class examples per class as the initial labeled set L0.
    Returns labeled indices and unlabeled indices.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(train_dataset.targets)
    labeled_idx = []
    for cls in range(10):
        cls_idx = np.where(targets == cls)[0]
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        labeled_idx.extend(chosen.tolist())
    labeled_idx = sorted(labeled_idx)
    all_idx = set(range(len(train_dataset)))
    unlabeled_idx = sorted(all_idx - set(labeled_idx))
    return labeled_idx, unlabeled_idx


def make_simclr_loader(train_dataset, indices, batch_size=256, num_workers=2):
    """DataLoader that returns augmented pairs for SimCLR training."""
    subset = Subset(train_dataset, indices)
    pair_ds = PairDataset(subset, SimCLRTransform(size=32))
    return DataLoader(pair_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True)


def make_embed_loader(train_dataset, indices, batch_size=512, num_workers=2):
    """DataLoader with eval transform for extracting embeddings."""
    class EvalSubset(Dataset):
        def __init__(self, ds, idx, tfm):
            self.ds = ds
            self.idx = idx
            self.tfm = tfm
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            img, label = self.ds[self.idx[i]]
            return self.tfm(img), label

    ds = EvalSubset(train_dataset, indices, eval_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def make_classifier_loader(train_dataset, indices, batch_size=128,
                            num_workers=2, augment=True):
    """DataLoader for classifier training on labeled set."""
    class LabeledSubset(Dataset):
        def __init__(self, ds, idx, tfm):
            self.ds = ds
            self.idx = idx
            self.tfm = tfm
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            img, label = self.ds[self.idx[i]]
            return self.tfm(img), label

    tfm = train_transform if augment else eval_transform
    ds = LabeledSubset(train_dataset, indices, tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
