from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def build_dataloaders(data_root: str, batch_size: int, num_workers: int, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders for FashionMNIST."""
    train_tfms = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # dataset mean/std
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_full = datasets.FashionMNIST(root=data_root, train=True, transform=train_tfms, download=True)
    test_ds = datasets.FashionMNIST(root=data_root, train=False, transform=test_tfms, download=True)

    # Split train into train/val (90/10)
    val_size = int(0.1 * len(train_full))
    train_size = len(train_full) - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
