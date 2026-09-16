import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy inside each DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _split_indices(dataset_size: int, val_size: int, split_seed: int) -> Tuple[list, list]:
    """Create a fixed train/validation split independent of the training run seed."""
    if not 0 < val_size < dataset_size:
        raise ValueError(
            f"val_size must be between 1 and {dataset_size - 1}; got {val_size}."
        )

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def _loader_kwargs(batch_size: int, num_workers: int, pin_memory: bool) -> dict:
    # persistent_workers=False is deliberate: workers are recreated each epoch,
    # which makes epoch-boundary checkpoint/resume behavior more reproducible.
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }


def build_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    split_seed: int = 42,
    val_size: int = 5000,
):
    """
    Build CIFAR-10 train/validation/test loaders.

    IMPORTANT:
    `split_seed` controls only which samples belong to train vs validation.
    The training run seed is set separately via src.utils.set_seed().
    This lets seed 42/123/2026 use the exact same data split.
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    augmented_train = datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )
    clean_train = datasets.CIFAR10(
        data_dir,
        train=True,
        download=False,
        transform=eval_tf,
    )
    test_set = datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=eval_tf,
    )

    train_idx, val_idx = _split_indices(
        dataset_size=len(augmented_train),
        val_size=val_size,
        split_seed=split_seed,
    )

    pin_memory = torch.cuda.is_available()
    common = _loader_kwargs(batch_size, num_workers, pin_memory)

    train_loader = DataLoader(
        Subset(augmented_train, train_idx),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Subset(clean_train, val_idx),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader


def build_test_loader(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
):
    """Build only the clean CIFAR-10 test loader for evaluation."""
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    test_set = datasets.CIFAR10(
        data_dir,
        train=False,
        download=True,
        transform=eval_tf,
    )

    common = _loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataLoader(test_set, shuffle=False, **common)
