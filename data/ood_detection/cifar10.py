"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import torch
import numpy as np
from torch.utils.data import Subset

from torchvision import datasets
from torchvision import transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - val_seed: fix seed for reproducibility.
    - val_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transforms
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    if augment:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    # load the dataset
    data_dir = "/root/shared-nvme/dataset"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform,)

    valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=valid_transform,)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transform
    transform = transforms.Compose([transforms.ToTensor(), normalize,])

    data_dir = "./data"
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform,)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
