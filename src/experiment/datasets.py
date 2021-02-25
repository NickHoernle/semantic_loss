import matplotlib.pyplot as plt
from torch.utils import data


import pickle
import os

import torch
import numpy as np

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(
    data_dir,
    batch_size,
    augment,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    dataset="cifar10",
    num_workers=4,
    pin_memory=False,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    # load the dataset
    train_dataset = datasets.__dict__[dataset.upper()](
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    valid_dataset = datasets.__dict__[dataset.upper()](
        root=data_dir,
        train=True,
        download=True,
        transform=valid_transform,
    )

    meta_name = "batches.meta" if dataset.upper() == "CIFAR10" else "meta"
    with open(
        os.path.join(data_dir, train_dataset.base_folder, meta_name), "rb"
    ) as infile:
        key = "label_names" if dataset.upper() == "CIFAR10" else "fine_label_names"
        data = pickle.load(infile, encoding="latin1")
        classes = data[key]

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, classes


def get_test_loader(
    data_dir,
    batch_size,
    dataset="cifar10",
    shuffle=True,
    num_workers=4,
    pin_memory=False,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.__dict__[dataset.upper()](
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


class generator:
    def plot(self, samples=2500):
        fig = plt.figure(figsize=(4, 4))

        ax = fig.gca()
        x = self.sample(samples)
        ax.scatter(x[:, 0], x[:, 1], s=5, alpha=0.1)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)

        plt.show()


class GaussianMixture(generator):
    """ 4 mixture of gaussians """

    def sample(self, n):
        self.NUM_CATEGORIES = 4

        assert n % 2 == 0
        offset = 3
        r = np.r_[
            np.random.randn(n // 2, 2) * 1 + np.array([-offset, 0]),
            np.random.randn(n // 2, 2) * 1 + np.array([offset, 0]),
            np.random.randn(n // 2, 2) * 1 + np.array([0, -offset]),
            np.random.randn(n // 2, 2) * 1 + np.array([0, offset]),
        ]

        return r.astype(np.float32)


class ConstraintedSampler(GaussianMixture):
    def term1(self, x):
        valid = (x[:, 1] > 0.5) & (x[:, 1] < 5.5) & (x[:, 0] > -0.25) & (x[:, 0] < 0.25)
        return valid

    def rotate(self, x, theta=np.pi / 4):
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        return self.term1(x.dot(rotation))

    def sample(self, n, get_term_labels=False):
        x = super().sample(n)
        rotations = [
            0,
            np.pi / 4,
            2 * np.pi / 4,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            6 * np.pi / 4,
            7 * np.pi / 4,
        ]
        terms = [x[self.rotate(x, theta=theta)] for i, theta in enumerate(rotations)]
        if get_term_labels:
            return (
                np.concatenate(terms, axis=0),
                np.concatenate(
                    [i * np.ones_like(t[:, 0]).astype(int) for i, t in enumerate(terms)]
                ),
            )
        return np.concatenate(terms, axis=0)

    def plot(self, with_term_labels=False):
        if not with_term_labels:
            return super().plot()

        fig = plt.figure(figsize=(4, 4))

        ax = fig.gca()
        x, labels = self.sample(5000, get_term_labels=True)
        for i in range(np.max(labels) + 1):
            ax.scatter(x[labels == i, 0], x[labels == i, 1], s=5, alpha=0.1, label=i)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)

        plt.show()
        return


class SyntheticDataset(data.Dataset):
    def __init__(self, sampler, include_labels: bool = False, nsamples: int = 5000):

        super(SyntheticDataset, self).__init__()

        self.samples, self.labels = sampler.sample(nsamples, include_labels)
        self.listIDs = np.arange(len(self.labels))

    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        id = self.listIDs[index]
        x = torch.tensor(self.samples[id])
        return x


if __name__ == "__main__":
    dataset = ConstraintedSampler()
    dataset.plot(with_term_labels=True)
