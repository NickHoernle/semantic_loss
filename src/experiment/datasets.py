import matplotlib.pyplot as plt
from torch.utils import data

import pickle
import os

import torch
import numpy as np

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from experiment.class_mapping import mnist_domain_knowledge as knowledge


# for MNIST experiment, needs to fix the headers from urllib default:
# https://stackoverflow.com/questions/60548000
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


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
    do_normalize=True,
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

    train_transforms = []
    valid_transforms = [transforms.ToTensor()]

    if augment:
        train_transforms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    train_transforms += [transforms.ToTensor()]

    if do_normalize:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        valid_transforms += normalize
        train_transforms += normalize

    # define transforms
    valid_transform = transforms.Compose(valid_transforms)
    train_transform = transforms.Compose(train_transforms)

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

    if dataset.upper() in ["CIFAR10", "CIFAR100"]:
        meta_name = "batches.meta" if dataset.upper() == "CIFAR10" else "meta"
        with open(
            os.path.join(data_dir, train_dataset.base_folder, meta_name), "rb"
        ) as infile:
            key = "label_names" if dataset.upper() == "CIFAR10" else "fine_label_names"
            data = pickle.load(infile, encoding="latin1")
            classes = data[key]
    else:
        # mnist
        classes = list(range(10))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    if dataset.upper() == "MNIST":
        train_dataset = build_mixture_dataset(train_dataset, train_idx)
        valid_dataset = build_mixture_dataset(valid_dataset, valid_idx)
        train_sampler = None
        valid_sampler = None

    else:
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
    do_normalize=True,
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
    test_transforms = [transforms.ToTensor()]
    if do_normalize:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        test_transforms.append(normalize)

    # define transform
    transform = transforms.Compose(test_transforms)

    dataset = datasets.__dict__[dataset.upper()](
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if dataset.upper() == "MNIST":
        test_idxs = np.arange(len(dataset))
        dataset = build_mixture_dataset(dataset, test_idxs)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


class generator:
    def plot(self, ax=None, samples=2500):
        if type(ax) == type(None):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.gca()
        x = self.sample(samples)
        ax.scatter(x[:, 0], x[:, 1], s=5, alpha=0.1)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)
        return ax


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


class Gaussian(generator):
    def sample(self, n):
        return 5 * np.random.randn(n, 2)


class ConstraintedSampler(Gaussian):
    def __init__(self, **kwargs):
        super().__init__()
        self.rotations = kwargs.get(
            "rotations",
            [
                0,
                np.pi / 4,
                2 * np.pi / 4,
                3 * np.pi / 4,
                np.pi,
                5 * np.pi / 4,
                6 * np.pi / 4,
                7 * np.pi / 4,
            ],
        )

    def term1(self, x):
        valid = (x[:, 1] > 2.5) & (x[:, 1] < 5.5) & (x[:, 0] > -0.5) & (x[:, 0] < 0.5)
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
        terms = []

        for i, theta in enumerate(self.rotations):
            constrained = []
            count = 1
            while len(constrained) < n:
                x = super().sample(n * 2 ** count)
                constrained = x[self.rotate(x, theta=theta)]
                count += 1

            terms.append(constrained)

        samples = np.concatenate(terms, axis=0)
        labels = np.concatenate(
            [i * np.ones_like(t[:, 0]).astype(int) for i, t in enumerate(terms)]
        )
        idxs = np.random.choice(np.arange(len(labels)), size=len(labels), replace=False)
        samples = samples[idxs]
        labels = labels[idxs]

        if get_term_labels:
            return samples[:n], labels[:n]
        return samples[:n]

    def plot(self, ax=None, with_term_labels=False):
        if not with_term_labels:
            return super().plot(ax=ax)

        if type(ax) == type(None):
            fig = plt.figure(figsize=(4, 4))
            ax = fig.gca()

        x, labels = self.sample(5000, get_term_labels=True)
        for i in range(np.max(labels) + 1):
            ax.scatter(x[labels == i, 0], x[labels == i, 1], s=5, alpha=0.1, label=i)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)

        return ax


class SyntheticDataset(data.Dataset):
    def __init__(self, sampler, nsamples: int = 5000):

        super(SyntheticDataset, self).__init__()

        samples, labels = sampler.sample(nsamples, get_term_labels=True)

        self.samples = torch.tensor(samples).float()
        self.labels = torch.tensor(labels).long()
        self.sampler = sampler

        self.listIDs = np.arange(len(self.labels))

    def __len__(self):
        return len(self.listIDs)

    def __getitem__(self, index):
        id = self.listIDs[index]
        x = self.samples[id]
        l = self.labels[id]
        return x, l


def get_synthetic_loaders(
    train_size: int = 5000,
    valid_size: int = 1000,
    test_size: int = 1000,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = False,
    sampler_params: dict = {},
):
    c = ConstraintedSampler(**sampler_params)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train = torch.utils.data.DataLoader(
        SyntheticDataset(c, nsamples=train_size), **kwargs
    )
    valid = torch.utils.data.DataLoader(
        SyntheticDataset(c, nsamples=valid_size), **kwargs
    )
    test = torch.utils.data.DataLoader(
        SyntheticDataset(c, nsamples=test_size), **kwargs
    )

    return train, valid, test


class Joint(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index], self.dataset3[index]

    def __len__(self):
        return len(self.dataset1)


def build_mixture_dataset(dataset, indices):
    nd = len(indices)

    ind1 = np.random.choice(indices, size=2 * nd, replace=True)
    ind2 = np.random.choice(indices, size=2 * nd, replace=True)

    try:
        labels = np.array(dataset.train_labels)
    except:
        labels = np.array(dataset.test_labels)

    target = labels[ind1] + labels[ind2]

    dset_1 = []
    dset_2 = []
    dset_t = []

    for k, conditions in knowledge.items():
        for val in conditions:
            valid_idxs = (
                (target == k) & (labels[ind1] == val[0]) & (labels[ind2] == val[1])
            )

            dset_1 += ind1[valid_idxs].tolist()
            dset_2 += ind2[valid_idxs].tolist()
            dset_t += ind1[labels[ind1] == k][: valid_idxs.sum()].tolist()

    indexes = np.arange(len(dset_1))
    np.random.shuffle(indexes)

    return Joint(
        Subset(dataset, np.array(dset_1)[indexes]),
        Subset(dataset, np.array(dset_2)[indexes]),
        Subset(dataset, np.array(dset_t)[indexes]),
    )


if __name__ == "__main__":
    dataset = ConstraintedSampler()
    dataset.plot(with_term_labels=True)
