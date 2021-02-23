import pickle
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data


class generator:

    def plot(self, samples=2500):
        fig = plt.figure(figsize=(4, 4))

        ax =  fig.gca()
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
            np.random.randn(n // 2, 2) * 1 + np.array([0, offset])
        ]

        return r.astype(np.float32)


class ConstraintedSampler(GaussianMixture):

    def term1(self, x):
        valid = (x[:, 1] > .5) & (x[:, 1] < 5.5) & (x[:, 0] > -.25) & (x[:, 0] < .25)
        return valid

    def rotate(self, x, theta=np.pi/4):
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        return self.term1(x.dot(rotation))

    def sample(self, n, get_term_labels=False):
        x = super().sample(n)
        rotations = [0, np.pi/4, 2*np.pi/4, 3*np.pi/4, np.pi, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4]
        terms = [x[self.rotate(x, theta=theta)] for i, theta in enumerate(rotations)]
        if get_term_labels:
            return np.concatenate(terms, axis=0), np.concatenate([i*np.ones_like(t[:, 0]).astype(int) for i, t in enumerate(terms)])
        return np.concatenate(terms, axis=0)

    def plot(self, with_term_labels=False):
        if not with_term_labels:
            return super().plot()

        fig = plt.figure(figsize=(4, 4))

        ax = fig.gca()
        x, labels = self.sample(5000, get_term_labels=True)
        for i in range(np.max(labels) + 1):
            ax.scatter(x[labels==i, 0], x[labels==i, 1], s=5, alpha=0.1, label=i)

        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.grid(True)

        plt.show()
        return


class SyntheticDataset(data.Dataset):
    def __init__(
        self,
        sampler,
        include_labels: bool=False,
        nsamples: int = 5000
    ):

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