import numpy as np
import torch

def get_samplers(labels, n=100, n_categories=10):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler

    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_categories)]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices_labeled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_categories)])

    indices_unlabeled = torch.from_numpy(indices[~np.in1d(indices, indices_labeled)])
    indices_labeled = torch.from_numpy(indices_labeled)

    return SubsetRandomSampler(indices_labeled), SubsetRandomSampler(indices_unlabeled)


def to_logits(cls, x):
    """Convert the input image `x` to logits.

    Args:
        x (torch.Tensor): Input image.
        sldj (torch.Tensor): Sum log-determinant of Jacobian.

    Returns:
        y (torch.Tensor): Dequantized logits of `x`.

    See Also:
        - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
        - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
    """
    bounds = torch.tensor([0.9], dtype=torch.float32)
    y = (2 * x - 1) * bounds
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
          - F.softplus((1. - bounds).log() - bounds.log())
    sldj = ldj.flatten(1).sum(-1)

    return y, sldj


def convert_to_one_hot(cls, num_categories, labels):
    labels = torch.unsqueeze(labels, 1)
    one_hot = torch.FloatTensor(len(labels), num_categories).zero_()
    one_hot.scatter_(1, labels, 1)
    return one_hot