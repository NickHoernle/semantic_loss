import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from wideresnet import WideResNet


class GEQConstant(nn.Module):
    def __init__(self, ixs_pos, ixs_not, ixs_neg, limit_threshold):
        super(GEQConstant, self).__init__()
        self.ixs_pos = ixs_pos
        self.ixs_neg = ixs_neg
        self.ixs_not = ixs_not

        self.limit_threshold = limit_threshold

        self.forward_transform = self.ixs_pos + self.ixs_neg + self.ixs_not
        self.reverse_transform = np.argsort(self.forward_transform)

    def threshold1p(self):
        if self.limit_threshold < 20:
            self.limit_threshold += 1

    def forward(self, x):
        x_ = x[:, self.forward_transform]
        split1, split2, split3 = x_.split([len(self.ixs_pos), len(self.ixs_neg), len(self.ixs_not)], dim=1)
        return torch.cat((F.softplus(split1),
                          -F.relu(-split2) - self.limit_threshold,
                          split3),
                         dim=1)[:, self.reverse_transform]


class GEQ(GEQConstant):

    def forward(self, x):
        x_ = x[:, self.forward_transform]
        split1, split2, split3 = x_.split([len(self.ixs_pos), len(self.ixs_neg), len(self.ixs_not)], dim=1)

        max_val = F.relu(split2).max(dim=1)[0]
        min_val = (F.relu(-split1).max(dim=1)[0])

        return torch.cat((split1 + min_val.unsqueeze(1) + max_val.unsqueeze(1),
                          split2 - self.limit_threshold,
                          split3), dim=1)[:, self.reverse_transform]


class OrList(nn.Module):
    def __init__(self, terms):
        super(OrList, self).__init__()
        self.layers = nn.ModuleList(terms)
        self.tau_ = 5

    @property
    def tau(self):
        return self.tau_

    def update_tau(self):
        self.tau_ = np.max([.96 * self.tau_, .5])

    def threshold1p(self):
        for layer in self.layers:
            layer.threshold1p()

    def forward(self, x, class_prediction, test=False):
        log_py = class_prediction.log_softmax(dim=1)
        pred = torch.stack([f(x) for f in self.layers], dim=1)
        if test:
            return (pred[np.arange(len(log_py)), log_py.argmax(dim=1)])
        return pred, log_py


class ConstrainedModel(nn.Module):
    def __init__(self, depth, classes, layers, widen_factor=1, dropRate=0.0):
        super(ConstrainedModel, self).__init__()

        self.nterms = len(layers)
        self.nclasses = classes

        self.encoder = WideResNet(depth, classes+self.nterms, widen_factor, dropRate=dropRate)
        self.decoder = OrList(terms=layers)

    def threshold1p(self):
        self.decoder.threshold1p()

    def forward(self, x, test=False):
        enc = self.encoder(x)
        ps, preds = enc.split((self.nclasses, self.nterms), dim=1)
        return self.decoder(ps, preds, test)


# 0, 'airplane', 1 'automobile', 2 'bird', 3'cat', 4 'deer', 5 'dog', 6 'frog', 7 'horse', 8 'ship', 9 'truck'
def get_logic_terms(dataset):
    if dataset == "cifar10":
        terms = [
            GEQConstant(ixs_pos=[0, 8], ixs_not=[], ixs_neg=[1, 2, 3, 4, 5, 6, 7, 9], limit_threshold=0),
            GEQConstant(ixs_pos=[1, 9], ixs_not=[], ixs_neg=[0, 2, 3, 4, 5, 6, 7, 8], limit_threshold=0),
            GEQConstant(ixs_pos=[3, 5], ixs_not=[], ixs_neg=[0, 1, 2, 4, 6, 7, 8, 9], limit_threshold=0),
            GEQConstant(ixs_pos=[4, 7], ixs_not=[], ixs_neg=[0, 1, 2, 3, 5, 6, 8, 9], limit_threshold=0),
            GEQConstant(ixs_pos=[2], ixs_not=[], ixs_neg=[0, 1, 3, 4, 5, 6, 7, 8, 9], limit_threshold=0),
            GEQConstant(ixs_pos=[6], ixs_not=[], ixs_neg=[0, 1, 2, 3, 4, 5, 7, 8, 9], limit_threshold=0)
        ]
        return terms


def get_class_ixs(dataset):
    if dataset == "cifar10":
        return [t.ixs_pos for t in get_logic_terms(dataset)]

