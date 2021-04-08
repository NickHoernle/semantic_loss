import torch.nn as nn

from experiment.wideresnet import WideResNet
from symbolic.symbolic import OrList


class ConstrainedModel(nn.Module):
    def __init__(self, depth, classes, layers, widen_factor=1, dropRate=0.0):
        super(ConstrainedModel, self).__init__()

        self.nterms = len(layers)
        self.nclasses = classes

        self.encoder = WideResNet(depth, classes, widen_factor, dropRate=dropRate)

        self.logic_pred = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(self.classes), nn.Linear(self.nterms)
        )

        self.decoder = OrList(terms=layers)

    def threshold1p(self):
        self.decoder.threshold1p()

    def forward(self, x, test=False):
        ps = self.encoder(x)
        preds = self.logic_pred(ps)

        return self.decoder(ps, preds, test)
