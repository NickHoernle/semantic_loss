import torch
import torch.nn as nn
from symbolic.symbolic import OrList
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class LinearVAE(nn.Module):
    def __init__(self, ndims: int = 2, nhidden: int = 50, nlatent: int = 15, **kwargs):
        super(LinearVAE, self).__init__()

        self.ndims = ndims
        self.nhidden = nhidden
        self.nlatent = nlatent

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(ndims, nhidden), nn.ReLU(True), nn.Linear(nhidden, 2 * nlatent)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(nlatent, nhidden), nn.ReLU(True), nn.Linear(nhidden, ndims)
        )

        self.apply(init_weights)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def encode(self, x):
        return self.encoder(x).split(self.nlatent, -1)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, nsamples: int = 1000):
        z = 2 * torch.randn((1000, self.nlatent))
        return self.decode(z)

    def forward(self, x, test=False):
        # encoding
        encoded = self.encode(x)

        mu = encoded[0]
        log_var = encoded[1]

        z = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(z)
        return reconstruction, (mu, log_var)


class ConstrainedVAE(LinearVAE):
    def __init__(self, terms, **kwargs):
        super().__init__(**kwargs)

        self.nterms = len(terms)
        self.logic = OrList(terms=terms)

        self.encoder = nn.Sequential(
            nn.Linear(self.ndims, self.nhidden),
            nn.ReLU(True),
            nn.Linear(self.nhidden, 2 * self.nlatent + self.nterms),
        )

    def encode(self, x):
        return self.encoder(x).split((self.nlatent, self.nlatent, self.nterms), dim=-1)

    def decode(self, z, labels=None, test=False, **kwargs):
        decoded = self.decoder(z)

        if type(labels) == type(None):
            labelz = torch.randint(low=0, high=self.nterms, size=(len(z),))
            pred = self.logic.all_predictions(decoded)
            idxs = np.arange(len(pred))
            return pred[idxs, labelz]

        return self.logic(decoded, labels, test, **kwargs)

    def forward(self, x, test=False, **kwargs):
        # encoding
        encoded = self.encode(x)

        mu = encoded[0]
        log_var = encoded[1]
        pred_y = encoded[2]

        z = self.reparameterize(mu, log_var)

        # decoding
        return self.decode(z, pred_y, test, **kwargs), (mu, log_var)
