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


class MnistVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, num_labels=10, num_terms=55):
        super().__init__()

        self.z_dim = z_dim
        self.num_labels = num_labels

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim1), nn.ReLU(), nn.Linear(h_dim1, h_dim2), nn.ReLU()
        )

        self.label_predict = nn.Linear(h_dim2, num_labels)

        self.mu = nn.Linear(h_dim2 + num_labels, z_dim)
        self.lv = nn.Linear(h_dim2 + num_labels, z_dim)

        self.mu_prior = nn.Linear(num_labels, z_dim)
        self.lv_prior = nn.Linear(num_labels, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, x_dim),
        )

        self.apply(init_weights)

    def get_one_hot(self, x, y):
        y_one_hot = torch.zeros_like(x[:, 0])[:, None].repeat(1, self.num_labels)
        y_one_hot[:, y] = 1
        return y_one_hot

    def get_priors(self, y):
        return self.mu_prior(y), self.lv_prior(y)

    def encode(self, x):
        h = self.encoder(x)
        return h, self.label_predict(h).log_softmax(dim=1)

    def get_latent(self, encoded):
        return self.mu(encoded), self.lv(encoded)

    def decode_one(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def collect(self, encoded, label):
        one_hot = self.get_one_hot(encoded, label)

        mu, lv = self.get_latent(torch.cat((encoded, one_hot), dim=1))
        mu_prior, lv_prior = self.get_priors(one_hot)

        z = self.reparameterize(mu, lv)
        recon = self.decode_one(z)

        return (recon, mu, lv, mu_prior, lv_prior, z)

    def decode(self, encoded):
        return [self.collect(encoded, label) for label in range(self.num_labels)]

    def forward(self, in_data):
        x1, x2, x3 = in_data
        encoded1, log_pred1 = self.encode(x1.view(-1, 784))
        encoded2, log_pred2 = self.encode(x2.view(-1, 784))
        encoded3, log_pred3 = self.encode(x3.view(-1, 784))

        d1 = self.decode(encoded1)
        d2 = self.decode(encoded2)
        d3 = self.decode(encoded3)

        # logic_pred = self.logic_pred(torch.cat((log_pred1, log_pred2, log_pred3), dim=1)).log_softmax(dim=1)
        return (d1, d2, d3), (log_pred1, log_pred2, log_pred3), None


class ConstrainedMnistVAE(MnistVAE):
    def __init__(self, num_terms=55, **kwargs):
        super().__init__(**kwargs)
        self.num_terms = num_terms
        self.logic_pred = nn.Linear(3 * self.num_labels, num_terms)

    def forward(self, in_data):
        x1, x2, x3 = in_data
        encoded1, log_pred1 = self.encode(x1.view(-1, 784))
        encoded2, log_pred2 = self.encode(x2.view(-1, 784))
        encoded3, log_pred3 = self.encode(x3.view(-1, 784))

        d1 = self.decode(encoded1)
        d2 = self.decode(encoded2)
        d3 = self.decode(encoded3)

        logic_pred = self.logic_pred(
            torch.cat((log_pred1, log_pred2, log_pred3), dim=1)
        ).log_softmax(dim=1)
        return (d1, d2, d3), (log_pred1, log_pred2, log_pred3), logic_pred