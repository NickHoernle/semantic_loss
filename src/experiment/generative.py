import torch
import torch.nn as nn
from symbolic.symbolic import OrList
import numpy as np
from torch.nn import functional as F
from experiment.class_mapping import flat_class_mapping as flat_knowledge


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == nn.Embedding:
        nn.init.uniform_(m.weight, -.1, .1)

class LinearVAE(nn.Module):
    def __init__(self, ndims: int = 2, nhidden: int = 50, nlatent: int = 15, **kwargs):
        super(LinearVAE, self).__init__()

        self.ndims = ndims
        self.nhidden = nhidden
        self.nlatent = nlatent

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(ndims, nhidden), nn.ReLU(
                True), nn.Linear(nhidden, 2 * nlatent)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(nlatent, nhidden), nn.ReLU(
                True), nn.Linear(nhidden, ndims)
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
        return reconstruction, (mu, log_var), None


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

        self._logic_prior = nn.Parameter(torch.randn(len(terms)))

    @property
    def logic_prior(self):
        return self._logic_prior.log_softmax(dim=0)

    def encode(self, x):
        return self.encoder(x).split((self.nlatent, self.nlatent, self.nterms), dim=-1)

    def decode(self, z, labels=None, test=False, **kwargs):
        decoded = self.decoder(z)

        if type(labels) == type(None):
            labelz = torch.tensor(np.random.choice(np.arange(self.nterms), replace=True, size=len(
                z), p=self.logic_prior.exp().detach().numpy())).long()
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
        return self.decode(z, pred_y, test, **kwargs), (mu, log_var), self.logic_prior


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size=1024):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class MnistVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, num_labels=10, num_terms=55):
        super().__init__()

        self.z_dim = z_dim
        self.num_labels = num_labels
        self.h_dim2 = h_dim2

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim1),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim2),
        )

        self.label_predict = nn.Linear(h_dim2, num_labels)

        self.mu = nn.Sequential(nn.ReLU(), nn.Linear(h_dim2, z_dim))
        self.lv = nn.Sequential(nn.ReLU(), nn.Linear(h_dim2, z_dim))

        # self.label_encoder_dec2 = nn.Sequential(nn.Embedding(num_labels, z_dim), nn.Sigmoid())
        self.label_encoder_dec1 = nn.Embedding(num_labels, z_dim)
        self.label_encoder_dec2 = nn.Sequential(nn.Embedding(num_labels, z_dim), nn.Softplus())
        self.mu_prior = nn.Embedding(num_labels, z_dim)
        self.lv_prior = nn.Sequential(
            nn.Embedding(num_labels, z_dim), nn.Tanh())

        self.decoder = nn.Sequential(
            # nn.ReLU(),
            # nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim + num_labels, h_dim2),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim2),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim1),
            nn.Linear(h_dim1, x_dim),
        )

        self.apply(init_weights)

    def get_one_hot(self, x, y):
        y_one_hot = torch.zeros_like(
            x[:, 0])[:, None].repeat(1, self.num_labels)
        y_one_hot[:, y] = 1
        return y_one_hot

    def get_priors(self, y):
        return self.mu_prior(y), self.lv_prior(y)

    def encode(self, x):
        h = self.encoder(x)
        return h, self.label_predict(h).log_softmax(dim=1)

    def get_latent(self, encoded):
        return self.mu(encoded), self.lv(encoded)

    def decode_one(self, z, label):
        lbl = torch.ones_like(z[:, 0]).long() * label
        one_hot = self.get_one_hot(z, label)
        return self.decoder(torch.cat((one_hot, z + self.label_encoder_dec1(lbl)), dim=1))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def collect(self, z, mu, lv, label):
        recon = self.decode_one(z, label)
        return (recon, mu, lv, None, None, z)

    def decode(self, encoded):
        mu, lv = self.get_latent(encoded)
        z = self.reparameterize(mu, lv)
        # z = torch.randn_like(mu)
        return [self.collect(z, mu, lv, label) for label in range(self.num_labels)]

    def forward(self, in_data, test=False):
        x1, x2, x3, x4 = in_data

        encoded1, log_pred1 = self.encode(x1)
        encoded2, log_pred2 = self.encode(x2)
        encoded3, log_pred3 = self.encode(x3)
        encoded4, log_pred4 = self.encode(x4)

        d1 = self.decode(encoded1)
        d2 = self.decode(encoded2)
        d3 = self.decode(encoded3)
        d4 = self.decode(encoded4)

        return (d1, d2, d3, d4), (log_pred1, log_pred2, log_pred3, log_pred4), None


class ConstrainedMnistVAE(MnistVAE):
    def __init__(self, terms, **kwargs):
        super().__init__(**kwargs)
        self.num_terms = len(terms)
        self.logic_decoder = OrList(terms=terms)

        self.logic_pred1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.num_labels),
            nn.Linear(2 * self.num_labels, len(terms)),
        )

        self.logic_pred2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.num_labels),
            nn.Linear(2 * self.num_labels, len(terms)),
        )

        self.warmup = nn.Linear(self.h_dim2, self.z_dim)
        self.apply(init_weights)

    def encode(self, x):
        h = self.encoder(x)
        return h, self.label_predict(h)

    def threshold1p(self):
        self.logic_decoder.threshold1p()

    def forward(self, in_data, test=False, warmup=False):
        # if warmup:
        #     encoded, log_pred1 = self.encode(in_data)
        #     return self.decoder(self.warmup(encoded))

        x1, x2, x3, x4 = in_data

        encoded1, log_pred1 = self.encode(x1)
        encoded2, log_pred2 = self.encode(x2)
        encoded3, log_pred3 = self.encode(x3)
        encoded4, log_pred4 = self.encode(x4)

        d1 = self.decode(encoded1)
        d2 = self.decode(encoded2)
        d3 = self.decode(encoded3)
        d4 = self.decode(encoded4)

        # cp1 = torch.cat(((log_pred1),
        #                 (log_pred2)), dim=1)
        lp1 = log_pred1.log_softmax(dim=1)
        lp2 = log_pred2.log_softmax(dim=1)
        lp = []
        for l1, l2, k in flat_knowledge:
            lp += [lp1[:, l1] + lp2[:, l2]]

        lp = torch.stack(lp, dim=1).log_softmax(dim=1)

        return (
            (d1, d2, d3, d4),
            (log_pred1, log_pred2, log_pred3, log_pred4),
            lp
        )
