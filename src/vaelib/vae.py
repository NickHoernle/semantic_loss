from __future__ import print_function
import argparse
import itertools
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal, Uniform, \
    TransformedDistribution, SigmoidTransform, Normal, Categorical

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight,0,.05)
        m.bias.data.fill_(0)

class VAE(nn.Module):
    def __init__(self, data_dim, hidden_dim, condition=False, num_condition=0, **kwargs):
        super().__init__()

        input_dim = kwargs.get('input_dim', data_dim)
        # encoding layers
        self.encoder = nn.Sequential(
            # nn.Linear(input_dim, 500),
            # nn.LeakyReLU(.01),
            # nn.Linear(500, 250),
            # nn.LeakyReLU(.01),
            nn.Linear(input_dim, 40),
            nn.LeakyReLU(.01),
            nn.Linear(40, 40),
            nn.LeakyReLU(.01)
            )

        self.mu_enc = nn.Linear(250, hidden_dim)
        self.sig_enc = nn.Linear(250, hidden_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim+condition*num_condition, 250),
            nn.LeakyReLU(.01),
            nn.Linear(250, 250),
            nn.LeakyReLU(.01),
            nn.Linear(250, 500),
            nn.LeakyReLU(.01),
            nn.Linear(500, data_dim),
        )
        self.hidden_dim = hidden_dim
        self.condition = condition
        self.num_condition = num_condition

        self.zeros = torch.zeros(hidden_dim)
        self.eye = torch.eye(hidden_dim)
        self.base = MultivariateNormal(self.zeros, self.eye)

        self.apply(init_weights)

    def encode(self, x, **kwargs):
        input = x
        # if self.condition:
        #     input = torch.cat((x, kwargs['condition_variable']), -1)
        latent_q = self.encoder(input)
        return self.mu_enc(latent_q), self.sig_enc(latent_q)

    def reparameterize(self, mu, logvar):

        sigma = torch.exp(logvar)
        # sigma = torch.ones_like(logvar)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma

        return z

    def decode(self, z, **kwargs):
        if self.condition:
            return self.decoder(torch.cat((z, kwargs['condition_variable']), -1))
        return self.decoder(z)

    def forward(self, x, **kwargs):
        q_mu, q_logvar = self.encode(x, **kwargs)

        z = self.reparameterize(q_mu, q_logvar)

        x_reconstructed = self.decode(z, **kwargs)

        return x_reconstructed, (q_mu, q_logvar)

    def reconstruct(self, z, **kwargs):
        return self.decode(z, **kwargs)

    def sample(self, num_samples):
        z = torch.randn((num_samples, self.hidden_dim))
        return self.decode(z)

    @classmethod
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

class VAE_Gaussian(VAE):
    def __init__(self, data_dim, hidden_dim, condition=False, num_condition=0, **kwargs):
        super().__init__(data_dim, hidden_dim, condition=condition, num_condition=num_condition, **kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + condition * num_condition, 250),
            nn.LeakyReLU(.01),
            nn.Linear(250, 250),
            nn.LeakyReLU(.01),
            nn.Linear(250, 500),
            nn.LeakyReLU(.01)
        )
        self.mu_dec = nn.Linear(500, data_dim)
        self.sig_dec = nn.Linear(500, data_dim)

    def decode(self, z, **kwargs):
        if self.condition:
            mid_var = self.decoder(torch.cat((z, kwargs['condition_variable']), -1))
        else:
            mid_var = self.decoder(z)
        return self.mu_dec(mid_var), self.sig_dec(mid_var)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class CNN(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        data_dim = kwargs.get("data_dim", 32)
        channel_num = kwargs.get("channel_num", 1)
        kernel_num = kwargs.get("kernel_num", 100)
        hidden_dim = kwargs.get("hidden_dim", 100)

        self.image_size = data_dim
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = hidden_dim

        # encoding layers
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num, last=True),
        )

        # encoded feature's size and volume
        self.feature_size = self.image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, hidden_dim, relu=False)
        self.q_logvar = self._linear(self.feature_volume, hidden_dim, relu=False)

        # projection
        self.project = self._linear(hidden_dim, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, last=True),
        )

    ##########  LAYERS  ###########
    def _conv(self, channel_size, kernel_num, last=False):
        conv = nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=3, stride=2, padding=1,
        )
        return conv if last else nn.Sequential(
            conv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(True),
        )

    def _deconv(self, channel_num, kernel_num, last=False):
        deconv = nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )
        return deconv if last else nn.Sequential(
            deconv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(True),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(True),
        ) if relu else nn.Linear(in_size, out_size)

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)


class VAE_Categorical(CNN):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150, condition=False, num_condition=0):
        super().__init__(data_dim=data_dim, hidden_dim=hidden_dim, channel_num=channel_num,
                         kernel_num=kernel_num, condition=condition, num_condition=num_condition)

        self.NUM_CATEGORIES = NUM_CATEGORIES
        self.category_prior = 1

        self.means = nn.Parameter(torch.rand(NUM_CATEGORIES, hidden_dim))
        self.q_log_var = nn.Parameter(torch.ones(NUM_CATEGORIES, hidden_dim))

        self.prior = torch.zeros(hidden_dim)
        self.post_cov = torch.eye(hidden_dim)
        self.eye = torch.eye(hidden_dim)

        self.tau = 2.
        self.reparameterize_means()

        self.apply(init_weights)

    def discriminator(self, z_means, q_mu, q_logvar):

        qs = q_mu.unsqueeze(1).repeat(1, self.NUM_CATEGORIES, 1)
        sigs = torch.exp(q_logvar).unsqueeze(1).repeat(1, self.NUM_CATEGORIES, 1)

        ms = z_means.unsqueeze(0).repeat(len(q_mu), 1, 1)
        ms_sigs = torch.exp(self.q_log_var).unsqueeze(0).repeat(len(q_mu), 1, 1)

        base_dist = MultivariateNormal(self.prior, self.eye)

        return base_dist.log_prob((qs - ms)/(sigs + ms_sigs))

    def forward(self, data_sample, **kwargs):

        (data, labels) = data_sample
        if type(labels) != type(None):
            return self.forward_labelled(data, labels, **kwargs)
        return self.forward_unlabelled(data, **kwargs)

    def forward_unlabelled(self, x, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar) = self.q(encoded)

        z = self.reparameterize(q_mu, q_logvar)
        z_means = self.reparameterize(self.means, self.q_log_var)

        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        label_log_prob = self.discriminator(self.means, q_mu, q_logvar)
        pred_label_sm_log = label_log_prob - torch.logsumexp(label_log_prob, dim=1).unsqueeze(1)

        return x_reconstructed, (z, z_means), (q_mu, q_logvar, self.means, self.q_log_var, pred_label_sm_log)

    def forward_labelled(self, x, one_hot_labels, **kwargs):

        # encode x
        encoded = self.encoder(x)
        (q_mu, q_logvar) = self.q(encoded)

        z = self.reparameterize(q_mu, q_logvar)

        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        z_means = self.reparameterize(self.means, self.q_log_var)

        label_log_prob = self.discriminator(self.means, q_mu, q_logvar)
        pred_label_sm_log = label_log_prob - torch.logsumexp(label_log_prob, dim=1).unsqueeze(1)

        return x_reconstructed, (z, z_means), (q_mu, q_logvar, self.means, self.q_log_var, pred_label_sm_log), one_hot_labels

    def sample(self, num_samples, **kwargs):
        num_test_samples = num_samples * self.NUM_CATEGORIES
        labels = torch.zeros(num_test_samples, self.NUM_CATEGORIES).to(kwargs['device'])
        labels[torch.arange(num_test_samples), torch.arange( self.NUM_CATEGORIES).repeat(num_test_samples // self.NUM_CATEGORIES)] = 1
        return self.sample_labelled(labels)

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.prior, self.eye)
        latent = (labels.unsqueeze(dim=2)*(self.means.repeat(n_samps, 1, 1)
                    + base_dist.sample((n_samps,)).unsqueeze(dim=1).repeat(1,self.NUM_CATEGORIES,1))).sum(dim=1)

        # z = self.base_dist.sample((n_samps,))
        # latent = torch.cat((z, labels), -1)

        z_projected = self.project(latent).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        return self.decoder(z_projected)

    @torch.no_grad()
    def update_means(self, sum_means, n, annealing):
        new_means = (sum_means/n.view(-1,1))
        self.means.copy_((1-annealing)*self.means + annealing*new_means)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.prior = self.prior.to(*args, **kwargs)
        self.post_cov = self.post_cov.to(*args, **kwargs)
        self.eye = self.eye.to(*args, **kwargs)
        return self


class M2(CNN):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim, hidden_dim=hidden_dim, channel_num=channel_num, kernel_num=kernel_num)

        self.log_q_y = self._linear(self.feature_volume, NUM_CATEGORIES, relu=False)
        self.project = self._linear(hidden_dim, self.feature_volume, relu=False)
        self.proj_y = self._linear(NUM_CATEGORIES, hidden_dim, relu=False)

        self.num_categories = NUM_CATEGORIES
        self.eye = torch.eye(hidden_dim)

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled), self.log_q_y(unrolled)

    def forward(self, data_sample, **kwargs):

        (data, labels) = data_sample
        if type(labels) != type(None):
            return self.forward_labelled(data, labels, **kwargs)
        return self.forward_unlabelled(data, **kwargs)

    def forward_labelled(self, x, one_hot_labels, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        pred_label_sm_log = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)
        Ws = self.softplus(self.proj_y(one_hot_labels)).unsqueeze(1) * self.eye
        h1 = self.relu(torch.matmul(Ws, z.unsqueeze(2)).squeeze(-1))

        z_projected = self.project(h1).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        x_reconstructed = self.decoder(z_projected)

        return {"reconstructed": [x_reconstructed],
                "latent_samples": [z],
                "q_vals": [q_mu, q_logvar, pred_label_sm_log],
                "labels": [one_hot_labels]}

    def forward_unlabelled(self, x, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        pred_label_sm_log = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)

        reconstructions = []

        for cat in range(self.num_categories):
            labels = torch.zeros_like(pred_label_sm_log)
            labels[:, cat] = 1

            z = self.reparameterize(q_mu, q_logvar)
            Ws = self.softplus(self.proj_y(labels)).unsqueeze(1) * self.eye
            h1 = self.relu(torch.matmul(Ws, z.unsqueeze(2)).squeeze(-1))

            z_projected = self.project(h1).view(
                -1, self.kernel_num,
                self.feature_size,
                self.feature_size,
            )

            reconstructions.append(self.decoder(z_projected))

        return {"reconstructed": reconstructions,
                "latent_samples": [z],
                "q_vals": [q_mu, q_logvar, pred_label_sm_log]}

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.zeros, self.eye)
        z = base_dist.sample((n_samps,))
        # latent = (labels.unsqueeze(dim=2)*(self.means.repeat(n_samps, 1, 1)
        #             + base_dist.sample((n_samps,)).unsqueeze(dim=1).repeat(1,self.NUM_CATEGORIES,1))).sum(dim=1)

        # z = self.base_dist.sample((n_samps,))
        Ws = self.softplus(self.proj_y(labels)).unsqueeze(1) * self.eye
        h1 = self.relu(torch.matmul(Ws, z.unsqueeze(2)).squeeze(-1))

        z_projected = self.project(h1).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        return self.decoder(z_projected)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.eye = self.eye.to(*args, **kwargs)
        return self
