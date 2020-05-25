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
import math

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

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.eye = self.eye.to(*args, **kwargs)
        self.zeros = self.zeros.to(*args, **kwargs)
        return self

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

        self.encoding_cnn = nn.Sequential(
            nn.Conv2d(channel_num, kernel_num//4, kernel_size=4, stride=2, padding=1),    # [batch, kernel_num//4, 16, 16]
            nn.LeakyReLU(.01),
            nn.Conv2d(kernel_num//4, kernel_num//2, kernel_size=4, stride=2, padding=1),  # [batch, kernel_num//2, 8, 8]
            nn.LeakyReLU(.01),
            nn.Conv2d(kernel_num//2, kernel_num, kernel_size=4, stride=2, padding=1),     # [batch, kernel_num, 4, 4]
            nn.LeakyReLU(.01),
        )

        self.feature_size = self.image_size // (2 ** 3)
        self.feature_volume = kernel_num * self.feature_size * self.feature_size

        self.encoder_linear = [
            nn.Linear(self.feature_volume, self.feature_volume),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume, self.feature_volume),
            nn.LeakyReLU(.01),
        ]

        self.q_mean = nn.Sequential(
            nn.Linear(self.feature_volume, self.feature_volume//2),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume//2, hidden_dim)
        )
        self.q_logvar = nn.Sequential(
            nn.Linear(self.feature_volume, self.feature_volume//2),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume//2, hidden_dim)
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(kernel_num, kernel_num//2, kernel_size=4, stride=2, padding=1),  # [batch, K/2, 8, 8]
            nn.LeakyReLU(.01),
            nn.ConvTranspose2d(kernel_num//2, kernel_num//4, kernel_size=4, stride=2, padding=1),  # [batch, K/4, 16, 16]
            nn.LeakyReLU(.01),
            nn.ConvTranspose2d(kernel_num//4, channel_num, kernel_size=4, stride=2, padding=1),  # [batch, channel_num, 32, 32]?
        )

        # projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, self.feature_volume // 2),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume // 2, self.feature_volume),
            nn.LeakyReLU(.01),
        )

        self.encoder = nn.Sequential(
            self.encoding_cnn,
            Flatten(),
            *self.encoder_linear
        )

    # def encoder(self, x):
    #     unrolled = self.encoding_cnn(x).view(len(x), -1)
    #     unrolled = self.encoder_linear(unrolled)
    #     return unrolled

    def decoder(self, z):
        rolled = self.project(z).view(len(z), -1, self.feature_size, self.feature_size)
        rolled = self.decoder_cnn(rolled)
        return rolled

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def autoencoder(self, x):
        encoded = self.encoding_cnn(x)
        decoded = self.decoder_cnn(encoded)
        return decoded


class LinearVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        data_dim = kwargs.get("data_dim", 32)
        channel_num = kwargs.get("channel_num", 1)
        kernel_num = kwargs.get("kernel_num", 100)
        hidden_dim = kwargs.get("hidden_dim", 100)

        mid_dim = 500
        self.mid_dim = mid_dim

        self.image_size = data_dim
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = hidden_dim

        # encoding layers
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
        )

        # encoded feature's size and volume
        # self.feature_size = self.image_size // 8
        # self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, hidden_dim)
        )
        self.q_logvar = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, hidden_dim)
        )

        # projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LeakyReLU(.01),
        )

        # decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(.01),
            nn.Linear(mid_dim, data_dim),
        )

        self.decoder = self.decoder_linear

    def q(self, encoded):
        return self.q_mean(encoded), self.q_logvar(encoded)


class VAE_Categorical_Base(VAE):
    def __init__(self, NUM_CATEGORIES, **kwargs):
        super().__init__(**kwargs)
        self.num_categories = NUM_CATEGORIES

    def forward(self, data_sample, **kwargs):

        (data, labels) = data_sample
        if type(labels) != type(None):
            return self.forward_labelled(data, labels, **kwargs)
        return self.forward_unlabelled(data, **kwargs)

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.zeros, self.eye)
        z = base_dist.sample((n_samps,))
        x_reconstructed, _ = self.decoder(z, labels)
        return x_reconstructed


class M2(VAE_Categorical_Base, CNN):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES,
                         channel_num=channel_num,
                         kernel_num=kernel_num)

        self.log_q_y = nn.Sequential(
            nn.Linear(self.feature_volume, self.feature_volume // 2),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume // 2, NUM_CATEGORIES)
        )
        self.proj_y = nn.Sequential(nn.Linear(NUM_CATEGORIES, hidden_dim))
        self.Wy = nn.Sequential(nn.Linear(NUM_CATEGORIES, hidden_dim))

        self.softplus = nn.Softplus()
        self.relu = nn.Tanh()

        self.apply(init_weights)

    def q(self, encoded):
        unrolled = encoded.view(len(encoded), -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled), self.log_q_y(unrolled)

    def decoder(self, z, labels):

        Wz = torch.sqrt(self.softplus(self.proj_y(labels)))
        Wyy = self.Wy(labels)
        MG = Wyy + Wz * z

        h1 = self.relu(MG)
        rolled = self.project(h1).view(len(h1), -1, self.feature_size, self.feature_size)
        rolled = self.decoder_cnn(rolled)
        return rolled, MG

    def forward_labelled(self, x, labels, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        pred_label_sm_log = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)
        x_reconstructed, pred_means = self.decoder(z, labels)

        return {"reconstructed": [x_reconstructed],
                "latent_samples": [z, pred_means],
                "q_vals": [q_mu, q_logvar, pred_label_sm_log]}

    def forward_unlabelled(self, x, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        log_q_ys = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)
        pred_means = []

        reconstructions = []

        for cat in range(self.num_categories):

            labels = torch.zeros_like(log_q_ys)
            labels[:, cat] = 1

            z = self.reparameterize(q_mu, q_logvar)
            x_reconstructed, pred_means_ = self.decoder(z, labels)

            pred_means.append(pred_means_)
            reconstructions.append(x_reconstructed)

        return {"reconstructed": reconstructions,
                "latent_samples": [z, pred_means],
                "q_vals": [q_mu, q_logvar, log_q_ys]}


class M2_Linear(LinearVAE, M2):

    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES):

        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES)
        self.log_q_y = nn.Sequential(
            nn.Linear(self.mid_dim, self.feature_volume // 2),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume // 2, NUM_CATEGORIES)
        )

    def decoder(self, z, labels):

        Wz = torch.sqrt(self.softplus(self.proj_y(labels)))
        Wyy = self.Wy(labels)
        MG = Wyy + Wz * z

        h1 = self.relu(MG)
        rolled = self.project(h1)
        rolled = self.decoder_linear(rolled)
        return rolled, MG

    def q(self, encoded):
        unrolled = encoded.view(len(encoded), -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled), self.log_q_y(unrolled)


class GMM_VAE(M2):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES,
                         channel_num=channel_num,
                         kernel_num=kernel_num)

        self.q_global_means = nn.Parameter(torch.rand(self.num_categories, self.hidden_dim))
        self.q_global_log_var = nn.Parameter(torch.zeros(self.num_categories, self.hidden_dim))

        # override unnecessary params
        # self.log_q_y = None
        self.Wy = None
        self.proj_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )

    # def q(self, encoded):
    #     unrolled = encoded.view(len(encoded), -1)
    #     return self.q_mean(unrolled), self.q_logvar(unrolled)

    def decoder(self, z):
        h1 = self.proj_y(z)
        rolled = self.project(h1).view(len(h1), -1, self.feature_size, self.feature_size)
        rolled = self.decoder_cnn(rolled)
        return rolled, None

    # def discriminator(self, q_mu, q_logvar):
    #
    #     q_mus = q_mu.unsqueeze(1).repeat(1, self.num_categories, 1)
    #     q_sigs = torch.exp(q_logvar).unsqueeze(1).repeat(1, self.num_categories, 1)
    #
    #     q_global_means = self.q_global_means.unsqueeze(0).repeat(len(q_mu), 1, 1)
    #     q_global_sigs = torch.exp(self.q_global_log_var).unsqueeze(0).repeat(len(q_mu), 1, 1)
    #
    #     base_dist = MultivariateNormal(self.zeros, self.eye)
    #
    #     return base_dist.log_prob((q_mus - q_global_means)/(q_sigs + q_global_sigs))

    def forward_labelled(self, x, labels, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        pred_label_sm_log = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        # z_global = self.reparameterize(self.q_global_means, self.q_global_log_var)
        z_mean_expanded = (labels.unsqueeze(-1) * (self.q_global_means.unsqueeze(0).repeat(len(x), 1, 1))).sum(dim=1)
        z = self.reparameterize(q_mu, q_logvar)

        x_reconstructed, _ = self.decoder(z + z_mean_expanded)

        return {"reconstructed": [x_reconstructed],
                "latent_samples": [z, None],
                "q_vals": [q_mu, q_logvar, self.q_global_means, self.q_global_log_var, pred_label_sm_log]}

    def forward_unlabelled(self, x, **kwargs):

        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        log_q_ys = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        # z_global = self.reparameterize(self.q_global_means, self.q_global_log_var)

        reconstructions = []

        for cat in range(self.num_categories):
            # labels = torch.zeros_like(log_q_ys)
            # labels[:, cat] = 1

            # q_mean_expanded = self.q_global_means[cat].unsqueeze(0).repeat(len(x), 1)
            z = self.reparameterize(q_mu, q_logvar)

            z_mean_expanded = self.q_global_means[cat].unsqueeze(0).repeat(len(x), 1)

            x_reconstructed, _ = self.decoder(z + z_mean_expanded)

            reconstructions.append(x_reconstructed)

        return {"reconstructed": reconstructions,
                "latent_samples": [z, None],
                "q_vals": [q_mu, q_logvar, self.q_global_means, self.q_global_log_var, log_q_ys]}

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.zeros, self.eye)
        z = base_dist.sample((n_samps,))
        q_mean_expanded = (labels.unsqueeze(-1) * (self.q_global_means.unsqueeze(0).repeat(len(labels), 1, 1))).sum(dim=1)
        x_reconstructed, _ = self.decoder(z + q_mean_expanded)
        return x_reconstructed

class M2_Gumbel(M2):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES,
                         channel_num=channel_num,
                         kernel_num=kernel_num)
        self.tau = 2

    def forward_unlabelled(self, x, **kwargs):
        encoded = self.encoder(x)
        (q_mu, q_logvar, log_p_y) = self.q(encoded)
        log_q_ys = log_p_y - torch.logsumexp(log_p_y, dim=1).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)

        # rather sample from the gumbel softmax here
        sampled_label = F.gumbel_softmax(log_p_y, tau=self.tau, hard=False)
        x_reconstructed, pred_means = self.decoder(z, sampled_label)

        return {"reconstructed": [x_reconstructed],
                "latent_samples": [z, pred_means, sampled_label],
                "q_vals": [q_mu, q_logvar, log_q_ys]}


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)