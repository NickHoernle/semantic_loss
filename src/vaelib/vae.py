from __future__ import print_function
import argparse
import itertools
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import weight_norm as wn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal, Uniform, \
    TransformedDistribution, SigmoidTransform, Normal, Categorical
import math

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight,0,.05)
        m.bias.data.fill_(0)


class Flatten(nn.Module):

    def __init__(self, feature_volume):
        super().__init__()
        self.feature_volume = feature_volume

    def forward(self, x):
        return x.reshape(-1, self.feature_volume)


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
        self.bounds = torch.tensor([0.9], dtype=torch.float32)

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

    def to_logits(self, x):
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
        y = (2 * x - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.eye = self.eye.to(*args, **kwargs)
        self.zeros = self.zeros.to(*args, **kwargs)
        self.bounds = self.bounds.to(*args, **kwargs)
        self.base = MultivariateNormal(self.zeros, self.eye)
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
            nn.BatchNorm2d(kernel_num//4),
            nn.ELU(True),
            nn.Dropout2d(0.15),
            nin(kernel_num//4, kernel_num//4),
            nn.ELU(True),
            nn.Dropout2d(0.15),
            nn.Conv2d(kernel_num//4, kernel_num//2, kernel_size=4, stride=2, padding=1),  # [batch, kernel_num//2, 8, 8]
            nn.BatchNorm2d(kernel_num//2),
            nn.ELU(True),
            nn.Dropout2d(0.15),
            nin(kernel_num//2, kernel_num//2),
            nn.ELU(True),
            nn.Dropout2d(0.15),
            nn.Conv2d(kernel_num//2, kernel_num, kernel_size=4, stride=2, padding=1),     # [batch, kernel_num, 4, 4]
            nn.BatchNorm2d(kernel_num),
            nn.ELU(True),
            nn.Dropout2d(0.15),
        )

        self.feature_size = self.image_size // (2 ** 3)
        self.feature_volume = kernel_num * self.feature_size * self.feature_size

        self.encoder_linear = nn.Sequential(
            nn.Linear(self.feature_volume, self.feature_volume//2), # need the div 4 due to max pool
            nn.BatchNorm1d(self.feature_volume//2),
            nn.ELU(True),
            nn.Dropout(0.15),
            nn.Linear(self.feature_volume//2, self.feature_volume//4),
            nn.ELU(True),
            nn.Dropout(0.15),
        )

        self.encoder = nn.Sequential(
            self.encoding_cnn,
            Flatten(feature_volume=self.feature_volume),
            self.encoder_linear
        )

        self.q_mean = nn.Sequential(
            nn.Linear(self.feature_volume//4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.q_logvar = nn.Sequential(
            nn.Linear(self.feature_volume//4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # num_mix = 3 if channel_num == 1 else 10
        num_mix = 10
        self.nr_logistic_mix = 5

        # projection
        self.project = nn.Sequential(
            # nn.BatchNorm1d(hidden_dim),
            # nn.ELU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(True),
            nn.Linear(hidden_dim, self.feature_volume),
            nn.BatchNorm1d(self.feature_volume),
            nn.ELU(True),
        )

        self.decoder_cnn = nn.Sequential(
            # nin(kernel_num, kernel_num),
            nn.ConvTranspose2d(kernel_num, kernel_num//2, kernel_size=4, stride=2, padding=1),  # [batch, ?, 8, 8]
            nn.BatchNorm2d(kernel_num // 2),
            nn.ELU(True),
            nin(kernel_num//2, kernel_num//2),
            nn.ELU(True),
            nn.ConvTranspose2d(kernel_num//2, kernel_num//4, kernel_size=4, stride=2, padding=1),  # [batch, ?, 16, 16]
            nn.BatchNorm2d(kernel_num // 4),
            nn.ELU(True),
            nin(kernel_num//4, kernel_num//4),
            nn.ELU(True),
            nn.ConvTranspose2d(kernel_num//4, self.channel_num, kernel_size=4, stride=2, padding=1),  # [batch, ?, 32, 32]?
            # nn.BatchNorm2d(self.channel_num, momentum=0.01),
            # nn.ConvTranspose2d(kernel_num // 4, kernel_num//8, kernel_size=4, stride=2, padding=1),
            # nn.ELU(True),
            # nin(kernel_num//8, num_mix * self.nr_logistic_mix),
            # nin(kernel_num // 8, num_mix * self.nr_logistic_mix)
        )

    def decoder(self, z):
        rolled = self.project(z).view(len(z), -1, self.feature_size, self.feature_size)
        rolled = self.decoder_cnn(rolled)
        return rolled

    def q(self, encoded):
        unrolled = encoded.view(len(encoded), -1)
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
        x_reconstructed, _ = self.decoder(z)
        x_reconstructed = sample_from_discretized_mix_logistic(x_reconstructed, self.nr_logistic_mix)
        return x_reconstructed


class M2(VAE_Categorical_Base, CNN):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES,
                         channel_num=channel_num,
                         kernel_num=kernel_num)

        self.log_q_y = nn.Sequential(
            nn.Linear(self.feature_volume//4, self.feature_volume // 2),
            nn.Dropout(0.1),
            nn.LeakyReLU(.01),
            nn.Linear(self.feature_volume // 2, NUM_CATEGORIES),
            nn.Dropout(0.1),
        )
        self.proj_y = nn.Sequential(nn.Linear(NUM_CATEGORIES, hidden_dim))
        self.Wy = nn.Sequential(nn.Linear(NUM_CATEGORIES, hidden_dim))

        self.softplus = nn.Softplus()
        self.elu = nn.ELU()

        self.apply(init_weights)

    def q(self, encoded):
        unrolled = encoded.view(len(encoded), -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled), self.log_q_y(unrolled)

    def decoder(self, z, labels):

        Wz = torch.sqrt(self.softplus(self.proj_y(labels)))
        Wyy = self.Wy(labels)
        MG = Wyy + Wz * z

        h1 = self.elu(MG)
        rolled = self.project(h1).view(len(h1), -1, self.feature_size, self.feature_size)
        rolled = self.decoder_cnn(rolled)
        # rolled = rolled.permute(0, 2, 3, 1)
        # import pdb
        # pdb.set_trace()
        out_logits = self.nin_out(F.elu(rolled))
        # import pdb
        # pdb.set_trace()
        return out_logits, MG

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

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.zeros, self.eye)
        z = base_dist.sample((n_samps,))
        x_reconstructed, _ = self.decoder(z, labels)
        x_reconstructed = sample_from_discretized_mix_logistic(x_reconstructed, self.nr_logistic_mix)
        return x_reconstructed


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


class GMM_VAE(VAE_Categorical_Base, CNN):
    def __init__(self, data_dim, hidden_dim, NUM_CATEGORIES, channel_num=1, kernel_num=150):
        super().__init__(data_dim=data_dim,
                         hidden_dim=hidden_dim,
                         NUM_CATEGORIES=NUM_CATEGORIES,
                         channel_num=channel_num,
                         kernel_num=kernel_num)

        self.q_global_means = nn.Parameter(self.num_categories*torch.rand(self.num_categories, self.hidden_dim))
        self.q_global_log_var = nn.Parameter(0*torch.ones(self.num_categories, self.hidden_dim))
        # self.log_q_y = nn.Sequential(
        #     nn.ELU(True),
        #     nn.Linear(self.feature_volume // 4, self.feature_volume // 2),
        #     nn.ELU(True),
        #     nn.Linear(self.feature_volume // 2, NUM_CATEGORIES)
        # )

        self.apply(init_weights)

    def log_q_y(self, q_mu, q_logvar):

        q_mus = q_mu.unsqueeze(1).repeat(1, self.num_categories, 1)
        q_sigs = torch.exp(q_logvar).unsqueeze(1).repeat(1, self.num_categories, 1)

        q_global_means = self.q_global_means.unsqueeze(0).repeat(len(q_mu), 1, 1)
        q_global_sigs = torch.exp(self.q_global_log_var).unsqueeze(0).repeat(len(q_mu), 1, 1)

        return self.base.log_prob((q_mus - q_global_means)/(1 + q_sigs + q_global_sigs))

    def q(self, encoded):
        unrolled = encoded.view(len(encoded), -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def forward_labelled(self, x, labels, **kwargs):

        return self.forward_unlabelled(x)

    def forward_unlabelled(self, x, **kwargs):
        encoded = self.encoder(x)
        (q_mu, q_logvar) = self.q(encoded)

        log_p_y = self.log_q_y(q_mu, q_logvar)
        log_q_ys = log_p_y - log_sum_exp(log_p_y).unsqueeze(1)

        z = self.reparameterize(q_mu, q_logvar)

        x_reconstructed = self.decoder(z)

        return {"reconstructed": [x_reconstructed],
                "latent_samples": [z, None],
                "log_p_y": log_p_y,
                "q_vals": [q_mu, q_logvar, self.q_global_means, self.q_global_log_var, log_q_ys]}

    def sample_labelled(self, labels):
        n_samps = len(labels)
        base_dist = MultivariateNormal(self.zeros, self.eye)
        z = base_dist.sample((n_samps,))
        q_mean_expanded = (labels.unsqueeze(-1) * (self.q_global_means.unsqueeze(0).repeat(len(labels), 1, 1))).sum(dim=1)
        x_reconstructed = self.decoder(z + q_mean_expanded)
        # x_reconstructed = sample_from_discretized_mix_logistic(x_reconstructed, self.nr_logistic_mix)
        return torch.sigmoid(x_reconstructed)

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


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


class nin(nn.Module):
    """
    Borrowed from https://github.com/pclucas14/pixel-cnn-pp
    """
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)
