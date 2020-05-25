"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal

from vaelib.vae import M2
from semi_supervised.semi_supervised_trainer import SemiSupervisedTrainer


def build_model(data_dim=10, hidden_dim=10, num_categories=10, kernel_num=50, channel_num=1):
    return M2(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        NUM_CATEGORIES=num_categories,
        kernel_num=kernel_num,
        channel_num=channel_num
    )

class M2SemiSupervisedTrainer(SemiSupervisedTrainer):
    def __init__(
        self,
        input_data,
        output_data,
        dataset="MNIST",
        max_grad_norm=1,
        hidden_dim=10,
        kernel_num=50,
        num_epochs=100,
        s_loss=False,
        batch_size=256,
        lr2=1e-2,
        lr=1e-3,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        early_stopping_lim=50,
        additional_model_config_args=['hidden_dim', 'num_labeled_data_per_class', 'lr2'],
        num_loader_workers=8,
        num_labeled_data_per_class=100,
        name="m2",
        disable_tqdm_print=True,
    ):
        model_parameters = {
            "data_dim": 32,
            "hidden_dim": hidden_dim,
            "kernel_num": kernel_num,
        }
        self.lr2 = lr2
        self.hidden_dim = hidden_dim
        self.s_loss = s_loss

        super().__init__(
            build_model,
            model_parameters,
            input_data=input_data,
            output_data=output_data,
            dataset=dataset,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            use_cuda=use_cuda,
            num_test_samples=num_test_samples,
            seed=seed,
            gamma=gamma,
            resume=resume,
            early_stopping_lim=early_stopping_lim,
            additional_model_config_args=additional_model_config_args,
            num_loader_workers=num_loader_workers,
            num_labeled_data_per_class=num_labeled_data_per_class,
            name=name,
            disable_tqdm_print=disable_tqdm_print
        )

    def run(self):
        """
        Run the main function        
        """
        self.main()

    @staticmethod
    def labeled_loss(data, labels, net, reconstructed, latent_samples, q_vals):
        """
        Loss for the labeled data
        """
        data_recon = reconstructed[0]
        # z, pred_means = latent_samples

        q_mu, q_logvar, log_q_y = q_vals
        true_y = labels

        recon_err = discretized_mix_logistic_loss(data, data_recon)

        # recon_err = 1e1*torch.norm(data_recon - data)
        # BCE = nn.BCELoss(size_average=False)(torch.sigmoid(data_recon), data)/data.size(0)
        # TODO impoelemnt norm here
        # import pdb
        # pdb.set_trace()

        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).sum()

        discriminator_loss = -(true_y * log_q_y).sum(dim=1).sum()

        return recon_err + KLD_cont + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, net, reconstructed, latent_samples, q_vals):
        """
        Loss for the unlabeled data
        """
        # z, pred_means = latent_samples

        q_mu, q_logvar, log_q_ys = q_vals
        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).mean()

        loss_u = 0
        for cat in range(len(reconstructed)):
            pred = reconstructed[cat]
            true = data

            # TODO impoelemnt norm here
            # BCE = F.binary_cross_entropy(pred, true, reduction="none").sum(dim=(1, 2, 3))
            log_q_y = log_q_ys[:, cat]
            q_y = torch.exp(log_q_y)

            loss_u += torch.sum(q_y*BCE + q_y*log_q_y)/data.size(0)

        return loss_u + KLD_cont

    def semantic_loss(self, epoch, net, labeled_results, unlabeled_results, labels, *args, **kwargs):

        return torch.tensor(0)
        # if not self.s_loss:
        #     return torch.tensor(0)
        # # if epoch < 5:
        # #     return torch.tensor(0)
        #
        # num_cats = net.num_categories
        # idxs = np.arange(net.num_categories)
        #
        # pred_means = labeled_results['latent_samples'][1]
        #
        # means = labels.unsqueeze(-1)*pred_means.unsqueeze(1).repeat(1, num_cats, 1)
        # means = means.sum(dim=0) / labels.sum(dim=0).unsqueeze(1)
        #
        # loss_s_l = 0
        # for j in range(num_cats):
        #     distances = torch.sqrt(torch.square(means[j] - means[idxs[idxs != j]]).sum(dim=1))
        #     loss_s_l += torch.where(distances < 10, 10 - distances, torch.zeros_like(distances)).sum()
        #     idxs = idxs[1:]
        #
        # return loss_s_l
        # import pdb
        # pdb.set_trace()
        #
        # log_q_y = unlabeled_results['q_vals'][-1].unsqueeze(-1)
        # q_y = torch.exp(log_q_y)
        # pred_means = torch.cat([m.unsqueeze(1) for m in unlabeled_results['latent_samples'][1]], dim=1)
        # weighted_means = q_y * pred_means
        # weighted_means = weighted_means.sum(dim=0) / q_y.sum(dim=0).unsqueeze(1)
        #
        # loss_s_u = 0
        # for j in range(num_cats):
        #     distances = torch.sqrt(torch.square(weighted_means[j] - weighted_means[idxs[idxs != j]]).sum(dim=1))
        #     loss_s_u += torch.where(distances < 10, 10 - distances, torch.zeros_like(distances))
        #
        # import pdb
        # pdb.set_trace()

        # if epoch < 10:
        #     return epoch/1000 * (loss_s_l.sum() + loss_s_u.sum())
        # return (loss_s_l.sum() + loss_s_u.sum())


def calculate_semantic_loss(pred_means, pred_labels, hidden_dim, num_cats, net):

    means_expanded = pred_means.unsqueeze(1).repeat(1, num_cats, 1)
    labels_expanded = torch.exp(pred_labels).unsqueeze(-1).repeat(1, 1, hidden_dim)
    inf_means = (means_expanded * labels_expanded).mean(dim=0)

    base_dist = MultivariateNormal(net.zeros, net.eye)
    means = inf_means.repeat(1, num_cats).view(-1, hidden_dim) - inf_means.repeat(num_cats, 1)
    return base_dist.log_prob(means)


def discretized_mix_logistic_loss(x, l):
    """ 
    log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval 
    Borrowed from: https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]

    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
    log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))