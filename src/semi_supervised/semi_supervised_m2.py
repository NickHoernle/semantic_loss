"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import numpy as np

import torch
from torch.nn import functional as F
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
    ):
        model_parameters = {
            "data_dim": 32,
            "hidden_dim": hidden_dim,
            "kernel_num": kernel_num,
        }
        self.lr2 = lr2
        self.hidden_dim = hidden_dim
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

        BCE = F.binary_cross_entropy(torch.sigmoid(data_recon), data, reduction="sum")

        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).sum()

        discriminator_loss = -(true_y * log_q_y).sum(dim=1).sum()

        return BCE + KLD_cont.sum() + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, net, reconstructed, latent_samples, q_vals):
        """
        Loss for the unlabeled data
        """
        # z, pred_means = latent_samples

        q_mu, q_logvar, log_q_ys = q_vals
        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).sum()

        loss_u = 0
        for cat in range(len(reconstructed)):
            pred = torch.sigmoid(reconstructed[cat]).view(len(data), -1)
            true = data.view(len(data), -1)

            BCE = F.binary_cross_entropy(pred, true, reduction="none").sum(dim=1)
            log_q_y = log_q_ys[:, cat]
            q_y = torch.exp(log_q_y)

            loss_u += torch.sum(q_y*BCE + q_y*log_q_y)

        return loss_u + KLD_cont

    @staticmethod
    def semantic_loss(data, net, labeled_results, unlabeled_results, *args, **kwargs):
        pred_means = labeled_results['latent_samples'][1]
        log_q_y = labeled_results['q_vals'][-1]

        loss_s_l = calculate_semantic_loss(pred_means,
                                           log_q_y,
                                           hidden_dim=pred_means.size(1),
                                           num_cats=log_q_y.size(1),
                                           net=net)

        loss_s_u = 0
        for cat, pred_means in enumerate(unlabeled_results['latent_samples'][1]):

            log_q_y = unlabeled_results['q_vals'][-1]

            means_dist = calculate_semantic_loss(pred_means,
                                               log_q_y,
                                               hidden_dim=pred_means.size(1),
                                               num_cats=log_q_y.size(1),
                                               net=net)
            loss_s_u += torch.exp(log_q_y[:, cat])*means_dist

        return loss_s_l[loss_s_l > -10].sum() + loss_s_u[loss_s_u > -10].sum()


def calculate_semantic_loss(pred_means, pred_labels, hidden_dim, num_cats, net):

    means_expanded = pred_means.unsqueeze(1).repeat(1, num_cats, 1)
    labels_expanded = torch.exp(pred_labels).unsqueeze(-1).repeat(1, 1, hidden_dim)
    inf_means = (means_expanded * labels_expanded).mean(dim=0)

    base_dist = MultivariateNormal(net.zeros, net.eye)
    means = inf_means.repeat(1, num_cats).view(-1, hidden_dim) - inf_means.repeat(num_cats, 1)
    return base_dist.log_prob(means)