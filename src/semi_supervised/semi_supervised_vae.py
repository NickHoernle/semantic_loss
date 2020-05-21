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

from vaelib.vae import GMM_VAE
from semi_supervised.semi_supervised_trainer import SemiSupervisedTrainer


def build_model(data_dim=10, hidden_dim=10, num_categories=10, kernel_num=50, channel_num=1):
    return GMM_VAE(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        NUM_CATEGORIES=num_categories,
        kernel_num=kernel_num,
        channel_num=channel_num
    )


class VAESemiSupervisedTrainer(SemiSupervisedTrainer):
    def __init__(
        self,
        input_data,
        output_data,
        dataset="MNIST",
        max_grad_norm=1,
        hidden_dim=10,
        num_epochs=100,
        kernel_num=50,
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
        name="vae-semi-supervised",
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

    def get_optimizer(self, net):
        """
        This allows for different learning rates for means params vs other params
        """
        params_ = ['means', "q_log_var"]
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in params_, net.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in params_, net.named_parameters()))))
        # return optim.Adam(net.parameters(), lr=self.lr)
        return optim.Adam([
            {"params": params, "lr": self.lr2},
            {"params": base_params}], lr=self.lr)

    @staticmethod
    def labeled_loss(data, labels, net, reconstructed, latent_samples, q_vals):
        """
        Loss for the labeled data
        """
        data_recon = reconstructed[0]
        z, z_global = latent_samples

        q_mu, q_logvar, q_global_means, q_global_log_var, log_q_y = q_vals
        true_y = labels
        num_categories = len(log_q_y[0])

        # get the means that z should be associated with
        q_means = q_mu - (true_y.unsqueeze(-1) * z_global.unsqueeze(0).repeat(len(q_mu), 1, 1)).sum(dim=1)

        # reconstruction loss
        BCE = F.binary_cross_entropy(torch.sigmoid(data_recon), data, reduction="sum")

        # KLD for Z2
        KLD_cont = - 0.5 * ((1 + q_logvar - q_means.pow(2) - q_logvar.exp()).sum(dim=1)).sum()

        KLD_cont_main = -0.5 * torch.sum(1 + q_global_log_var - np.log(num_categories**2) -
                                         (q_global_log_var.exp() + q_global_means.pow(2)) / (num_categories**2))

        discriminator_loss = -(true_y * log_q_y).sum(dim=1).sum()

        return BCE + KLD_cont.sum() + KLD_cont_main + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, net, reconstructed, latent_samples, q_vals):
        """
        Loss for the unlabeled data
        """
        data_recon = reconstructed[0]
        z, z_global = latent_samples

        q_mu, q_logvar, q_global_means, q_global_log_var, log_q_ys = q_vals
        num_categories = len(log_q_ys[0])

        # reconstruction loss
        BCE = F.binary_cross_entropy(torch.sigmoid(data_recon), data, reduction="sum")

        # latent unlabeled loss
        loss_u = 0
        for cat in range(num_categories):

            log_q_y = log_q_ys[:, cat]
            q_y = torch.exp(log_q_y)

            q_means = z_global[cat].unsqueeze(0).repeat(len(q_mu), 1, )
            KLD_cont = - 0.5 * (1 + q_logvar - (q_mu - q_means).pow(2) - q_logvar.exp()).sum(dim=1)

            loss_u += (q_y*(KLD_cont + log_q_y)).sum()

        KLD_cont_main = -0.5 * torch.sum(1 + q_global_log_var - np.log(num_categories ** 2) -
                                         (q_global_log_var.exp() + q_global_means.pow(2)) / (num_categories ** 2))
        #
        # loss_u += BCE

        return BCE + loss_u + KLD_cont_main

    @staticmethod
    def semantic_loss(epoch, net):
        """
        Semantic loss applied to latent space
        """
        if epoch < 10:
            return 0

        n_cat = net.num_categories
        h_dim = net.hidden_dim
        base_dist = MultivariateNormal(net.zeros, net.eye)
        means = net.q_global_means.repeat(1, n_cat).view(-1, h_dim) - net.q_global_means.repeat(n_cat, 1)
        log_probs = base_dist.log_prob(means)

        return log_probs[log_probs>-20].sum()

    @staticmethod
    def simple_loss(data, reconstructed, latent_samples, q_vals):
        data_recon = reconstructed[0]
        q_mu, q_logvar, q_global_means, q_global_log_var, log_q_ys = q_vals
        BCE = F.binary_cross_entropy(torch.sigmoid(data_recon), data, reduction="sum")
        KLD_cont = - 0.5 * (1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum()
        return BCE + KLD_cont

    def sample_examples(self, epoch, net):
        labels = torch.zeros(64, self.num_categories).to(self.device)
        labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        img_sample = net.sample_labelled(labels)
        img_sample = torch.sigmoid(img_sample)
        save_image(img_sample, f'{self.figure_path}/sample_' + str(epoch) + '.png')
