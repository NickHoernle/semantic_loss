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

from vaelib.vae import VAE_Categorical
from semi_supervised.semi_supervised_trainer import SemiSupervisedTrainer


def build_model(data_dim=10, hidden_dim=10, num_categories=10):
    return VAE_Categorical(
        data_dim=data_dim, hidden_dim=hidden_dim, NUM_CATEGORIES=num_categories
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
            "num_categories": 10
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
        params_ = ['means', "q_log_var"]
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in params_, net.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in params_, net.named_parameters()))))
        # return optim.Adam(net.parameters(), lr=self.lr)
        return optim.Adam([
            {"params": params, "lr": self.lr2},
            {"params": base_params}], lr=self.lr)

    def get_means(self, results):

        recon_data, samps, latent_params, labels = results
        zs = latent_params[0]

        counts = (labels).sum(dim=0)

        return (labels.unsqueeze(-1).repeat(1, 1, zs.size(1)) * zs.unsqueeze(1).repeat(1, self.num_categories, 1)).sum(dim=0), counts

    def get_means_param(self, net):
        return [net.means]

    def get_not_means_param(self, net):
        for name, param in net.named_parameters():
            if name not in ["means"]:
                yield param

    def get_decoder_params(self, net):
        for name, param in net.named_parameters():
            if 'decoder' in name:
                yield param

    def get_pred(self, net_args):
        return torch.argmax(net_args[2][-1], dim=1)

    @staticmethod
    def labeled_loss(data, data_reconstructed, latent_samples, latent_params, true_label):
        """
        Loss for the labeled data
        """

        BCE = F.binary_cross_entropy(
            torch.sigmoid(data_reconstructed), data, reduction="sum"
        )

        z, z_means = latent_samples
        q_mu, q_logvar, q_main_mu, q_main_logvar, q_label_logprob = latent_params

        z_means_ = (true_label.unsqueeze(-1) * z_means.unsqueeze(0).repeat(len(q_mu), 1, 1)).sum(dim=1)

        KLD_cont = -0.5 * ((1 + q_logvar - (q_mu-z_means_).pow(2) - q_logvar.exp()).sum(dim=1)).sum()
        # can perform full posterior inference over the means as well not just ML on these parameters.
        KLD_cont_main = -0.5 * torch.sum(1 + q_main_logvar - np.log(100) - (q_main_logvar.exp() + q_main_mu.pow(2))/100)

        discriminator_loss = -(true_label*(q_label_logprob)).sum(dim=1).sum()

        # Amazingly we don't even need the discriminator loss here
        # return BCE + KLD_cont.sum() + discriminator_loss

        return BCE + KLD_cont.sum() + KLD_cont_main.sum() + discriminator_loss
        # return BCE + KLD_cont.sum() + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, data_reconstructed, latent_samples, latent_params, num_categories, one_hot_func):
        """
        Loss for the unlabeled data
        """
        BCE = F.binary_cross_entropy(torch.sigmoid(data_reconstructed), data, reduction="sum")

        q_mu, q_logvar, net_means, net_q_log_var, log_pred_label_sm = latent_params
        z, means = latent_samples

        loss_u = 0
        for cat in range(num_categories):

            log_q_y = log_pred_label_sm[:, cat]
            q_y = torch.exp(log_q_y)

            # TODO: going to cause an issue as vector is not on target device
            ones_vector = torch.ones_like(q_y).long()
            one_hot_u = one_hot_func(num_categories=num_categories, labels=cat*ones_vector)

            # TODO: don't need the one hot. Can rather just index here.
            z_means_ = (one_hot_u.unsqueeze(-1) * means.unsqueeze(0).repeat(len(q_mu), 1, 1)).sum(dim=1)
            KLD_cont = - 0.5 * (1 + q_logvar - (q_mu - z_means_).pow(2) - q_logvar.exp()).sum(dim=1)

            # loss_u += ((q_y * KLD_cont).mean() - (q_y * log_q_y + (1 - q_y) * torch.log(1 - q_y + 1e-10)).sum())
            loss_u += ((q_y * KLD_cont).sum() + (q_y * log_q_y).sum())

        KLD_cont_main = -0.5 * torch.sum(1 + net_q_log_var - np.log(100) - (net_q_log_var.exp() + net_means.pow(2)) / 100)
        #
        # loss_u += BCE

        return BCE + loss_u + KLD_cont_main

    def sample_examples(self, epoch, net):
        labels = torch.zeros(64, self.num_categories).to(self.device)
        labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        img_sample = net.sample_labelled(labels)
        img_sample = torch.sigmoid(img_sample)
        save_image(img_sample, f'{self.figure_path}/sample_' + str(epoch) + '.png')
