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
        lr=1e-3,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        early_stopping_lim=50,
        additional_model_config_args=[],
        num_loader_workers=8,
        num_labeled_data_per_class=100,
        name="vae-semi-supervised",
    ):
        model_parameters = {
            "data_dim": 28*28,
            "hidden_dim": hidden_dim,
            "num_categories": 10
        }
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
        self.main()

    def get_means(self, results):

        recon_data, samps, latent_params, labels = results
        zs = latent_params[0]

        counts = (labels).sum(dim=0)

        return (labels.unsqueeze(-1).repeat(1, 1, zs.size(1)) * zs.unsqueeze(1).repeat(1, self.num_categories, 1)).sum(dim=0), counts

    def get_means_param(self, net):
        return net.means

    def get_decoder_params(self, net):
        for name, param in net.named_parameters():
            if 'decoder' in name:
                yield param

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

        KLD_cont = -0.5 * torch.sum(1 + q_logvar - (q_mu-z_means_).pow(2) - q_logvar.exp())
        # KLD_cont_main = -0.5 * torch.sum(1 + q_main_logvar - np.log(100) - (q_main_logvar.exp() + q_main_mu.pow(2))/100)

        discriminator_loss = -(true_label*(q_label_logprob)).sum(dim=1).sum()
        return BCE + KLD_cont.sum() + discriminator_loss
        # return BCE + KLD_cont.sum() + KLD_cont_main.sum() + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, data_reconstructed, latent_samples, latent_params, label_sample):
        """
        Loss for the unlabeled data
        """
        BCE = F.binary_cross_entropy(
            torch.sigmoid(data_reconstructed), data, reduction="sum"
        )
        z, means = latent_samples
        q_mu, q_logvar, q_label_logprob, cat_prior = latent_params

        KLD_continuous = -0.5 * torch.sum(1 + q_logvar - (z-means).pow(2) - q_logvar.exp())
        KLD_discrete = - (label_sample * (cat_prior - q_label_logprob)).sum(dim=1).sum()

        return BCE + KLD_continuous + KLD_discrete

    def get_optimizer(self, net):

        # params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ['means'], net.named_parameters()))))
        # base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in ['means'], net.named_parameters()))))
        return optim.Adam(net.parameters(), lr=self.lr)
        # return optim.Adam([
        #     {"params": params,  "lr": 1e-1},
        #     {"params": base_params}], lr=self.lr)

    def sample_examples(self, epoch, net):
        labels = torch.zeros(64, self.num_categories).to(self.device)
        labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        img_sample = net.sample_labelled(labels)
        img_sample = torch.sigmoid(img_sample)
        save_image(img_sample.view(64, 1, 28, 28), f'{self.figure_path}/sample_' + str(epoch) + '.png')