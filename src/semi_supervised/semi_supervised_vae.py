"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import torch
from torch.nn import functional as F

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

    @staticmethod
    def labeled_loss(data, data_reconstructed, latent_params, label_sample):
        """
        Loss for the labeled data
        """
        BCE = F.binary_cross_entropy(
            torch.sigmoid(data_reconstructed), data, reduction="sum"
        )
        mu, logvar, label_logprob = latent_params

        KLD_continuous = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        discriminator_loss = -(label_sample * (label_logprob)).sum(dim=1).sum()

        return BCE + KLD_continuous + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, data_reconstructed, latent_params, label_sample):
        """
        Loss for the unlabeled data
        """
        BCE = F.binary_cross_entropy(
            torch.sigmoid(data_reconstructed), data, reduction="sum"
        )
        mu, logvar, label_logprob, prior = latent_params

        KLD_continuous = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD_discrete = -(label_sample * (label_logprob + prior)).sum(dim=1).sum()

        return BCE + KLD_continuous + KLD_discrete
