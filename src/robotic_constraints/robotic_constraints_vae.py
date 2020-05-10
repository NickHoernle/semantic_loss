"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import itertools

import torch
from torch import optim

from rss_code_and_data import (DMP)
from robotic_constraints.robotic_constraints_trainer import RCTrainer
from utils.data import dequantize, to_logits
from robotic_constraints.utils import plot_trajectories
from robotic_constraints.dataloader import NavigateFromTo

from vaelib.vae import VAE

mdl_name = 'maf'

def build_model(data_dim=10, hidden_dim=10, num_categories=10):
    return VAE(data_dim=data_dim, hidden_dim=hidden_dim)


class RC_VAE(RCTrainer):
    """
    Semi-supervised flow to infer the categories of images.
    """

    def __init__(
        self,
        input_data,
        output_data,
        backward=False,
        max_grad_norm=1,
        hidden_dim=10,
        num_epochs=100,
        batch_size=1000,
        lr=1e-4,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        data_dims=20,
        early_stopping_lim=50,
        additional_model_config_args=[],
        num_loader_workers=0,
        name="vae_rc",
    ):
        model_parameters = {
            "data_dim": data_dims,
            "hidden_dim": hidden_dim,
            "num_categories": 1
        }
        self.hidden_dim = hidden_dim
        super().__init__(
            build_model,
            model_parameters,
            input_data=input_data,
            output_data=output_data,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            use_cuda=use_cuda,
            num_test_samples=num_test_samples,
            seed=seed,
            gamma=gamma,
            backward=backward,
            resume=resume,
            early_stopping_lim=early_stopping_lim,
            additional_model_config_args=additional_model_config_args,
            num_loader_workers=num_loader_workers,
            name=name
        )

    def run(self):
        """
        Run the main function        
        """
        self.main()

    @staticmethod
    def backward_loss(y_track, weights_recon, constraints):
        """
        Loss for the labeled data
        """

        neg_loss = 0
        const = 1e10
        for constraint in constraints:
            neg_loss -= (const * torch.where((y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 < constraint[
                                                 'radius'] ** 2,
                                             (y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 - constraint[
                                                 'radius'] ** 2,
                                             torch.zeros_like(y_track[:, :, 0])).sum(dim=1) ** 2)

        return -neg_loss.mean()

    @staticmethod
    def forward_loss(data, weights_recon, latent_params, traj_recon):
        """
        Loss for the unlabeled data
        """
        weights, trajectories = data

        recon_loss = 1e2*torch.norm(trajectories - traj_recon.view(traj_recon.size(0), -1))

        q_mu, q_logvar = latent_params
        KLD_cont = -0.5 * torch.sum(1 + q_logvar - (q_mu).pow(2) - q_logvar.exp())

        # cat_kl_div = -(label_sample * (label_logprob + prior)).sum(dim=1).sum()
        return KLD_cont + recon_loss

    def sample_examples(self, epoch, net):

        z = torch.randn((self.num_test_samples, net.hidden_dim))
        xs = net.decode(z)
        weights = xs.view(self.num_test_samples, self.data_dims // 2, -1).transpose(1, 2)

        self.plot_gen_traj(weights, epoch)

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        train_ds = NavigateFromTo(type='train', data_path=self.input_data, trajectory=True)
        valid_ds = NavigateFromTo(type='validate', data_path=self.input_data, trajectory=True)

        return train_ds, valid_ds

