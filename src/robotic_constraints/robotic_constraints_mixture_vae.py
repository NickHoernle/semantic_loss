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

import numpy as np

from vaelib.vae import M2_Linear

mdl_name = 'maf'
n_cats = 3
def build_model(data_dim=10, hidden_dim=10, num_categories=n_cats):
    return M2_Linear(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        NUM_CATEGORIES=num_categories,
    )


class RC_mixture_VAE(RCTrainer):
    """
    Semi-supervised flow to infer the categories of images.
    """

    def __init__(
        self,
        input_data,
        output_data,
        backward=False,
        max_grad_norm=1,
        hidden_dim=4,
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
            "num_categories": n_cats
        }
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

    def get_means_param(self, net):
        return [net.means, net.q_log_var]


    @staticmethod
    def forward_loss(data, net, reconstructed, latent_samples, q_vals, recon_traj):
        """
        Loss forward. We marginalize out the latent categorical variable
        """
        z, pred_means = latent_samples

        q_mu, q_logvar, log_q_ys = q_vals
        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).sum()

        weights, trajectories = data

        loss_u = 0
        for cat in range(len(reconstructed)):

            pred = recon_traj[cat]
            true = trajectories.view(len(pred), -1, 2)

            error = 1e1*torch.norm(true - pred, dim=(1,2))

            log_q_y = log_q_ys[:, cat]
            q_y = torch.exp(log_q_y)

            loss_u += torch.sum(q_y * error + q_y * log_q_y)

        return loss_u.sum() + KLD_cont

    @staticmethod
    def backward_loss(y_track, constraints):
        """
        Loss for the labeled data
        """

        neg_loss = 0
        const = 1e2
        for constraint in constraints:
            neg_loss -= (const * torch.where((y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 < constraint[
                                                 'radius'] ** 2,
                                             (y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 - constraint[
                                                 'radius'] ** 2,
                                             torch.zeros_like(y_track[:, :, 0])).sum(dim=1) ** 2)

        return -neg_loss.sum()

    def sample_examples(self, epoch, net):

        num_test_samples = 100*n_cats
        labels = torch.zeros(num_test_samples, n_cats).to(self.device)
        labels[torch.arange(num_test_samples), torch.arange(n_cats).repeat(num_test_samples//n_cats)] = 1

        xs = net.sample_labelled(labels)

        # import pdb
        # pdb.set_trace()

        weights = xs.view(num_test_samples, self.data_dims // 2, -1).transpose(1, 2)

        self.plot_gen_traj(weights, epoch)

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        train_ds = NavigateFromTo(type='train', data_path=self.input_data, trajectory=True)
        valid_ds = NavigateFromTo(type='validate', data_path=self.input_data, trajectory=True)

        return train_ds, valid_ds

    def call_net(self, net, inputs):
        if len(inputs) == 2:
            return net(inputs)
        return net((inputs, None))

    def get_optimizer(self, net):
        params_ = ['means', "q_log_var"]
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in params_, net.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in params_, net.named_parameters()))))
        # return optim.Adam(net.parameters(), lr=self.lr)
        return optim.Adam([
            {"params": params,  "lr": 0.5},
            {"params": base_params}], lr=self.lr)

