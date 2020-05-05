"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import itertools

import torch
from torch import optim

from torchvision.utils import save_image

from nflib import (
    AffineHalfFlow,
    NormalizingFlowModel,
    BaseDistributionMixtureGaussians,
    MAF,
    Invertible1x1Conv,
    ActNorm,
    BaseDistribution,
    SS_Flow,
)

from rss_code_and_data import (DMP)
from robotic_constraints.robotic_constraints_trainer import RCTrainer
from utils.data import dequantize, to_logits
from robotic_constraints.utils import plot_trajectories


mdl_name = 'maf'

def build_model(data_dim=10, num_layers=10, num_categories=10):
    """
    Construct the model for semi-supervised learning
    """
    # base_dist = BaseDistributionMixtureGaussians(data_dim, num_categories)
    base_dist = BaseDistribution(data_dim)
    # print(num_layers)
    # print(data_dim)
    #
    flows = [MAF(dim=data_dim, parity=i % 2, nh=50) for i in range(num_layers)]
    # flows = [AffineHalfFlow(dim=data_dim, parity=i % 2, nh=50) for i in range(num_layers)]
    convs = [Invertible1x1Conv(dim=data_dim) for _ in flows]
    norms = [ActNorm(dim=data_dim) for _ in flows]

    flows = list(itertools.chain(*zip(flows, convs, norms)))

    flow_main = NormalizingFlowModel(base_dist, flows)
    return flow_main
    # return SS_Flow(flows=flow_main, NUM_CATEGORIES=num_categories, dims=data_dim)


class RC_Flow(RCTrainer):
    """
    Semi-supervised flow to infer the categories of images.
    """

    def __init__(
        self,
        input_data,
        output_data,
        backward=False,
        max_grad_norm=1,
        num_layers=10,
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
        num_labeled_data_per_class=100,
        name="vae-semi-supervised",
    ):
        model_parameters = {
            "data_dim": data_dims,
            "num_layers": num_layers,
            "num_categories": 1
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


            # def get_optimizer(self, net):
    #     params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ['means'], net.named_parameters()))))
    #     base_params = list(
    #         map(lambda x: x[1], list(filter(lambda kv: kv[0] not in ['means'], net.named_parameters()))))
    #
    #     return optim.Adam([
    #         {"params": params, "lr": 1e-1},
    #         {"params": base_params}], lr=self.lr)
        # return optim.Adam(net.parameters(), lr=self.lr)
    def get_y_tracks(self, weights):
        import numpy as np
        num_samples = len(weights)
        condition_params = torch.tensor([np.random.uniform(0, .05, num_samples),
                                         np.random.uniform(0, .05, num_samples),
                                         np.random.uniform(.95, 1., num_samples),
                                         np.random.uniform(.95, 1., num_samples)]).float().T

        dims = weights.size(1)
        dmp = DMP(weights.size()[-1] // 2, dt=1 / 100, d=2, device=self.device)
        y_track, dy_track, ddy_track = dmp.rollout_torch(condition_params[:, :2],
                                                         condition_params[:, -2:],
                                                         weights.view(num_samples, dims // 2, -1).transpose(1, 2),
                                                         device=self.device)
        return y_track

    @staticmethod
    def backward_loss(y_track, data_samp, log_det, constraints=[]):
        """
        Loss for the labeled data
        """
        neg_loss = 0
        const = 1e5
        for constraint in constraints:
            neg_loss -= (const * torch.where((y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 < constraint[
                                                 'radius'] ** 2,
                                             (y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                             (y_track[:, :, 1] - constraint['coords'][1]) ** 2 - constraint[
                                                 'radius'] ** 2,
                                             torch.zeros_like(y_track[:, :, 0])).sum(dim=1) ** 2)

        lsdj = log_det
        # prior_logprob, lsdj, label_logprob = latent_loss
        # discriminator_loss = -(true_label * (label_logprob)).sum(dim=1).sum()
        # return
        return -(neg_loss.sum() + lsdj.sum())/1e2

    @staticmethod
    def forward_loss(data, latent_samp, prior_log_prob, log_det):
        """
        Loss for the unlabeled data
        """
        # cat_kl_div = -(label_sample * (label_logprob + prior)).sum(dim=1).sum()
        return - (prior_log_prob.sum() + log_det.sum())

    def sample_examples(self, epoch, net):

        z = net.prior.sample(self.num_test_samples)
        xs, _ = net.backward(z)
        weights = xs[-1].view(self.num_test_samples, self.data_dims // 2, -1).transpose(1, 2)

        self.plot_gen_traj(weights, epoch)
