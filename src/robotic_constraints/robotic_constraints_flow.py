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
    SS_Flow,
)
from robotic_constraints.robotic_constraints_trainer import RCTrainer
from utils.data import dequantize, to_logits


mdl_name = 'maf'

def build_model(data_dim=10, num_layers=10, num_categories=10):
    """
    Construct the model for semi-supervised learning
    """
    base_dist = BaseDistributionMixtureGaussians(data_dim, num_categories)

    # print(num_layers)
    # print(data_dim)
    #
    # flows = [MAF(dim=data_dim, parity=i % 2) for i in range(num_layers)]
    # convs = [Invertible1x1Conv(dim=data_dim) for _ in flows]
    # norms = [ActNorm(dim=data_dim) for _ in flows]
    #
    # flows = list(itertools.chain(*zip(flows, convs, norms)))

    flows = [AffineHalfFlow(dim=2, parity=i % 2, nh=250) for i in range(9)]

    flow_main = NormalizingFlowModel(base_dist, flows)

    return SS_Flow(flows=flow_main, NUM_CATEGORIES=num_categories, dims=data_dim)


class RC_Flow(RCTrainer):
    """
    Semi-supervised flow to infer the categories of images.
    """

    def __init__(
        self,
        input_data,
        output_data,
        dataset="MNIST",
        max_grad_norm=1,
        num_layers=5,
        num_epochs=100,
        batch_size=1000,
        lr=1e-4,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        data_dims=28*28,
        early_stopping_lim=50,
        additional_model_config_args=[],
        num_loader_workers=8,
        num_labeled_data_per_class=100,
        name="vae-semi-supervised",
    ):
        model_parameters = {
            "data_dim": data_dims,
            "num_layers": num_layers,
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

    @staticmethod
    def labeled_loss(data, latent_samp, latent_loss, true_label):
        """
        Loss for the labeled data
        """
        prior_logprob, lsdj, label_logprob = latent_loss
        discriminator_loss = -(true_label * (label_logprob)).sum(dim=1).sum()
        return prior_logprob + lsdj + discriminator_loss

    @staticmethod
    def unlabeled_loss(data, latent_samp, latent_loss, label_sample):
        """
        Loss for the unlabeled data
        """
        prior_logprob, lsdj, label_logprob, prior = latent_loss
        cat_kl_div = -(label_sample * (label_logprob + prior)).sum(dim=1).sum()
        return prior_logprob + lsdj + cat_kl_div

    def sample_examples(self, epoch, net):
        pass
        # labels = torch.zeros(64, self.num_categories).to(self.device)
        # labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        # img_sample, _ = net.sample_labelled(labels)
        # img_sample = torch.sigmoid(img_sample)
        # save_image(img_sample.view(64, 1, 28, 28), f'{self.figure_path}/sample_' + str(epoch) + '.png')