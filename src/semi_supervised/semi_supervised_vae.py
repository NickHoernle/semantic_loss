"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import numpy as np

import torch
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable

from torchvision.utils import save_image
from torch.distributions import MultivariateNormal

from vaelib.vae import GMM_VAE
from semi_supervised.semi_supervised_trainer import SemiSupervisedTrainer

pixelwise_loss = torch.nn.L1Loss()

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
        lr3=1e-2,
        s_loss=False,
        s_loss_mag=5,
        lr=1e-3,
        use_cuda=True,
        num_test_samples=0,
        seed=0,
        gamma=0.9,
        resume=False,
        early_stopping_lim=200,
        additional_model_config_args=['hidden_dim', 'num_labeled_data_per_class', 'lr2'],
        num_loader_workers=8,
        num_labeled_data_per_class=100,
        name="gmm",
        disable_tqdm_print=True,
    ):
        model_parameters = {
            "data_dim": 32,
            "hidden_dim": hidden_dim,
            "kernel_num": kernel_num,
        }

        self.s_loss = s_loss
        self.lr2 = lr2
        self.lr3 = lr3
        self.hidden_dim = hidden_dim
        self.s_loss_mag = s_loss_mag

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

    def get_optimizer(self, net):
        """
        This allows for different learning rates for means params vs other params
        """
        encoder_parameters = [v for k, v in net.named_parameters() if "encoder" in k]
        decoder_params = [v for k, v in net.named_parameters() if "decoder" in k]
        # return [optim.Adam(net.parameters(), lr=self.lr),
        #         optim.Adam(encoder_parameters, lr=self.lr2),
        #         optim.Adam(decoder_params, lr=self.lr3)]

        return [optim.SGD(net.parameters(), lr=self.lr),
                optim.SGD(encoder_parameters, lr=self.lr2),
                optim.SGD(decoder_params, lr=self.lr3)]

        # return [optim.SGD(net.parameters(), lr=self.lr, weight_decay=0.001, momentum=0.9),
        #         optim.SGD(encoder_parameters, lr=self.lr2, weight_decay=0.001, momentum=0.9),
        #         optim.SGD(decoder_params, lr=self.lr3, weight_decay=0.001, momentum=0.9)]

    @staticmethod
    def labeled_loss(data, labels, epoch, reconstructed, latent_samples, q_vals, **kwargs):
        """
        Loss for the labeled data
        """
        predictions = reconstructed[0]
        categorical_loss = F.cross_entropy(predictions, labels, reduction="mean")

        return categorical_loss


    @staticmethod
    def unlabeled_loss(data, epoch, reconstructed, latent_samples, q_vals, **kwargs):
        """
        Loss for the unlabeled data
        """
        q_mu, q_logvar = q_vals

        KLD_cont = - 0.5 * ((1 + q_logvar - q_mu.pow(2) - q_logvar.exp()).sum(dim=1)).mean()

        return KLD_cont

    @staticmethod
    def semantic_loss(log_predictions, all_labels):
        """
        Semantic loss applied to latent space
        """
        predictions = F.softmax(log_predictions, dim=1)
        part1 = torch.stack([predictions ** all_labels[i] for i in range(all_labels.shape[0])])
        part2 = torch.stack([(1 - predictions) ** (1 - all_labels[i]) for i in range(all_labels.shape[0])])

        return 1e-2*-torch.log(torch.sum(torch.prod(part1 * part2, dim=2), dim=0))
