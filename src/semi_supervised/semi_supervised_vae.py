"""
Author: Nick Hoernle
Define semi-supervised class for training VAE models
"""
import torch
import torch.nn.functional as F

from vaelib.vae import RESNET_VAE
from semi_supervised.semi_supervised_trainer import SemiSupervisedTrainer

pixelwise_loss = torch.nn.L1Loss()

def build_model(hidden_dim=10, num_categories=10):
    return RESNET_VAE(
        depth=10,
        width=2,
        hidden_dim=hidden_dim,
        num_output=num_categories
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
        s_loss=False,
        s_loss_mag=5,
        lr=5e-1,
        weight_decay=0.9,
        use_cuda=True,
        num_test_samples=0,
        seed=0,
        gamma=0.9,
        resume=False,
        early_stopping_lim=200,
        additional_model_config_args=['hidden_dim', 'num_labelled'],
        num_loader_workers=8,
        num_labelled=100,
        name="gmm",
        disable_tqdm_print=True,
    ):
        model_parameters = {
            "hidden_dim": hidden_dim
        }

        self.s_loss = s_loss
        self.weight_decay = weight_decay
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
            num_labelled=num_labelled,
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
        print('creating optimizer with lr = ', self.lr)
        params = net.parameters()
        decoder_params = net.decoder_params
        encoder_params = net.encoder_params
        return [torch.optim.SGD(params, self.lr, momentum=0.9, weight_decay=self.weight_decay),
                torch.optim.SGD(encoder_params, self.lr, momentum=0.9, weight_decay=self.weight_decay),
                torch.optim.SGD(decoder_params, self.lr, momentum=0.9, weight_decay=self.weight_decay)]


    @staticmethod
    def labeled_loss(data, labels, x, z, latent_params, *args, **kwargs):
        """
        Loss for the labeled data
        """
        mu, logvar = latent_params

        recon_loss = F.cross_entropy(x, labels, reduction="mean")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + KLD


    @staticmethod
    def unlabeled_loss(data, x, z, latent_params, **kwargs):
        """
        Loss for the unlabeled data
        """
        mu, logvar = latent_params
        KLD =  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    @staticmethod
    def semantic_loss(log_predictions, all_labels):
        """
        Semantic loss applied to latent space
        """
        import pdb
        pdb.set_trace()
        return 0
