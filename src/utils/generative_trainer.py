"""
Author: Nick Hoernle
Base class that defines the main, train, test methods that are used by the ML models in this repository. 
Each experimet will extend this class to define it's own parameters for the experiment but this class allows 
for more efficiency in the code.
"""

import os
import random
import logging

import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms

from utils.logging import raise_cuda_error
from torchvision.utils import save_image

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class GenerativeTrainer:
    """
    Base class that defines the main, train, test methods that are used by the ML models in this repository. 
    Each experimet will extend this class to define it's own parameters for the experiment but this class allows 
    for more efficiency in the code.
    """

    def __init__(
        self,
        model_builder,
        model_parameters,
        input_data,
        output_data,
        max_grad_norm=1,
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
        data_shuffle=True,
        disable_tqdm_print=True,
        name="base-model",
    ):

        # Returns a class that extends nn.Model that will be used for training. Implements .forward and .backward
        self.model_builder = model_builder
        # Parameters that are direclty related to the model init. Passed to model_builder
        self.model_parameters = model_parameters
        self.global_step = 0  # Starting step of the iteration
        self.best_loss = (
            np.inf
        )  # Starting best validation loss that is used for early stopping and logging

        self.output_data_path = output_data  # Path to where logs, models are saved
        self.input_data = input_data # path to input data

        # training parameters
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs  # Number of epochs to train
        self.batch_size = batch_size  # Batch size
        self.lr = lr  # Peak learning rate
        self.seed = seed  # Random seed for reproducibility
        self.gamma = gamma  # Learning rate step gamma (default: 0.9)
        self.resume = resume  # Resume from checkpoint
        # Early stopping implemented after N epochs with no improvement (default: 50)
        self.early_stopping_lim = early_stopping_lim
        self.additional_model_config_args = (
            additional_model_config_args  # list of config args to include in logging
        )
        self.name = name  # Basic evaluator interpretable name of model

        self.num_loader_workers = (
            num_loader_workers  # Number of workers to use for the dataset loaders
        )
        # Whether or not to shuffle the data for the loaders. Not compatible with samplers if semi-supervised training
        self.data_shuffle = data_shuffle

        self.num_test_samples = num_test_samples  # Number of samples at test time

        self.device_loaded = False
        self.device = init_device(use_cuda)
        self.tqdm_print = disable_tqdm_print

        # init loaders
        self.loader_params = {
            "batch_size": self.batch_size,
            "shuffle": self.data_shuffle,
            "num_workers": num_loader_workers,
        }
        # train_ds, valid_ds = input_torch_datasets  # tuple of train, test torch datasets

        self.network_loaded = False

        # logging
        create_folder_if_not_exists(self.output_data_path)
        self.logger_path = create_folder_if_not_exists(f"{self.output_data_path}/logs/")
        self.models_path = create_folder_if_not_exists(f"{self.output_data_path}/models/")
        self.figure_path = create_folder_if_not_exists(f"{self.output_data_path}/figures/")
        self.log_fh = open(
            f"{self.logger_path}/{self.model_name}.log", "w"
        )

        self.num_categories = 0

    @property
    def model_name(self):
        config_args = [str(getattr(self, source)) for source in self.model_config_args]
        return "_".join(config_args)

    @property
    def model_config_args(self):
        return [
            "name",
            "batch_size",
            "lr",
            "gamma",
            "seed",
        ] + self.additional_model_config_args

    @torch.enable_grad()
    def train(self, **kwargs):
        """
        Train step of model returned by model_builder
        """
        raise NotImplementedError(
            "Class must be overridden and train should be implemented"
        )

    @torch.no_grad()
    def test(self, **kwargs):
        """
        Test step of model returned by model_builder
        """
        raise NotImplementedError(
            "Class must be overridden and test should be implemented"
        )

    def get_datasets(self):
        train_ds = datasets.MNIST(
            self.input_data, train=True, download=True, transform=transforms.ToTensor()
        )
        valid_ds = datasets.MNIST(
            self.input_data, train=False, transform=transforms.ToTensor()
        )
        return train_ds, valid_ds

    def get_loaders(self, *args):
        train_loader = torch.utils.data.DataLoader(args[0], **self.loader_params)
        valid_loader = torch.utils.data.DataLoader(args[1], **self.loader_params)
        return train_loader, valid_loader

    def sample_examples(self, epoch, net):
        pass

    def get_optimizer(self, net):
        return optim.Adam(net.parameters(), lr=self.lr)

    def main(self):
        """
        Method that runs the main training and testing loop
        """

        net = self.model_builder(**self.model_parameters)
        logging.info(
            "number of params: ", sum(p.numel() for p in net.parameters())
        )
        print("number of params: ", sum(p.numel() for p in net.parameters()))

        net.to(self.device)

        train_loader, valid_loader = self.get_loaders(*self.get_datasets())
        set_seeds(self.seed)

        start_epoch = 0

        if self.resume:
            # Load checkpoint.
            logging.info("Resuming from checkpoint at save/best.pth.tar...")
            assert os.path.isdir(
                self.models_path
            ), "Error: no checkpoint directory found!"
            checkpoint = torch.load(
                os.path.join(self.models_path, f"{self.model_name}.best.pt")
            )
            net.load_state_dict(checkpoint["net"])
            best_loss = checkpoint["test_loss"]
            self.global_step = start_epoch * len(train_loader.dataset)

        optimizer = self.get_optimizer(net)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        count_valid_not_improving = 0

        for epoch in range(start_epoch, start_epoch + self.num_epochs):

            loss = self.train(epoch, net, optimizer, train_loader)
            vld_loss = self.test(epoch, net, optimizer, valid_loader)

            scheduler.step()

            if not np.isnan(vld_loss):
                self.log_fh.write(f"{epoch},{loss},{vld_loss}\n")

            # count for early stopping
            if vld_loss >= self.best_loss:
                count_valid_not_improving += 1
            else:
                count_valid_not_improving = 0

            if count_valid_not_improving > self.early_stopping_lim:
                logging.info(f"Early stopping implemented at epoch #: {epoch}")
                break

            if np.isnan(vld_loss):
                logging.info(f"Early stopping: valid loss is NAN")
                break

            # Save checkpoint
            if vld_loss < self.best_loss:
                logging.info(f'Saving...  {self.model_name}.best.pt')
                state = {
                    'net': net.state_dict(),
                    'valid_loss': vld_loss,
                    'epoch': epoch,
                }
                os.makedirs(self.models_path, exist_ok=True)
                torch.save(state, os.path.join(self.models_path, f'{self.model_name}.best.pt'))
                self.best_loss = vld_loss

            if self.num_test_samples > 0:
                with torch.no_grad():
                    self.sample_examples(epoch, net)

        self.log_fh.close()

        state_curr = {"net": net.state_dict()}
        torch.save(
            state_curr, os.path.join(self.models_path, f"{self.model_name}.final.pt")
        )


def set_seeds(seed):
    """
    Set the seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return True


def init_device(use_cuda_flag):
    """
    Get the device from input args
    """
    use_cuda = use_cuda_flag and torch.cuda.is_available()
    if use_cuda_flag and not use_cuda:
        raise_cuda_error()

    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def create_folder_if_not_exists(path):
    """
    Helper to safely create folders along the path
    """
    if not os.path.exists(path):
        logging.info(f"{path} does not exist, creating...")
        os.makedirs(path, exist_ok=True)
    return path
