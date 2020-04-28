"""
Author: Nick Hoernle
Define semi-supervised class for training generative models with both a labeled and an 
unlabeled loss
"""
import os

import numpy as np

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms

from utils.generative_trainer import GenerativeTrainer
from utils.logging import AverageMeter
from utils.data import get_samplers, convert_to_one_hot


class SemiSupervisedTrainer(GenerativeTrainer):
    """
    Implements a semi-supervised learning training class for MNIST, CIFAR10 and CIFAR100 data.
    """

    def __init__(
        self,
        model_builder,
        model_parameters,
        input_data,
        output_data,
        dataset="MNIST",
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
        num_labeled_data_per_class=100,
        transformer=lambda x: x,
        name="base-model",
    ):
        train_ds, valid_ds, num_categories = get_dataset(input_data, dataset)

        dims = 1
        for i in train_ds.__getitem__(0)[0].shape:
            dims *= i

        model_parameters["data_dim"] = dims
        model_parameters["num_categories"] = num_categories

        super().__init__(
            model_builder,
            model_parameters,
            (train_ds, valid_ds),
            output_data,
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
            data_shuffle=False,
            name=name,
        )

        self.dataset = dataset  # Must be one of MNIST|CIFAR10|CIFAR100'
        self.num_categories = num_categories

        labelled_sampler, unlabelled_sampler = get_samplers(
            train_ds.train_labels.numpy(),
            n=num_labeled_data_per_class,
            n_categories=num_categories,
        )
        self.train_loader_labeled = torch.utils.data.DataLoader(
            train_ds, sampler=labelled_sampler, **self.loader_params
        )
        self.train_loader_unlabeled = torch.utils.data.DataLoader(
            train_ds, sampler=unlabelled_sampler, **self.loader_params
        )
        self.train_loader = (self.train_loader_labeled, self.train_loader_unlabeled)
        self.valid_loader = torch.utils.data.DataLoader(valid_ds, **self.loader_params)

        self.transformer = transformer

    @torch.enable_grad()
    def train(self, epoch, optimizer, **kwargs):
        """
        Train step of model returned by model_builder
        """
        net = self.net
        device = self.device
        dims = self.data_dims

        net.train()
        loss_meter = AverageMeter()

        (train_loader_labelled_, train_loader) = self.train_loader
        train_loader_labelled = iter(train_loader_labelled_)

        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for data_u, labels_u in train_loader:

                optimizer.zero_grad()

                try:
                    data_l, target_l = next(train_loader_labelled)
                except StopIteration:
                    train_loader_labelled = iter(train_loader_labelled_)
                    data_l, target_l = next(train_loader_labelled)

                data_u = self.transformer(data_u)
                data_u = data_u.view(-1, dims).to(device)

                data_l = self.transformer(data_l)
                data_l = data_l.view(-1, dims).to(device)
                one_hot = convert_to_one_hot(
                    num_categories=self.num_categories, labels=target_l
                ).to(device)

                labeled_results = net((data_l, one_hot))
                unlabeled_results = net((data_u, None))

                loss_l = self.labeled_loss(*labeled_results)
                loss_u = self.unlabeled_loss(*unlabeled_results)

                loss = loss_l + loss_u

                loss_meter.update(loss.item(), data_u.size(0))
                loss.backward()
                if self.max_grad_norm > 0:
                    clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()

                progress_bar.set_postfix(
                    nll=loss_meter.avg, lr=optimizer.param_groups[0]["lr"]
                )
                progress_bar.update(data_u.size(0))
                self.global_step += data_u.size(0)
        return loss_meter.avg

    @torch.no_grad()
    def test(self, epoch, optimizer, **kwargs):
        """
        Test step of model returned by model_builder
        """

        net = self.net
        device = self.device
        dims = self.data_dims

        net.train()
        loss_meter = AverageMeter()

        accuracy = []

        with tqdm(total=len(self.valid_loader.dataset)) as progress_bar:
            for data, labels in self.valid_loader:

                data = self.transformer(data)
                data = data.view(-1, dims).to(device)
                one_hot = convert_to_one_hot(
                    num_categories=self.num_categories, labels=labels
                ).to(device)

                net_args = net((data, one_hot))

                loss = self.labeled_loss(*net_args)

                loss_meter.update(loss.item(), data.size(0))
                progress_bar.set_postfix(nll=loss_meter.avg)
                progress_bar.update(data.size(0))

                accuracy += (
                    (torch.argmax(latent_params[-1], dim=1) == labels)
                    .float()
                    .detach()
                    .numpy()
                ).tolist()

        print(f"===============> Accuracy: {np.mean(accuracy)}")
        return loss_meter.avg

    @staticmethod
    def labeled_loss(*args):
        """
        Loss for the labeled data
        """
        raise NotImplementedError("Not implemented labeled loss")

    @staticmethod
    def unlabeled_loss(*args):
        """
        Loss for the unlabeled data
        """
        raise NotImplementedError("Not implemented unlabeled loss")


def get_dataset(input_data, dataset):
    """
    Returns the dataset that is requested by the dataset param
    """
    if dataset == "MNIST":
        train_ds = datasets.MNIST(
            input_data, train=True, download=True, transform=transforms.ToTensor()
        )
        valid_ds = datasets.MNIST(
            input_data, train=False, transform=transforms.ToTensor()
        )
        num_categories = 10
    elif dataset == "CIFAR10":
        train_ds = datasets.MNIST(
            input_data, train=True, download=True, transform=transforms.ToTensor()
        )
        valid_ds = datasets.MNIST(
            input_data, train=False, transform=transforms.ToTensor()
        )
        num_categories = 10
    elif dataset == "CIFAR100":
        train_ds = datasets.MNIST(
            input_data, train=True, download=True, transform=transforms.ToTensor()
        )
        valid_ds = datasets.MNIST(
            input_data, train=False, transform=transforms.ToTensor()
        )
        num_categories = 100
    else:
        raise ValueError("Dataset not in {MNIST|CIFAR10|CIFAR100}")
    return train_ds, valid_ds, num_categories
