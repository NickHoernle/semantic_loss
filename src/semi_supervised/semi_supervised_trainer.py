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

from torch.nn import functional as F


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
        name="base-model",
    ):
        super().__init__(
            model_builder,
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
            resume=resume,
            early_stopping_lim=early_stopping_lim,
            additional_model_config_args=additional_model_config_args,
            num_loader_workers=num_loader_workers,
            data_shuffle=False,
            name=name,
        )
        self.dataset = dataset
        self.num_labeled_data_per_class = num_labeled_data_per_class
        self.data_dims = model_parameters['data_dim']
        self.num_categories = model_parameters['num_categories']

    @torch.enable_grad()
    def train(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Train step of model returned by model_builder
        """
        device = self.device
        dims = self.data_dims

        net.train()
        loss_meter = AverageMeter()

        (train_loader_labelled_, train_loader) = loaders
        train_loader_labelled = iter(train_loader_labelled_)

        # anneal the tau parameter
        net.tau = np.max((0.5, net.tau * np.exp(-5e-3 * (epoch))))
        means, counts = torch.zeros_like(net.means), torch.zeros_like(net.means[:,0])

        with tqdm(total=len(train_loader.sampler)) as progress_bar:
            for (data_u, labels_u) in train_loader:

                (data_l, target_l) = next(train_loader_labelled)

                # prepare the data
                data_u = data_u.view(-1, dims).to(device)
                data_l = data_l.view(-1, dims).to(device)

                one_hot = convert_to_one_hot(
                    num_categories=self.num_categories, labels=target_l
                ).to(device)

                # discriminative learning
                self.get_means_param(net).requires_grad = True
                optimizer.zero_grad()

                q_mu, label_log_prob = net.encode_means(data_l)
                label_log_prob_sm = torch.log(torch.softmax(label_log_prob, dim=1) + 1e-10)

                loss = -(one_hot * label_log_prob_sm).sum(dim=1).sum()

                loss.backward()
                optimizer.step()

                # do the semi-supervised learning
                if epoch > 2:
                    self.get_means_param(net).requires_grad = False

                    optimizer.zero_grad()

                    # labeled results
                    labeled_results = net((data_l, one_hot))
                    loss_l = self.labeled_loss(data_l, *labeled_results)

                    loss_u = []

                    (q_mu, q_logvar) = net.encode(data_u)
                    label_log_prob = net.discriminator(q_mu)
                    log_pred_label_sm = torch.log(torch.softmax(label_log_prob, dim=1) + 1e-10)
                    KLD = -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - q_logvar.exp())

                    for cat in range(self.num_categories):

                        one_hot_u = convert_to_one_hot(
                            num_categories=self.num_categories, labels=cat * torch.ones(len(data_u)).long()
                        ).to(device)

                        means = (one_hot_u.unsqueeze(-1) * net.means.unsqueeze(0).repeat(len(q_mu), 1, 1)).sum(dim=1)
                        z = net.reparameterize(q_mu - means, q_logvar)

                        latent = torch.cat((z, one_hot_u), dim=1)
                        data_reconstructed = net.decode(latent)

                        BCE = F.binary_cross_entropy(
                            torch.sigmoid(data_reconstructed), data_u, reduction="sum"
                        )

                        loss_u += [((BCE+KLD) + (one_hot_u*(log_pred_label_sm)).sum(dim=1).sum()).unsqueeze(0)]

                    loss_u = torch.logsumexp(torch.cat(loss_u), dim=0)
                    # loss_u = self.unlabeled_loss(data_u, *unlabeled_results)
                    loss = loss_l + loss_u

                    loss.backward()

                    if self.max_grad_norm > 0:
                        clip_grad_norm_(net.parameters(), self.max_grad_norm)
                    optimizer.step()

                loss_meter.update(loss.item(), data_u.size(0))

                # do the semi-supervised learning
                # optimizer.zero_grad()
                #
                # labeled_results = net((data_l, one_hot))
                # unlabeled_results = net((data_u, None))
                #
                # loss_l = self.labeled_loss(data_l, *labeled_results)
                # loss_u = self.unlabeled_loss(data_u, *unlabeled_results)
                #
                # loss = loss_l + loss_u
                # loss = loss_l
                #
                # loss_meter.update(loss.item(), data_u.size(0))
                # loss.backward()

                # if self.max_grad_norm > 0:
                #     clip_grad_norm_(net.parameters(), self.max_grad_norm)
                # optimizer.step()

                progress_bar.set_postfix(
                    nll=loss_meter.avg, lr=optimizer.param_groups[0]["lr"]
                )
                progress_bar.update(data_u.size(0))
                self.global_step += data_u.size(0)

                # m, c = self.get_means(labeled_results)
                # means += m
                # counts += c

                # stochastic VI update
                # net.update_means(m, c, 1/((epoch+1)**2+100))

        return loss_meter.avg

    @torch.no_grad()
    def test(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Test step of model returned by model_builder
        """

        device = self.device
        dims = self.data_dims

        net.eval()
        loss_meter = AverageMeter()

        accuracy = []

        with tqdm(total=len(loaders.dataset)) as progress_bar:
            for data, labels in loaders:

                data = data.view(-1, dims).to(device)
                one_hot = convert_to_one_hot(
                    num_categories=self.num_categories, labels=labels
                ).to(device)

                net_args = net((data, one_hot))

                loss = self.labeled_loss(data, *net_args)

                loss_meter.update(loss.item()/len(labels), data.size(0))
                progress_bar.set_postfix(nll=loss_meter.avg)
                progress_bar.update(data.size(0))

                accuracy += (
                    (torch.argmax(net_args[2][-1], dim=1) == labels)
                    .float()
                    .detach()
                    .numpy()
                ).tolist()

        print(f"===============> Epoch {epoch}; Accuracy: {np.mean(accuracy)}")
        # print(net.means)
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

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        if self.dataset == "MNIST":
            train_ds = datasets.MNIST(
                self.input_data, train=True, download=True, transform=transforms.ToTensor()
            )
            valid_ds = datasets.MNIST(
                self.input_data, train=False, transform=transforms.ToTensor()
            )
            num_categories = 10
        elif self.dataset == "CIFAR10":
            train_ds = datasets.MNIST(
                self.input_data, train=True, download=True, transform=transforms.ToTensor()
            )
            valid_ds = datasets.MNIST(
                self.input_data, train=False, transform=transforms.ToTensor()
            )
            num_categories = 10
        elif self.dataset == "CIFAR100":
            train_ds = datasets.MNIST(
                self.input_data, train=True, download=True, transform=transforms.ToTensor()
            )
            valid_ds = datasets.MNIST(
                self.input_data, train=False, transform=transforms.ToTensor()
            )
            num_categories = 100
        else:
            raise ValueError("Dataset not in {MNIST|CIFAR10|CIFAR100}")
        return train_ds, valid_ds, num_categories

    def get_loaders(self, train_ds, valid_ds, num_categories):
        labelled_sampler, unlabelled_sampler = get_samplers(
            train_ds.train_labels.numpy(),
            n=self.num_labeled_data_per_class,
            n_categories=num_categories,
        )
        train_loader_labeled = torch.utils.data.DataLoader(
            train_ds, sampler=labelled_sampler, **self.loader_params
        )
        train_loader_unlabeled = torch.utils.data.DataLoader(
            train_ds, sampler=unlabelled_sampler, **self.loader_params
        )
        train_loader = (train_loader_labeled, train_loader_unlabeled)
        valid_loader = torch.utils.data.DataLoader(valid_ds, **self.loader_params)
        return train_loader, valid_loader
