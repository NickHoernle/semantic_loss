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
from utils.data import get_samplers

from torchvision.utils import save_image

from torch.nn import functional as F

params = {
    "MNIST":{"num_categories": 10, "channel_num": 1},
    "CIFAR10":{"num_categories": 10, "channel_num": 3},
    "CIFAR100":{"num_categories": 100, "channel_num": 3},
}

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
        disable_tqdm_print=True,
    ):
        self.num_labeled_data_per_class = num_labeled_data_per_class
        self.dataset = dataset
        model_parameters = {**model_parameters, **params[dataset]}
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
            disable_tqdm_print=disable_tqdm_print,
        )
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
        sloss_meter = AverageMeter()

        (train_loader_labelled_, train_loader) = loaders
        train_loader_labelled = iter(train_loader_labelled_)

        # anneal the tau parameter
        # net.tau = np.max((0.5, net.tau * np.exp(-5e-3 * (epoch))))
        slosses = 0
        with tqdm(total=len(train_loader.sampler), disable=self.tqdm_print) as progress_bar:
            for i, (data_u, labels_u) in enumerate(train_loader):

                (data_l, target_l) = next(train_loader_labelled)

                # prepare the data
                data_u = data_u.to(device)
                # data_u = self.to_logits(data_u, device)
                data_l = data_l.to(device)
                # data_l = self.to_logits(data_l, device)

                target_l = target_l.to(device)

                one_hot = self.convert_to_one_hot(
                    num_categories=self.num_categories, labels=target_l
                ).to(device)

                optimizer.zero_grad()

                # if epoch == 0:
                #     ############## Warmup CNN ##################
                #     reconstruction = net.autoencoder(data_u)
                #     loss = self.autoencoder(data_u, reconstruction)
                #
                # else:

                labeled_results = net((data_l, one_hot))
                unlabeled_results = net((data_u, None))

                ############## Labeled step ################
                loss_l = self.labeled_loss(data_l, one_hot, epoch, **labeled_results)

                ############## Unlabeled step ##############
                loss_u = self.unlabeled_loss(data_u, epoch, **unlabeled_results)

                ############# Semantic Loss ################
                loss_s = self.semantic_loss(epoch, net, labeled_results, labeled_results, labels=one_hot)

                loss = loss_l + loss_s + loss_u

                sloss_meter.update(loss_s.item(), data_u.size(0))

                loss.backward()

                if self.max_grad_norm > 0:
                    clip_grad_norm_(net.parameters(), self.max_grad_norm)

                optimizer.step()

                loss_meter.update(loss.item(), data_u.size(0))

                progress_bar.set_postfix(nll=loss_meter.avg, sloss=sloss_meter.avg)
                progress_bar.update(data_u.size(0))

                self.global_step += data_u.size(0)
        print(f"===============> Epoch {epoch}; SLoss: {sloss_meter.avg}; NLL: {loss_meter.avg/(self.batch_size*32*32*3*np.log(2.))}")
        return loss_meter.avg

    @torch.no_grad()
    def test(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Test step of model returned by model_builder
        """

        return_accuracy = kwargs.get("return_accuracy", False)

        device = self.device
        dims = self.data_dims

        net.eval()
        loss_meter = AverageMeter()

        correct, total = 0.0, 0.0
        saved = False

        with tqdm(total=len(loaders.sampler), disable=self.tqdm_print) as progress_bar:
            for data, labels in loaders:

                data = data.to(device)
                # data = self.to_logits(data, device)
                labels = labels.to(device)

                one_hot = self.convert_to_one_hot(
                    num_categories=self.num_categories, labels=labels
                ).to(device)

                net_args = net((data, None))
                loss = self.unlabeled_loss(data, 51, **net_args)

                # net_args = net((data, one_hot))
                # loss = self.labeled_loss(data, one_hot, net, **net_args)

                # if not saved and not self.tqdm_print: #only save these if on local
                #     save_image(torch.sigmoid(net_args["reconstructed"][0]), f'{self.figure_path}/recon_{epoch}.png')
                #     save_image(data, f'{self.figure_path}/true_{epoch}.png')
                #     saved = True

                # loss = self.unlabeled_loss(data, net, **net_args)
                # loss = self.labeled_loss(data, net, *net_args)

                loss_meter.update(loss.item(), data.size(0))
                progress_bar.set_postfix(nll=loss_meter.avg)
                progress_bar.update(data.size(0))

                correct += (torch.argmax(net_args['q_vals'][-1], dim=1) == labels).sum().float()
                total += len(labels)

        print(f"===============> Epoch {epoch}; Accuracy: {correct/total}; NLL: {loss_meter.avg}")
        # print(net.q_global_means)
        # print(net.q_global_log_var)
        if return_accuracy:
            return loss_meter.avg, correct/total
        return loss_meter.avg

    @staticmethod
    def simple_loss(*args, **kwargs):
        """
        Loss for the labeled data
        """
        raise NotImplementedError("Not implemented labeled loss")

    @staticmethod
    def labeled_loss(*args, **kwargs):
        """
        Loss for the labeled data
        """
        raise NotImplementedError("Not implemented labeled loss")

    @staticmethod
    def unlabeled_loss(*args, **kwargs):
        """
        Loss for the unlabeled data
        """
        raise NotImplementedError("Not implemented unlabeled loss")

    def semantic_loss(self, *args, **kwargs):
        """
        Loss for the unlabeled data
        """
        return 0

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        _MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
        ]

        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        if self.dataset == "MNIST":
            TRAIN_DATASETS={'MNIST': datasets.MNIST(
                self.input_data, train=True, download=True,
                transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
            ),}
            TEST_DATASETS = {
            'MNIST': datasets.MNIST(
                self.input_data, train=False,
                transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
            ),}
        elif self.dataset == "CIFAR10":
            TRAIN_DATASETS = {'CIFAR10': datasets.CIFAR10(
                self.input_data, train=True, download=True,
                transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
            ),}
            TEST_DATASETS = {
                'CIFAR10': datasets.CIFAR10(
                self.input_data, train=False,
                transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
            ),}
        else:
            TRAIN_DATASETS = {'CIFAR100': datasets.CIFAR100(
                self.input_data, train=True, download=True,
                transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
            )}
            TEST_DATASETS = {
                'CIFAR100': datasets.CIFAR100(
                self.input_data, train=True, download=True,
                transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
            )}

        DATASET_CONFIGS = {
            'MNIST': {'size': 32, 'channels': 1, 'classes': 10},
            'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
            'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
        }

        train_ds = TRAIN_DATASETS[self.dataset]
        valid_ds = TEST_DATASETS[self.dataset]
        configs = DATASET_CONFIGS[self.dataset]
        num_categories = configs['size']

        return train_ds, valid_ds, num_categories

    def autoencoder(self, data, reconstructions):
        return F.binary_cross_entropy(torch.sigmoid(reconstructions), data, reduction="sum")

    def get_loaders(self, train_ds, valid_ds, num_categories):
        labelled_sampler, unlabelled_sampler, validation_sampler = get_samplers(
            np.array(train_ds.targets).astype(int),
            n=self.num_labeled_data_per_class,
            n_categories=num_categories,
        )
        train_loader_labeled = torch.utils.data.DataLoader(
            train_ds, sampler=labelled_sampler, **self.loader_params
        )
        train_loader_unlabeled = torch.utils.data.DataLoader(
            train_ds, sampler=unlabelled_sampler, **self.loader_params
        )
        train_loader_validation = torch.utils.data.DataLoader(
            train_ds, sampler=validation_sampler, **self.loader_params
        )
        train_loader = (train_loader_labeled, train_loader_unlabeled)
        # valid_loader = torch.utils.data.DataLoader(valid_ds, **self.loader_params)
        valid_loader = train_loader_validation

        return train_loader, valid_loader

    def convert_to_one_hot(self, num_categories, labels):
        labels = torch.unsqueeze(labels, 1)
        one_hot = torch.FloatTensor(len(labels), num_categories).zero_().to(self.device)
        one_hot.scatter_(1, labels, 1)
        return one_hot

    def to_logits(self, x, device):

        bounds = torch.tensor([0.9], dtype=torch.float32).to(device)
        y = (2 * x - 1) * bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        return y

    def sample_examples(self, epoch, net):
        labels = torch.zeros(64, self.num_categories).to(self.device)
        labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        img_sample = net.sample_labelled(labels)
        # img_sample = torch.sigmoid(img_sample)
        save_image(img_sample, f'{self.figure_path}/sample_' + str(epoch) + '.png')
