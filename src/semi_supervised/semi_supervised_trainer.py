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

from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)

from vaelib.vae import BaseNet

def build_flow(dim=10, num_layers=5):
    prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    flows = [AffineHalfFlow(dim=dim, parity=i%2, conditioning=False, num_conditioning=4) for i in range(num_layers)]
    return NormalizingFlowModel(prior, flows, conditioning=False)

from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform

prior = MultivariateNormal(torch.zeros(10), torch.eye(10))
flows = [AffineHalfFlow(dim=10, parity=i%2) for i in range(4)]
flow_model = NormalizingFlowModel(prior, flows)

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
        self.logic_net = BaseNet(10).to(self.device)
        self.logic_opt = torch.optim.SGD(self.logic_net.parameters(), lr=self.lr)

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
        opt_all, opt_encode, opt_decode = optimizer

        with tqdm(total=len(train_loader.sampler), disable=self.tqdm_print) as progress_bar:
            for i, (data_u, labels_u) in enumerate(train_loader):

                b_size = data_u.size(0)
                label = torch.full((b_size,), 1, device=device)

                (data_l, target_l) = next(train_loader_labelled)

                # prepare the data
                data_u = data_u.to(device)
                data_l = data_l.to(device)

                all_labels = torch.eye(self.num_categories).to(device)

                target_l = target_l.to(device)
                target_one_hot = self.convert_to_one_hot(num_categories=self.num_categories, labels=target_l).to(device)

                opt_all.zero_grad()
                labeled_results = net((data_l, target_l))
                log_probs = labeled_results["reconstructed"][0]
                loss = F.cross_entropy(log_probs, target_l)

                if epoch > 5:

                    labeled_results = net((data_u, None))
                    log_probs = labeled_results["reconstructed"][0]
                    predictions = F.softmax(log_probs, dim=1)
                    pred_res = self.logic_net(predictions).squeeze(dim=1)
                    loss_logic = F.binary_cross_entropy(pred_res, label)
                    loss += 0.001*loss_logic

                loss.backward()
                opt_all.step()

                # train on true labels
                self.logic_opt.zero_grad()
                true_res = self.logic(target_one_hot)
                pred_res = self.logic_net(target_one_hot).squeeze(dim=1)
                loss_logic = F.binary_cross_entropy(pred_res, true_res)
                loss_logic.backward()
                self.logic_opt.step()

                # train on predicted labels
                self.logic_opt.zero_grad()
                labeled_results = net((data_u, None))
                log_probs = labeled_results["reconstructed"][0]
                predictions = F.softmax(log_probs, dim=1)
                true_res = self.logic(predictions)
                pred_res = self.logic_net(predictions).squeeze(dim=1)
                loss_logic = F.binary_cross_entropy(pred_res, true_res)
                loss_logic.backward()
                self.logic_opt.step()

                # opt_encode.zero_grad()
                # labeled_results = net((data_l, target_l))
                # unlabeled_results = net((data_u, None))
                # loss_u = self.unlabeled_loss(data_u, epoch, **unlabeled_results)
                # loss_l = self.labeled_loss(data_l, target_l, epoch, **labeled_results)
                # loss_l = self.unlabeled_loss(data_l, epoch, **labeled_results)
                # loss = loss_u + loss_l
                # loss.backward()
                # opt_encode.step()

                # import pdb
                # pdb.set_trace()

                # opt_decode.zero_grad()
                # sample = torch.randn(len(data_u), net.hidden_dim).to(device)
                # generations = net.decode(sample)
                # loss_s = self.semantic_loss(generations, all_labels=all_labels).sum()
                # loss_s.backward()

                # if self.max_grad_norm > 0:
                #     clip_grad_norm_(net.parameters(), self.max_grad_norm)
                # opt_decode.step()

                sloss_meter.update(loss_logic.item(), data_u.size(0))
                loss_meter.update(loss.item(), data_u.size(0))

                progress_bar.set_postfix(nll=loss_meter.avg, sloss=sloss_meter.avg)
                progress_bar.update(data_u.size(0))

                self.global_step += data_u.size(0)
            
        print(f"===============> Epoch {epoch}; SLoss: {sloss_meter.avg}; NLL: {loss_meter.avg/(self.batch_size*32*32)}")
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
                labels = labels.to(device)
                all_labels = torch.eye(self.num_categories).to(device)

                net_args = net((data, labels))
                loss = F.cross_entropy(net_args['reconstructed'][0], labels)
                # loss = self.labeled_loss(data, labels, epoch, **net_args)

                loss_meter.update(loss.item(), data.size(0))
                progress_bar.set_postfix(nll=loss_meter.avg)
                progress_bar.update(data.size(0))

                correct += (torch.argmax(net_args['reconstructed'][0], dim=1) == labels).sum().float()
                total += len(labels)

        print(f"===============> Epoch {epoch}; Accuracy: {correct/total}; NLL: {loss_meter.avg/(self.batch_size*32*32)}")
        # print(net.q_global_means)
        # print(net.q_global_log_var)
        if return_accuracy:
            return loss_meter.avg, correct/total
        return loss_meter.avg

    def logic(self, predictions):
        logic = ((predictions > 0.95) | (predictions < 0.05)).all(dim=1)
        return logic.float()

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
            rescaling
        ]

        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            rescaling
        ]

        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            rescaling
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
        valid_loader = torch.utils.data.DataLoader(valid_ds, **self.loader_params)
        # valid_loader = train_loader_validation

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


def rescaling(x):
    return (x - .5) * 2.

# https://github.com/pratogab/batch-transforms

class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False, device='cpu'):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2), dtype=self.dtype, device=self.device)
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]],
                 columns[:, None]]
        return padded.permute(1, 0, 2, 3)
